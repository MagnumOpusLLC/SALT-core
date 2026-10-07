/* Maple CPU adapter. Portable engine owns the program, sampler and commits.
 * All model resources and mutable seats are initialized before READY. */
#define _POSIX_C_SOURCE 200809L
#include "salt/model.h"
#include "salt/text_verify.h"
#include "salt/attn.h"
#include "salt/int2.h"
#include "salt/kernels.h"
#include "salt/tokenizer.h"
#include "sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ML 24
#define ME 256
#define MH 2048
#define MV 151936
#define MK 512
#define MS 798720
#define MT (3 + ML * (9 + ME * 3))
extern const SaltModelDesc *salt_maple_model_descriptor(void);

typedef struct Maple {
    const SaltModelDesc *model;
    SaltTextLayerPlan plans[ML];
    SaltTextLayerExecDesc layers[ML];
    SaltTextExpertDesc experts[ML][ME];
    SaltTextKvLayerDesc kv[ML];
    SaltTextKvState state;
    SaltTensorResourceSpec resources[1 + ML * ME];
    SaltTextTensorDesc embedding, head, norm;
    SaltTextModelExecDesc descriptor;
    SaltTextVerifyProgram program;
    SaltTextVerifyCpuContext cpu;
    SaltTextVerifyExecutor executor;
    SaltTextTargetPolicy policy;
    SaltTextProjectionWindow projection;
    SaltTextTokenEpochController controller;
    SaltTextSchedulerBindings bindings;
    SaltTextScheduleStats stats;
    SaltExpertPool pool;
    SaltCache cache;
    SaltKvCache workers;
    SaltTokenizer tokenizer;
    void *trunk, *program_arena, *cpu_arena;
    size_t trunk_bytes, live_floats;
    float *live, *logits, *scratch_logits;
    int32_t *input_ids, *output_ids;
    char *input, *text;
    size_t text_capacity;
    uint32_t context, batch, width;
    uint8_t package_id[32];
    uint64_t rss_limit, peak_rss;
    int cache_ready;
} Maple;

typedef struct Projection {
    const SaltTensorDesc *tensor;
    const uint8_t *base;
    const float *input;
    float *output;
    uint32_t batch, workers;
    int failed[32];
} Projection;

static uint64_t now_ns(void *unused) {
    struct timespec t;
    (void)unused;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static int memory_gate(Maple *m) {
    SaltMemSnapshot s;
    if (salt_mem_snapshot(&s) || s.resident_b < 0) return -1;
    if ((uint64_t)s.resident_b > m->peak_rss) m->peak_rss = (uint64_t)s.resident_b;
    if (s.resident_os_peak_b > 0 && (uint64_t)s.resident_os_peak_b > m->peak_rss)
        m->peak_rss = (uint64_t)s.resident_os_peak_b;
    if ((uint64_t)s.resident_b > m->rss_limit) {
        fprintf(stderr,"MAPLE_RSS_REFUSE rss=%" PRIu64 " limit=%" PRIu64
            " anonymous=%" PRId64 " file_backed=%" PRId64 "\n",
            (uint64_t)s.resident_b,m->rss_limit,s.anonymous_b,s.file_backed_b);
        return -1;
    }
    return 0;
}
static int parallel_run(void *ctx, int workers, void (*fn)(int,void *), void *arg) {
    return salt_attn_pool_run_n(ctx,workers,fn,arg);
}
static int parallel_scope(void *ctx, int (*interpret)(void *), void *call) {
    return salt_attn_pool_team_run(ctx,interpret,call);
}
static float little_float(const uint8_t *p) {
    uint32_t u=(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
    float f; memcpy(&f,&u,4); return f;
}
typedef struct MapleNodeSeat {
    Projection projection;
    SaltInt2BatchJob jobs[256];
    int slots[256], experts[256], failed[32];
    Maple *model;
    uint32_t count, workers;
    int layer, leased, int2;
} MapleNodeSeat;
static size_t scratch_requirement(const SaltTensorDesc *t) { return t ? 0 : SIZE_MAX; }
static size_t retained_requirement(const SaltTensorDesc *t) {
    return t ? sizeof(MapleNodeSeat) : SIZE_MAX;
}
static const uint8_t *acquire_tensor(const SaltTensorDesc *t, int *slot) {
    Maple *m=(Maple *)t->binding;
    *slot=-1;
    if (t->storage.source_class == SALT_TENSOR_SOURCE_STATIC)
        return m->trunk;
    uint64_t id=t->storage.logical_resource_id;
    if (id >= ML*ME) return NULL;
    int expert=(int)(id%ME), layer=(int)(id/ME);
    if (salt_cache_getmany(&m->cache,layer,&expert,1,slot)<0) return NULL;
    return salt_cache_acquire(&m->cache,*slot,layer,expert);
}
static int release_tensor(const SaltTensorDesc *t,int slot) {
    Maple *m=(Maple *)t->binding;
    if (slot<0) return 0;
    return salt_cache_release(&m->cache,slot,
        (int)(t->storage.logical_resource_id/ME),(int)(t->storage.logical_resource_id%ME));
}
static int gather(const SaltTensorDesc *t,const int32_t *ids,uint32_t n,
                  float *out,void *scratch,size_t scratch_bytes) {
    int slot,rc=0;
    const uint8_t *base;
    (void)scratch; (void)scratch_bytes;
    if (!t || !ids || !out || !n) return -1;
    for (uint32_t b=0;b<n;b++) if (ids[b]<0 || (uint32_t)ids[b]>=t->rows) return -1;
    base=acquire_tensor(t,&slot);
    if (!base) return -1;
    for (uint32_t b=0;b<n && !rc;b++) {
        const uint8_t *p=base+t->storage.value_offset;
        float *y=out+(size_t)b*t->cols;
        if (t->storage.encoding==SALT_TENSOR_ENCODING_ROW_INT2) {
            rc=salt_int2_decode_row(p,(size_t)t->storage.value_bytes,
                base+t->storage.scale_offset,(size_t)t->storage.scale_bytes,
                (int)t->rows,(int)t->cols,ids[b],y);
        } else for (uint32_t c=0;c<t->cols;c++) {
            size_t index=(size_t)(uint32_t)ids[b]*t->cols+c;
            if (t->storage.encoding==SALT_TENSOR_ENCODING_F32) y[c]=little_float(p+4*index);
            else {
                uint32_t bits=((uint32_t)p[2*index]|(uint32_t)p[2*index+1]<<8)<<16;
                memcpy(&y[c],&bits,4);
            }
            if (!isfinite(y[c])) { rc=-1; break; }
        }
    }
    if (release_tensor(t,slot)) rc=-1;
    return rc;
}
static void projection_worker(int worker,void *opaque) {
    Projection *p=opaque;
    const SaltTensorDesc *t=p->tensor;
    int first=(int)((uint64_t)t->rows*(unsigned)worker/p->workers);
    int end=(int)((uint64_t)t->rows*((unsigned)worker+1u)/p->workers);
    const uint8_t *v=p->base+t->storage.value_offset;
    salt_kernels_set_in_expert(1);
    if (t->storage.encoding==SALT_TENSOR_ENCODING_ROW_INT2) {
        p->failed[worker]=salt_int2_matvec_batch_rows(v,p->base+t->storage.scale_offset,
            (int)t->rows,(int)t->cols,(int)p->batch,p->input,p->output,first,end);
    } else for (uint32_t b=0;b<p->batch;b++) {
        const float *x=p->input+(size_t)b*t->cols;
        float *y=p->output+(size_t)b*t->rows+first;
        if (t->storage.encoding==SALT_TENSOR_ENCODING_BF16)
            salt_bf16_matvec((const uint16_t *)(const void *)(v+(size_t)first*t->cols*2),
                end-first,(int)t->cols,x,NULL,y);
        else salt_f32_matvec((const float *)(const void *)(v+(size_t)first*t->cols*4),
                end-first,(int)t->cols,x,y);
        for (int r=0;r<end-first;r++) if (!isfinite(y[r])) p->failed[worker]=-1;
    }
    salt_kernels_set_in_expert(0);
}
static int project_batch(const SaltTensorDesc *t,const float *x,uint32_t n,
                         float *y,void *scratch,size_t bytes) {
    Maple *m=(Maple *)t->binding;
    Projection p;
    int slot,rc=0;
    (void)scratch; (void)bytes;
    memset(&p,0,sizeof p);
    p.tensor=t; p.input=x; p.output=y; p.batch=n;
    p.workers=m->width<t->rows?m->width:t->rows;
    p.base=acquire_tensor(t,&slot);
    if (!p.base || !n || !p.workers) return -1;
    if (salt_attn_pool_run_n(&m->workers,(int)p.workers,projection_worker,&p)) rc=-1;
    for (uint32_t w=0;w<p.workers;w++) if (p.failed[w]) rc=-1;
    if (release_tensor(t,slot)) rc=-1;
    return rc;
}
static int project(const SaltTensorDesc *t,const float *x,float *y,void *s,size_t n) {
    return project_batch(t,x,1,y,s,n);
}

typedef struct Int2Dispatch {
    const SaltInt2BatchJob *jobs;
    int count, workers;
    int failed[32];
} Int2Dispatch;

static void int2_dispatch_worker(int worker,void *opaque) {
    Int2Dispatch *d=opaque;
    salt_kernels_set_in_expert(1);
    d->failed[worker]=salt_int2_multi_batch_worker(d->jobs,d->count,worker,d->workers);
    salt_kernels_set_in_expert(0);
}

/* Bind the jobs already grouped/admitted by core. The existing quant row
 * operator partitions work and the existing pool executes it. This adapter
 * supplies canonical ranges and bounded leases only; it creates no work order,
 * resource identity, worker, buffer arena, output scatter or publication. */
static int project_multi(const SaltTensorHostBatch *jobs,uint32_t count,
                         void *scratch,size_t scratch_bytes) {
    SaltInt2BatchJob bound[256];
    int experts[256],slots[256];
    const uint8_t *payloads[256];
    Maple *m;
    uint32_t first=0;
    (void)scratch; (void)scratch_bytes;
    if (!jobs || count<2 || count>256 || !jobs[0].tensor ||
        !(m=(Maple *)jobs[0].tensor->binding) || m->width<1 || m->width>32)
        return -1;
    for (uint32_t j=0;j<count;j++) {
        const SaltTensorDesc *t=jobs[j].tensor;
        if (!t || t->binding!=m || t->storage.encoding!=SALT_TENSOR_ENCODING_ROW_INT2 ||
            !t->rows || t->rows>INT_MAX || !t->cols || t->cols>INT_MAX ||
            !jobs[j].row_count || jobs[j].row_count>INT_MAX ||
            !jobs[j].inputs || !jobs[j].outputs ||
            (t->storage.source_class!=SALT_TENSOR_SOURCE_STATIC &&
             t->storage.source_class!=SALT_TENSOR_SOURCE_SELECTED)) return -1;
    }
    while (first<count) {
        const SaltTensorDesc *head=jobs[first].tensor;
        int selected=head->storage.source_class==SALT_TENSOR_SOURCE_SELECTED;
        int layer=selected?(int)(head->storage.logical_resource_id/ME):-1;
        int n=0,prior=-1,rc=0;
        uint32_t end=first;
        while (end<count) {
            const SaltTensorDesc *t=jobs[end].tensor;
            if ((t->storage.source_class==SALT_TENSOR_SOURCE_SELECTED)!=selected) break;
            if (selected) {
                uint64_t id=t->storage.logical_resource_id;
                int expert=(int)(id%ME);
                if (id>=ML*ME || (int)(id/ME)!=layer) break;
                /* Core's grouped expert order is canonical; do not sort it. */
                if (expert<prior) return -1;
                if (expert!=prior) {
                    if (n==m->cache.nslot) break;
                    experts[n++]=expert;
                    prior=expert;
                }
            }
            end++;
        }
        if (end==first) return -1;
        if (selected && (salt_cache_getmany(&m->cache,layer,experts,n,slots)<0 ||
            salt_cache_acquire_many(&m->cache,slots,layer,experts,n,payloads))) return -1;
        for (uint32_t j=first;j<end;j++) {
            const SaltTensorDesc *t=jobs[j].tensor;
            const uint8_t *base=m->trunk;
            if (selected) {
                int expert=(int)(t->storage.logical_resource_id%ME);
                int slot=m->cache.resident_slot[layer*ME+expert];
                base=salt_cache_slot_for(&m->cache,slot,layer,expert);
            }
            if (!base) { rc=-1; break; }
            bound[j-first]=(SaltInt2BatchJob){base+t->storage.value_offset,
                base+t->storage.scale_offset,(int)t->rows,(int)t->cols,
                (int)jobs[j].row_count,jobs[j].inputs,jobs[j].outputs};
        }
        if (!rc) {
            Int2Dispatch d={.jobs=bound,.count=(int)(end-first),.workers=(int)m->width};
            rc=salt_attn_pool_run_n(&m->workers,d.workers,int2_dispatch_worker,&d);
            for (int w=0;w<d.workers;w++) if (d.failed[w]) rc=-1;
        }
        /* The pool call is the final-consumer completion for these leases. */
        if (selected && salt_cache_release_many(&m->cache,slots,layer,experts,n)) rc=-1;
        if (rc) return -1;
        first=end;
    }
    return 0;
}

static int retained_finish(void *opaque) {
    MapleNodeSeat *s=opaque;
    int rc=0;
    if (!s || !s->model) return -1;
    for (uint32_t w=0;w<s->workers;w++)
        if (s->failed[w] || s->projection.failed[w]) rc=-1;
    if (s->leased && salt_cache_release_many(&s->model->cache,s->slots,
            s->layer,s->experts,s->leased)) rc=-1;
    s->leased=0;
    return rc;
}

static int retained_prepare(const SaltTensorHostBatch *jobs,uint32_t count,
        uint32_t rows,uint32_t workers,void *seat,size_t bytes,SaltTensorHostNodePlan *plan) {
    MapleNodeSeat *s=seat;
    const SaltTensorDesc *head;
    int selected,n=0,prior=-1;
    const uint8_t *payloads[256];
    uint32_t end=0;
    (void)rows;
    if (!jobs || !count || !seat || bytes<sizeof *s || !plan ||
        !workers || workers>32 || !(head=jobs[0].tensor) || !head->binding) return -1;
    memset(s,0,sizeof *s); memset(plan,0,sizeof *plan);
    s->model=(Maple *)head->binding; s->workers=workers;
    selected=head->storage.source_class==SALT_TENSOR_SOURCE_SELECTED;
    s->layer=selected?(int)(head->storage.logical_resource_id/ME):-1;
    s->int2=head->storage.encoding==SALT_TENSOR_ENCODING_ROW_INT2;
    while (end<count && end<256u) {
        const SaltTensorDesc *t=jobs[end].tensor;
        if (!t || t->binding!=s->model || !jobs[end].inputs || !jobs[end].outputs ||
            !jobs[end].row_count || jobs[end].row_count>INT_MAX || !t->rows ||
            t->rows>INT_MAX || !t->cols || t->cols>INT_MAX) return -1;
        if ((t->storage.source_class==SALT_TENSOR_SOURCE_SELECTED)!=selected ||
            (t->storage.encoding==SALT_TENSOR_ENCODING_ROW_INT2)!=s->int2) break;
        if (selected) {
            uint64_t id=t->storage.logical_resource_id;
            int e=(int)(id%ME);
            if (id>=ML*ME || (int)(id/ME)!=s->layer || e<prior) return -1;
            if (e!=prior) {
                if (n==s->model->cache.nslot) break;
                s->experts[n++]=e; prior=e;
            }
        } else if (t->storage.source_class!=SALT_TENSOR_SOURCE_STATIC) return -1;
        end++;
        if (!s->int2) break;
    }
    if (!end) return -1;
    if (selected) {
        if (salt_cache_getmany(&s->model->cache,s->layer,s->experts,n,s->slots)<0 ||
            salt_cache_acquire_many(&s->model->cache,s->slots,s->layer,s->experts,n,payloads)) return -1;
        s->leased=n;
    }
    s->count=end;
    for (uint32_t j=0;j<end;j++) {
        const SaltTensorDesc *t=jobs[j].tensor;
        const uint8_t *base=s->model->trunk;
        if (selected) {
            int e=(int)(t->storage.logical_resource_id%ME);
            base=salt_cache_slot_for(&s->model->cache,
                s->model->cache.resident_slot[s->layer*ME+e],s->layer,e);
        }
        if (!base) { (void)retained_finish(s); return -1; }
        if (s->int2) {
            s->jobs[j]=(SaltInt2BatchJob){base+t->storage.value_offset,
                base+t->storage.scale_offset,(int)t->rows,(int)t->cols,
                (int)jobs[j].row_count,jobs[j].inputs,jobs[j].outputs};
        } else {
            s->workers=workers<t->rows?workers:t->rows;
            s->projection=(Projection){.tensor=t,.base=base,.input=jobs[j].inputs,
                .output=jobs[j].outputs,.batch=jobs[j].row_count,.workers=s->workers};
        }
    }
    plan->active_workers=s->workers;
    plan->consumed_jobs=end;
    return 0;
}

static int retained_worker(void *opaque,uint32_t worker) {
    MapleNodeSeat *s=opaque;
    if (!s || worker>=s->workers) return -1;
    if (s->int2) {
        salt_kernels_set_in_expert(1);
        s->failed[worker]=salt_int2_multi_batch_worker(s->jobs,(int)s->count,
                                                     (int)worker,(int)s->workers);
        salt_kernels_set_in_expert(0);
        return s->failed[worker];
    }
    projection_worker((int)worker,&s->projection);
    return s->projection.failed[worker];
}

static const SaltTensorHostNodeOps RETAINED={
    .fixed_requirement=retained_requirement,.prepare=retained_prepare,
    .worker=retained_worker,.finish=retained_finish,
};
static const SaltTensorHostOps HOST={
    .fixed_scratch_requirement=scratch_requirement,.gather_rows=gather,
    .matvec=project,.matvec_batch=project_batch,.matvec_multi_batch=project_multi,
};
static const SaltTensorHostRealizations REALIZATIONS={.cpu=&HOST};
static const SaltTensorHostOps COLLECTIVE_HOST={
    .fixed_scratch_requirement=retained_requirement,.gather_rows=gather,
    .matvec=project,.matvec_batch=project_batch,.matvec_multi_batch=project_multi,
    .retained_node=&RETAINED,
};
static const SaltTensorHostRealizations COLLECTIVE_REALIZATIONS={.cpu=&COLLECTIVE_HOST};

static SaltTensorDesc *tensor_destination(Maple *m,const char *name) {
    int l=-1,e=-1,used=0;
    char field[64];
    if (!strcmp(name,"model.embed_tokens.weight")) return &m->embedding;
    if (!strcmp(name,"lm_head.weight")) return &m->head;
    if (!strcmp(name,"model.norm.weight")) return &m->norm;
    if (sscanf(name,"model.layers.%d.mlp.experts.%d.%63s%n",&l,&e,field,&used)==3 &&
        used==(int)strlen(name) && l>=0 && l<ML && e>=0 && e<ME) {
        if (!strcmp(field,"gate_proj.weight")) return &m->experts[l][e].gate;
        if (!strcmp(field,"up_proj.weight")) return &m->experts[l][e].up;
        if (!strcmp(field,"down_proj.weight")) return &m->experts[l][e].down;
        return NULL;
    }
    used=0;
    if (sscanf(name,"model.layers.%d.%63s%n",&l,field,&used)!=2 ||
        used!=(int)strlen(name) || l<0 || l>=ML) return NULL;
#define ROLE(n,f) if (!strcmp(field,n)) return &m->layers[l].f
    ROLE("input_layernorm.weight",pre_attention_norm);
    ROLE("post_attention_layernorm.weight",pre_ffn_norm_2);
    ROLE("self_attn.q_norm.weight",q_norm); ROLE("self_attn.k_norm.weight",k_norm);
    ROLE("mlp.gate.weight",router); ROLE("self_attn.q_proj.weight",q);
    ROLE("self_attn.k_proj.weight",k); ROLE("self_attn.v_proj.weight",v);
    ROLE("self_attn.o_proj.weight",o);
#undef ROLE
    return NULL;
}
static int load_index(Maple *m,int fd) {
    FILE *f=fdopen(dup(fd),"r");
    char line[512],name[160];
    uint32_t count=0;
    if (!f) return -1;
    if (!fgets(line,sizeof line,f) || strcmp(line,"MAPLEIDX1\n")) { fclose(f); return -1; }
    while (fgets(line,sizeof line,f)) {
        unsigned enc,r,c;
        int l,e,used=0;
        uint64_t vo,so,vb,sb;
        if (sscanf(line,"%159s %u %u %u %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %n",
            name,&enc,&r,&c,&l,&e,&vo,&so,&vb,&sb,&used)!=10 ||
            line[used] || ++count>MT || r>INT_MAX || c>INT_MAX) { fclose(f); return -1; }
        SaltTensorDesc *t=tensor_destination(m,name);
        if (!t || t->stable_handle || (enc!=1 && enc!=2 && enc!=6) ||
            l < -1 || l>=ML || e < -1 || e>=ME || (e>=0 && l<0)) { fclose(f); return -1; }
        memset(t,0,sizeof *t);
        t->rows=r; t->cols=c; t->stable_handle=count; t->binding=m; t->host_ops=&REALIZATIONS;
        t->storage.encoding=(SaltTensorEncoding)enc;
        t->storage.source_class=e>=0?SALT_TENSOR_SOURCE_SELECTED:SALT_TENSOR_SOURCE_STATIC;
        t->storage.resource_kind=e>=0?SALT_TENSOR_RESOURCE_EXPERT:SALT_TENSOR_RESOURCE_TRUNK;
        t->storage.resource_id=e>=0?(uint32_t)(l*ME+e+1):0;
        t->storage.logical_resource_id=e>=0?(uint64_t)(l*ME+e):0;
        t->storage.value_offset=vo; t->storage.scale_offset=so;
        t->storage.value_bytes=vb; t->storage.scale_bytes=sb;
        t->storage.group_size=enc==6?c:0;
        if (salt_tensor_desc_validate(t) || salt_tensor_storage_resolves(&t->storage,m->resources,1+ML*ME)) {
            fclose(f); return -1;
        }
    }
    int rc=ferror(f)?-1:0;
    fclose(f);
    return count==MT?rc:-1;
}
static int execute_rows(Maple *m,const int32_t *ids,uint32_t count,int head,float *logits) {
    SaltTextExecuteAllResult r;
    if (salt_text_execute_prefill(&m->program,&m->executor,
            m->state.transition_generation+1,m->state.position,ids,count,(uint32_t)head,&r)) {
        fprintf(stderr,"MAPLE_EXECUTE_FAILURE position=%u rows=%u status=%d\n",m->state.position,count,(int)r.status);
        return -1;
    }
    if (head) memcpy(logits,r.view.canonical_base+m->program.layout.position_logits+
        (size_t)(count-1)*MV*sizeof(float),MV*sizeof(float));
    return memory_gate(m);
}
static int step_known(void *ctx,int32_t token,float *logits) { return execute_rows(ctx,&token,1,1,logits); }
static int stop_token(void *ctx,int32_t token) { (void)ctx; return token==151645; }
static int emit_token(void *ctx,uint32_t count,int32_t token,int stop) {
    (void)ctx; (void)count; (void)stop;
    printf("TOKEN %d\n",token); fflush(stdout); return 0;
}
static int publish(void *ctx,const SaltTextScheduleResult *r) {
    Maple *m=ctx;
    return m->state.position==r->result_position?0:-1;
}
static int open_runtime(Maple *m,int trunk_fd,int pool_fd,int index_fd,int tokenizer_fd,
                        uint32_t context,uint32_t batch,uint32_t workers,int slots,uint64_t rss) {
    struct stat st;
    size_t program_bytes,cpu_bytes,cursor=0;
    const char *team_env=getenv("SALT_MAPLE_CPU_TEAM");
    int team=team_env?atoi(team_env):0;
    if (team_env && strcmp(team_env,"0") && strcmp(team_env,"1") && strcmp(team_env,"2")) return -1;
    m->model=salt_maple_model_descriptor(); m->context=context; m->batch=batch;
    m->width=workers; m->rss_limit=rss;
    if (context<32 || context>4096 || !batch || batch>32 || batch>context ||
        workers<1 || workers>32 || slots<8 || slots>ML*ME || rss<100000000 ||
        fstat(trunk_fd,&st) || st.st_size<1) return -1;
    m->trunk_bytes=(size_t)st.st_size;
    m->trunk=mmap(NULL,m->trunk_bytes,PROT_READ,MAP_PRIVATE,trunk_fd,0);
    if (m->trunk==MAP_FAILED) { m->trunk=NULL; return -1; }
    m->resources[0]=(SaltTensorResourceSpec){SALT_TENSOR_RESOURCE_TRUNK,0,
        SALT_TENSOR_SOURCE_STATIC,0,m->trunk,m->trunk_bytes,m->trunk_bytes};
    /* CPU bounded-mmap needs only the authenticated fd and existing pool
     * offset table, as in the qualified model adapter. Never retain the
     * legacy flat whole-pool mapping alongside the bounded slot views. */
    uint64_t header[3];
    if (fstat(pool_fd,&st) || !S_ISREG(st.st_mode) ||
        (uint64_t)st.st_size != 24u+(uint64_t)ML*ME*MS ||
        pread(pool_fd,header,sizeof header,0)!=(ssize_t)sizeof header ||
        header[0]!=MS || header[1]!=ML || header[2]!=ME) return -1;
    m->pool.fd=dup(pool_fd);
    if (m->pool.fd<0) return -1;
    m->pool.owns_fd=1; m->pool.n_layers=ML; m->pool.n_experts=ME;
    m->pool.nbytes=MS; m->pool.map_len=(size_t)st.st_size;
    m->pool.ref=calloc(ML*ME,sizeof *m->pool.ref);
    if (!m->pool.ref) return -1;
    for (int i=0;i<ML*ME;i++)
        m->pool.ref[i]=(SaltExpertRef){24+(int64_t)i*MS,MS};
    if (salt_cache_init_bounded_mmap(&m->cache,&m->pool,slots,(int)workers)) return -1;
    m->cache_ready=1;
    for (int l=0;l<ML;l++) {
        if (salt_model_text_layer_plan(m->model,l,&m->plans[l])) return -1;
        m->layers[l].plan=&m->plans[l]; m->layers[l].kv=&m->kv[l];
        m->layers[l].experts=m->experts[l]; m->layers[l].expert_count=ME;
        uint32_t cap=l%4==3?context:(context<512?context:512);
        for (int v=0;v<2;v++) {
            SaltTextKvRowsDesc *d=v?&m->kv[l].values:&m->kv[l].keys;
            d->private_mode=l%4==3?SALT_TEXT_KV_PRIVATE_ABSOLUTE:SALT_TEXT_KV_PRIVATE_RING;
            d->private_row_capacity=cap; d->private_row_stride=MK;
            d->private_float_capacity=(size_t)cap*MK;
            cursor+=d->private_float_capacity;
        }
        for (int e=0;e<ME;e++) {
            unsigned id=(unsigned)(l*ME+e+1);
            m->resources[id]=(SaltTensorResourceSpec){SALT_TENSOR_RESOURCE_EXPERT,id,
                SALT_TENSOR_SOURCE_SELECTED,0,NULL,MS,0};
            m->experts[l][e].expert_id=(uint32_t)e;
        }
    }
    m->live_floats=cursor;
    m->live=calloc(cursor,sizeof(float));
    m->logits=calloc(MV,sizeof(float)); m->scratch_logits=calloc(MV,sizeof(float));
    m->input_ids=calloc(context,sizeof(int32_t)); m->output_ids=calloc(context,sizeof(int32_t));
    m->text_capacity=(size_t)context*64+1;
    m->input=calloc(m->text_capacity,1); m->text=calloc(m->text_capacity,1);
    if (!m->live || !m->logits || !m->scratch_logits || !m->input_ids ||
        !m->output_ids || !m->input || !m->text) return -1;
    cursor=0;
    for (int l=0;l<ML;l++) for (int v=0;v<2;v++) {
        SaltTextKvRowsDesc *d=v?&m->kv[l].values:&m->kv[l].keys;
        d->private_rows=m->live+cursor; cursor+=d->private_float_capacity;
    }
    if (load_index(m,index_fd) || salt_tokenizer_load_fd(&m->tokenizer,tokenizer_fd)) return -1;
    if (team==2) {
        m->embedding.host_ops=m->head.host_ops=m->norm.host_ops=&COLLECTIVE_REALIZATIONS;
        for (int l=0;l<ML;l++) {
            m->layers[l].q.host_ops=m->layers[l].k.host_ops=m->layers[l].v.host_ops=
                m->layers[l].o.host_ops=m->layers[l].router.host_ops=&COLLECTIVE_REALIZATIONS;
            for (int e=0;e<ME;e++)
                m->experts[l][e].gate.host_ops=m->experts[l][e].up.host_ops=
                    m->experts[l][e].down.host_ops=&COLLECTIVE_REALIZATIONS;
        }
    }
    m->descriptor=(SaltTextModelExecDesc){.model=m->model,.layers=m->layers,.layer_count=ML,
        .tensor_resources=m->resources,.tensor_resource_count=1+ML*ME,
        .vocabulary=MV,.maximum_context=context,.norm_epsilon=1e-6f,.embedding_scale=1.0f,
        .embedding=m->embedding,.final_norm=m->norm,.output_head=m->head};
    if (salt_text_verify_program_arena_requirement(&m->descriptor,batch,&program_bytes)) return -1;
    m->program_arena=calloc(1,program_bytes);
    if (!m->program_arena || salt_text_verify_program_compile(&m->program,&m->descriptor,
        &m->state,batch,m->program_arena,program_bytes)) return -1;
    m->policy=(SaltTextTargetPolicy){.execution_class=SALT_TEXT_EXECUTION_CPU_ONLY,
        .worker_budget=workers,.sequence_tiles=workers,.target_rows=1,.route_count=1,
        .queue_length=1,.candidate_count=workers,.kv_warmup_rows=1,.warm_target_rows=1,.ready=1};
    if (salt_text_verify_program_bind_target_policy(&m->program,&m->policy) ||
        salt_text_verify_cpu_arena_requirement(&m->program,&cpu_bytes)) return -1;
    m->cpu_arena=calloc(1,cpu_bytes);
    if (!m->cpu_arena || salt_text_verify_cpu_compile(&m->cpu,&m->program,m->cpu_arena,cpu_bytes) ||
        salt_text_verify_cpu_executor_init(&m->executor,&m->cpu)) return -1;
    m->workers.nthreads=(int)workers; m->workers.apool_threads=(int)workers;
    m->workers.aq4_threads=(int)workers; m->workers.max_tokens=(int)context;
    if (salt_attn_pool_init(&m->workers) ||
        salt_text_verify_cpu_parallel_bind(&m->cpu,parallel_run,&m->workers,
            m->workers.asc8,(size_t)8*context,workers)) return -1;
    if (team==1 && salt_text_verify_cpu_scope_bind(&m->cpu,parallel_scope)) return -1;
    if (team==2 && salt_text_verify_cpu_collective_bind(&m->cpu)) return -1;
    fprintf(stderr,"MAPLE_CPU_TEAM enabled=%d workers=%u scope=existing-program\n",team,workers);
    m->bindings=(SaltTextSchedulerBindings){
        .generation={.program=&m->program,.policy=&m->policy,.projection=&m->projection,
                     .context=m,.is_stop=stop_token},
        .stats=&m->stats,.context=m,.now_ns=now_ns,.step_known=step_known,
        .emit_token=emit_token,.publish=publish};
    if (salt_text_scheduler_bind(&m->controller,&m->bindings)) return -1;
    fprintf(stderr,"MAPLE_READY cpu=1 workers=%u context=%u prefill_b=%u expert_slots=%d expert_bytes=%d live_kv_bytes=%zu\n",
        workers,context,batch,slots,MS,m->live_floats*sizeof(float));
    return memory_gate(m);
}
static void close_runtime(Maple *m) {
    salt_kv_free(&m->workers);
    if (m->cache_ready) salt_cache_free(&m->cache);
    if (m->pool.owns_fd) (void)salt_pool_close(&m->pool);
    salt_tokenizer_free(&m->tokenizer);
    if (m->trunk) munmap(m->trunk,m->trunk_bytes);
    free(m->program_arena); free(m->cpu_arena); free(m->live); free(m->logits);
    free(m->scratch_logits); free(m->input_ids); free(m->output_ids); free(m->input); free(m->text);
}
static void state_hash(Maple *m,uint8_t hash[32]) {
    SaltSha256 s;
    salt_sha256_init(&s);
    salt_sha256_update(&s,"MAPLEKV1",8);
    salt_sha256_update(&s,m->package_id,32);
    salt_sha256_update(&s,&m->context,4);
    salt_sha256_update(&s,&m->state.position,4);
    salt_sha256_update(&s,m->live,m->live_floats*4);
    salt_sha256_update(&s,m->logits,MV*4);
    salt_sha256_final(&s,hash);
}
static void hex(const uint8_t *p,size_t n) { for(size_t i=0;i<n;i++) printf("%02x",p[i]); }
static int transfer(FILE *f,void *p,size_t n,int load) {
    return (load?fread(p,1,n,f):fwrite(p,1,n,f))==n?0:-1;
}
static int state_file(Maple *m,const char *path,int load) {
    uint8_t header[88]={0},hash[32];
    uint32_t position=m->state.position;
    uint64_t count=m->live_floats;
    int fd=open(path,(load?O_RDONLY:(O_WRONLY|O_CREAT|O_EXCL))|O_NOFOLLOW,0600);
    if (fd<0) return -1;
    struct stat st;
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || (load &&
        (uint64_t)st.st_size!=sizeof header+(uint64_t)m->live_floats*4+MV*4)) { close(fd); return -1; }
    FILE *f=fdopen(fd,load?"rb":"wb");
    if (!f) { close(fd); return -1; }
    if (!load) {
        memcpy(header,"MAPLEKV1",8); memcpy(header+8,m->package_id,32);
        memcpy(header+40,&m->context,4); memcpy(header+44,&position,4);
        memcpy(header+48,&count,8); state_hash(m,header+56);
    }
    int rc=transfer(f,header,sizeof header,load);
    if (load && !rc) {
        uint32_t context; uint64_t elements;
        memcpy(&context,header+40,4); memcpy(&position,header+44,4); memcpy(&elements,header+48,8);
        if (memcmp(header,"MAPLEKV1",8) || memcmp(header+8,m->package_id,32) ||
            context!=m->context || elements!=m->live_floats || position>=context) rc=-1;
    }
    if (!rc) rc=transfer(f,m->live,m->live_floats*4,load);
    if (!rc) rc=transfer(f,m->logits,MV*4,load);
    if (!load && !rc && (fflush(f) || fsync(fd))) rc=-1;
    if (fclose(f)) rc=-1;
    if (load && !rc) {
        m->state.position=position; state_hash(m,hash);
        if (memcmp(hash,header+56,32)) rc=-1;
        m->state.transition_generation=0;
        memset(&m->projection,0,sizeof m->projection);
    }
    /* Any failed import is fatal: partially read bytes must never be used. */
    return rc;
}
static int request(Maple *m,size_t bytes,uint32_t output,int proof,float temp,uint64_t seed,uint32_t topk,float repetition,float topp) {
    int n=salt_tokenizer_encode(&m->tokenizer,m->input,(int *)m->input_ids,(int)m->context);
    if (n<1 || output<1 || (uint64_t)m->state.position+(unsigned)n+output+1>m->context) {
        puts("REFUSE context-or-tokenizer"); fflush(stdout); return 0;
    }
    (void)bytes;
    fprintf(stderr,"MAPLE_SAMPLER temperature=%.9g top_k=%u seed=%" PRIu64
        " repetition_penalty=%.9g window=64 policy=divide-per-occurrence history=request-generated top_p=%.9g\n",
        (double)temp,topk,seed,(double)repetition,(double)topp);
    uint32_t before=m->state.position;
    uint64_t begin=now_ns(NULL),prefill_done;
    for (uint32_t at=0;at<(uint32_t)n;) {
        uint32_t b=(uint32_t)n-at;
        if (b>m->batch) b=m->batch;
        if (execute_rows(m,m->input_ids+at,b,at+b==(uint32_t)n,m->logits)) return -1;
        at+=b;
    }
    prefill_done=now_ns(NULL);
    SaltTextScheduleRequest q={.sampler={temp==0?SALT_SAMPLER_GREEDY_V1:SALT_SAMPLER_TEMPERATURE_COUNTER_V1,temp,seed,topk},
        .logits=m->logits,.scratch_logits=m->scratch_logits,.output_ids=m->output_ids,
        .output_limit=output,.close_token=151645,.history_ids=m->input_ids,.history_count=(uint32_t)n,
        .repetition_penalty=repetition,.top_p=topp};
    SaltTextScheduleResult r;
    if (salt_text_scheduler_run(&m->controller,&q,&r) ||
        salt_text_scheduler_finish(&m->controller,&q,&r) ||
        salt_text_scheduler_publish(&m->controller,&r)) return -1;
    uint64_t ended=now_ns(NULL);
    uint32_t visible=r.output_count;
    if (visible && m->output_ids[visible-1]==151645) visible--;
    int text_bytes=salt_tokenizer_decode(&m->tokenizer,(int *)m->output_ids,(int)visible,m->text,(int)m->text_capacity);
    if (text_bytes<0 || memory_gate(m)) return -1;
    uint8_t hash[32]={0};
    if (proof) state_hash(m,hash);
    printf("DONE %u %u %u %u %" PRIu64 " %.6f %.6f ",before,(unsigned)n,r.output_count,m->state.position,m->peak_rss,
        (double)(prefill_done-begin)/1e6/n,(double)(ended-prefill_done)/1e6/r.output_count);
    hex(hash,32); printf(" "); hex((uint8_t *)m->text,(size_t)text_bytes); puts(""); fflush(stdout);
    return 0;
}
int main(int argc,char **argv) {
    /* Internal descriptor-bound protocol. HTTP callers never select a model path. */
    if (argc!=11) { fprintf(stderr,"usage: maple-server TRUNK_FD POOL_FD INDEX_FD TOKENIZER_FD PACKAGE_SHA CTX B W SLOTS RSS\n"); return 2; }
    uint32_t endian=1;
    if (*(uint8_t *)&endian!=1 || sizeof(float)!=4) return 2;
    Maple *m=calloc(1,sizeof *m);
    if (!m) return 2;
    uint32_t ctx=(uint32_t)strtoul(argv[6],NULL,10), b=(uint32_t)strtoul(argv[7],NULL,10),w=(uint32_t)strtoul(argv[8],NULL,10);
    if (strlen(argv[5])!=64 || salt_sha256_hex_parse(argv[5],m->package_id) ||
        open_runtime(m,atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),ctx,b,w,atoi(argv[9]),strtoull(argv[10],NULL,10))) {
        fprintf(stderr,"MAPLE_STARTUP_FAILURE\n"); close_runtime(m); free(m); return 2;
    }
    puts("READY"); fflush(stdout);
    char line[4352];
    int rc=0;
    while (fgets(line,sizeof line,stdin)) {
        if (!strcmp(line,"QUIT\n")) break;
        if (!strcmp(line,"RESET\n")) {
            memset(m->live,0,m->live_floats*4); memset(m->logits,0,MV*4);
            memset(&m->state,0,sizeof m->state); memset(&m->projection,0,sizeof m->projection);
            puts("OK"); fflush(stdout); continue;
        }
        if (!strncmp(line,"SAVE ",5) || !strncmp(line,"LOAD ",5)) {
            line[strcspn(line,"\r\n")]=0;
            if (state_file(m,line+5,line[0]=='L')) { rc=1; break; }
            puts("OK"); fflush(stdout); continue;
        }
        size_t bytes; unsigned output,proof,topk; float temp,repetition=1.0f,topp=1.0f; uint64_t seed;
        int consumed=0;
        int fields=sscanf(line,"RUN %zu %u %u %f %" SCNu64 " %u %f %f %n",
            &bytes,&output,&proof,&temp,&seed,&topk,&repetition,&topp,&consumed);
        /* Preserve the six/seven-field internal commands as neutral top-p. */
        if (fields==7) {
            topp=1.0f;
            fields=sscanf(line,"RUN %zu %u %u %f %" SCNu64 " %u %f %n",
                &bytes,&output,&proof,&temp,&seed,&topk,&repetition,&consumed);
        }
        if (fields==6) {
            repetition=1.0f;
            fields=sscanf(line,"RUN %zu %u %u %f %" SCNu64 " %u %n",
                &bytes,&output,&proof,&temp,&seed,&topk,&consumed);
        }
        if ((fields!=6 && fields!=7 && fields!=8) || !consumed || line[consumed] ||
            bytes<1 || bytes>=m->text_capacity || proof>1 || !isfinite(temp) || temp<0 ||
            !isfinite(repetition) || repetition<1.0f || repetition>2.0f ||
            !isfinite(topp) || topp<=0.0f || topp>1.0f ||
            (repetition>1.0f && temp==0.0f) ||
            topk>256 || (temp>0 && topk==0) || fread(m->input,1,bytes,stdin)!=bytes ||
            memchr(m->input,0,bytes)) { rc=1; break; }
        m->input[bytes]=0;
        if (request(m,bytes,output,(int)proof,temp,seed,topk,repetition,topp)) { rc=1; break; }
    }
    if (rc) { puts("FATAL"); fflush(stdout); }
    close_runtime(m); free(m); return rc;
}
