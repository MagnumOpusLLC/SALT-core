#include "lfm_gpu.h"
#include "lfm.h"
#include "lfm_attention.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

#ifndef LFM_METAL_TILED_ATTENTION
#define LFM_METAL_TILED_ATTENTION 0
#endif
#ifndef LFM_METAL_Q6_WARP
#define LFM_METAL_Q6_WARP 0
#endif

typedef struct {
    id<MTLComputePipelineState> conv;
    id<MTLComputePipelineState> q6;
    id<MTLComputePipelineState> q6_warp;
    id<MTLComputePipelineState> f32;
    id<MTLComputePipelineState> bf16;
    id<MTLComputePipelineState> activation;
    id<MTLComputePipelineState> router_scores;
    id<MTLComputePipelineState> router;
    id<MTLComputePipelineState> route_complete;
    id<MTLComputePipelineState> routed_gather;
    id<MTLComputePipelineState> embedding_q6;
    id<MTLComputePipelineState> attention_scores;
    id<MTLComputePipelineState> attention_reduce;
    LfmAttentionIoLedger attention;
} LfmMetalState;

typedef struct {
    uint64_t scratch_float;
    uint64_t state_float;
    uint64_t kernel_byte;
    uint32_t rows;
    uint32_t hidden;
    uint32_t kernel_size;
} LfmMetalConvArgs;

typedef struct {
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t jobs;
} LfmMetalQ6Args;

typedef struct {
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint32_t vocabulary;
    uint32_t width;
    uint32_t tokens;
    float scale;
} LfmMetalEmbeddingArgs;

typedef struct {
    uint32_t maximum_context;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t heads_per_kv;
    uint32_t rows;
    uint32_t wave_first;
    uint32_t wave_rows;
    uint32_t query_width;
    uint32_t kv_width;
    uint32_t kv_ring;
    uint32_t kv_capacity;
    uint32_t score_workers;
    float score_scale;
} LfmMetalAttentionArgs;

static const char *const LFM_METAL_SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "inline void two_sum(float a,float b,thread float &x,thread float &y){x=a+b;float bv=x-a;y=(a-(x-bv))+(b-bv);}\n"
    "inline uint grow(thread const float *e,uint n,float b,thread float *h){float q=b;uint m=0;for(uint i=0;i<n;i++){float s,hh;two_sum(q,e[i],s,hh);if(hh!=0.0f)h[m++]=hh;q=s;}if(q!=0.0f||m==0u)h[m++]=q;return m;}\n"
    "inline uint add_term(thread float *e,uint n,float b){float h[16];uint m=grow(e,n,b,h);for(uint i=0;i<m;i++)e[i]=h[i];return m;}\n"
    "inline uint add_prod(thread float *e,uint n,float a,float b){float p=a*b;float er=fma(a,b,-p);n=add_term(e,n,er);return add_term(e,n,p);}\n"
    "inline float expansion_estimate(thread const float *e,uint n){float q=0.0f;for(uint i=0;i<n;i++)q+=e[i];return q;}\n"
    "inline int expansion_round_int(thread const float *e,uint n){float q=e[n-1u];float fl=floor(q);if(q-fl==0.5f){float tail=0.0f;for(int i=int(n)-2;i>=0;i--)if(e[i]!=0.0f){tail=e[i];break;}if(tail<0.0f)return int(fl);if(tail>0.0f)return int(fl+1.0f);return q<0.0f?int(fl):int(fl+1.0f);}return int(round(q));}\n"
    "inline float staged_madd(float a,float b,float c){float p=a*b;return p+c;}\n"
    "inline float salt_exp(float x){if(isnan(x))return x;if(x>88.722839f)return 3.402823466e38f;if(x< -87.336548f)return 0.0f;float e[16];uint m=0;m=add_prod(e,m,x,0x1.715476p+0f);m=add_prod(e,m,x,0x1.4ae0cp-26f);m=add_prod(e,m,x,-0x1p-51f);int n=expansion_round_int(e,m);m=0;m=add_term(e,m,x);m=add_prod(e,m,-float(n),0x1.62e43p-1f);m=add_prod(e,m,-float(n),-0x1.05c61p-29f);m=add_prod(e,m,-float(n),-0x1p-53f);float r=expansion_estimate(e,m);float p=2.48015873015873e-05f;p=staged_madd(p,r,1.98412698412698e-04f);p=staged_madd(p,r,1.38888888888889e-03f);p=staged_madd(p,r,8.33333333333333e-03f);p=staged_madd(p,r,4.16666666666667e-02f);p=staged_madd(p,r,1.66666666666667e-01f);p=staged_madd(p,r,0.5f);p=staged_madd(p,r,1.0f);p=staged_madd(p,r,1.0f);return ldexp(p,n);}\n"
    "inline float bf16(device const uchar *p){return as_type<float>((uint(p[0])|(uint(p[1])<<8u))<<16u);}\n"
    "struct A{ulong scratch_float,state_float,kernel_byte;uint rows,hidden,kernel_size;};\n"
    "kernel void lfm_short_conv(device float *canonical[[buffer(0)]],device const uchar *weights[[buffer(1)]],constant A &a[[buffer(2)]],uint ch[[thread_position_in_grid]]){if(ch>=a.hidden)return;device float *win=canonical+a.state_float+ulong(ch)*a.kernel_size;for(uint row=0;row<a.rows;row++){ulong base=a.scratch_float+ulong(row)*3u*a.hidden;float b=canonical[base+ch],c=canonical[base+a.hidden+ch],x=canonical[base+2u*a.hidden+ch];for(uint j=0;j+1u<a.kernel_size;j++)win[j]=win[j+1u];win[a.kernel_size-1u]=b*x;float acc=0.0f;for(uint j=0;j<a.kernel_size;j++){float product=bf16(weights+a.kernel_byte+(ulong(ch)*a.kernel_size+j)*2u)*win[j];acc+=product;}canonical[a.scratch_float+ulong(a.rows)*3u*a.hidden+ulong(row)*a.hidden+ch]=c*acc;}}\n"
    "struct Q{ulong value_offset,scale_offset,bias_offset;uint rows,cols,jobs;};\n"
    "kernel void lfm_f32_project(device const uchar *weights[[buffer(0)]],device const float *x[[buffer(1)]],device float *y[[buffer(2)]],constant Q &a[[buffer(3)]],uint index[[thread_position_in_grid]]){uint job=index/a.rows,row=index%a.rows;if(job>=a.jobs)return;device const float *w=reinterpret_cast<device const float *>(weights+a.value_offset)+ulong(row)*a.cols;float acc=0.0f;for(uint col=0;col<a.cols;col++){float product=w[col]*x[ulong(job)*a.cols+col];acc+=product;}y[ulong(job)*a.rows+row]=acc;}\n"
    "kernel void lfm_bf16_project(device const uchar *weights[[buffer(0)]],device const float *x[[buffer(1)]],device float *y[[buffer(2)]],constant Q &a[[buffer(3)]],uint index[[thread_position_in_grid]]){uint job=index/a.rows,row=index%a.rows;if(job>=a.jobs)return;device const uchar *w=weights+a.value_offset+ulong(row)*a.cols*2u;float acc=0.0f;for(uint col=0;col<a.cols;col++){float product=bf16(w+ulong(col)*2u)*x[ulong(job)*a.cols+col];acc+=product;}y[ulong(job)*a.rows+row]=acc;}\n"
    "kernel void lfm_q6_project(device const uchar *weights[[buffer(0)]],device const float *x[[buffer(1)]],device float *y[[buffer(2)]],constant Q &a[[buffer(3)]],uint index[[thread_position_in_grid]]){uint job=index/a.rows,row=index%a.rows;if(job>=a.jobs)return;float acc=0.0f;ulong rowbase=ulong(row)*a.cols;for(uint group_base=0;group_base<a.cols;group_base+=64u){ulong group=(rowbase+group_base)/64u;float scale=bf16(weights+a.scale_offset+group*2u),bias=bf16(weights+a.bias_offset+group*2u);uint stop=min(group_base+64u,a.cols);for(uint col=group_base;col<stop;col++){ulong p=rowbase+col,bit=p*6u,byte=bit>>3u;uint packed=uint(weights[a.value_offset+byte]);if((bit&7u)>2u)packed|=uint(weights[a.value_offset+byte+1u])<<8u;uint code=(packed>>uint(bit&7u))&63u;float input=x[ulong(job)*a.cols+col];float left=float(code)*scale;left*=input;float right=bias*input;float term=left+right;acc+=term;}}y[ulong(job)*a.rows+row]=acc;}\n"
    "kernel void lfm_q6_warp(device const uchar *weights[[buffer(0)]],device const float *x[[buffer(1)]],device float *y[[buffer(2)]],constant Q &a[[buffer(3)]],uint lane[[thread_index_in_simdgroup]],uint warp[[simdgroup_index_in_threadgroup]],uint3 group[[threadgroup_position_in_grid]]){uint row=group.x*4u+warp,job=group.y;float acc=0.0f;for(uint base=0;base<a.cols;base+=32u){uint col=base+lane;float product=0.0f;if(row<a.rows&&job<a.jobs&&col<a.cols){ulong p=ulong(row)*a.cols+col,bit=p*6u,byte=bit>>3u,grp=p/64u;uint packed=uint(weights[a.value_offset+byte]);if((bit&7u)>2u)packed|=uint(weights[a.value_offset+byte+1u])<<8u;uint code=(packed>>uint(bit&7u))&63u;float input=x[ulong(job)*a.cols+col];float left=float(code)*bf16(weights+a.scale_offset+grp*2u);left*=input;float right=bf16(weights+a.bias_offset+grp*2u)*input;product=left+right;}uint count=a.cols-base;if(count>32u)count=32u;for(uint source=0;source<count;source++){float selected=simd_shuffle(product,source);if(lane==0u)acc+=selected;}}if(lane==0u&&row<a.rows&&job<a.jobs)y[ulong(job)*a.rows+row]=acc;}\n"
    "struct E{ulong value_offset,scale_offset,bias_offset;uint vocabulary,width,tokens;float scale;};\n"
    "kernel void lfm_q6_embedding(device const uchar *weights[[buffer(0)]],device const int *ids[[buffer(1)]],device float *out[[buffer(2)]],constant E &a[[buffer(3)]],uint index[[thread_position_in_grid]]){ulong elements=ulong(a.tokens)*a.width;if(index>=elements)return;uint row=uint(index/a.width),col=uint(index-ulong(row)*a.width);int token=ids[row];if(token<0||uint(token)>=a.vocabulary)return;ulong p=ulong(uint(token))*a.width+col,bit=p*6u,byte=bit>>3u,group=p/64u;uint packed=uint(weights[a.value_offset+byte]);if((bit&7u)>2u)packed|=uint(weights[a.value_offset+byte+1u])<<8u;uint code=(packed>>uint(bit&7u))&63u;float value=float(code)*bf16(weights+a.scale_offset+group*2u);value+=bf16(weights+a.bias_offset+group*2u);out[index]=value*a.scale;}\n"
    "kernel void lfm_silu(device const float *gate[[buffer(0)]],device const float *up[[buffer(1)]],device float *out[[buffer(2)]],constant uint &count[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<count){float g=gate[i];out[i]=(g/(1.0f+salt_exp(-g)))*up[i];}}\n"
    "struct R{uint rows,experts,topk,normalize,hidden;float scale;};\n"
    "kernel void lfm_router_scores(device float *scores[[buffer(0)]],device const float *bias[[buffer(1)]],device float *raw[[buffer(2)]],constant R &a[[buffer(3)]],uint index[[thread_position_in_grid]]){uint total=a.rows*a.experts;if(index>=total)return;uint expert=index%a.experts;float weight=1.0f/(1.0f+salt_exp(-scores[index]));raw[index]=weight;scores[index]=weight+bias[expert];}\n"
    "kernel void lfm_router_select(device const float *scores[[buffer(0)]],device const float *raw[[buffer(1)]],device int *selected[[buffer(2)]],device float *weights[[buffer(3)]],constant R &a[[buffer(4)]],uint row[[thread_position_in_grid]]){if(row>=a.rows)return;device const float *rs=scores+ulong(row)*a.experts;device const float *rr=raw+ulong(row)*a.experts;device int *sr=selected+ulong(row)*a.topk;device float *wr=weights+ulong(row)*a.topk;float chosen=0.0f;for(uint j=0;j<a.topk;j++){int best=-1;float best_score=0.0f,best_weight=0.0f;for(uint e=0;e<a.experts;e++){bool taken=false;for(uint q=0;q<j;q++)if(sr[q]==int(e)){taken=true;break;}if(taken)continue;float score=rs[e];if(best<0||score>best_score){best=int(e);best_score=score;best_weight=rr[e];}}sr[j]=best;wr[j]=best_weight;chosen+=best_weight;}if(a.normalize)for(uint j=0;j<a.topk;j++)wr[j]/=chosen;if(a.scale!=0.0f)for(uint j=0;j<a.topk;j++)wr[j]*=a.scale;for(uint left=0;left<a.topk;left++)for(uint right=left+1u;right<a.topk;right++)if(sr[right]<sr[left]){int id=sr[left];float w=wr[left];sr[left]=sr[right];wr[left]=wr[right];sr[right]=id;wr[right]=w;}}\n"
    "kernel void lfm_route_complete(device const int *selected[[buffer(0)]],device int *grouped[[buffer(1)]],device int *inverse[[buffer(2)]],device int *scratch[[buffer(3)]],constant R &a[[buffer(4)]],uint tid[[thread_position_in_grid]]){if(tid)return;uint jobs=a.rows*a.topk;for(uint e=0;e<a.experts;e++)scratch[e]=0;for(uint c=0;c<jobs;c++)scratch[uint(selected[c])]++;uint next=0;for(uint e=0;e<a.experts;e++){uint pop=uint(scratch[e]);scratch[e]=int(next);next+=pop;}for(uint c=0;c<jobs;c++){uint e=uint(selected[c]),g=uint(scratch[e]++);grouped[g]=int(c);}for(uint g=0;g<jobs;g++)inverse[uint(grouped[g])]=int(g);}\n"
    "kernel void lfm_routed_gather(device const int *grouped[[buffer(0)]],device const float *source[[buffer(1)]],device float *routed[[buffer(2)]],constant R &a[[buffer(3)]],uint index[[thread_position_in_grid]]){ulong jobs=ulong(a.rows)*a.topk,elements=jobs*a.hidden;if(index>=elements)return;uint g=uint(index/a.hidden),col=uint(index-ulong(g)*a.hidden),c=uint(grouped[g]),row=c/a.topk;routed[index]=source[ulong(row)*a.hidden+col];}\n"
    "struct T{uint maximum_context,query_heads,kv_heads,head_dim,heads_per_kv,rows,wave_first,wave_rows,query_width,kv_width,kv_ring,kv_capacity,score_workers;float score_scale;};\n"
    "inline device const float *lfm_committed(device const float *rows,uint position,uint source,uint width,uint ring,uint cap){if(!rows||!width||!cap||position>=source)return nullptr;uint index;if(ring){uint oldest=source>cap?source-cap:0u;if(position<oldest)return nullptr;index=position%cap;}else{if(position>=cap)return nullptr;index=position;}return rows+ulong(index)*width;}\n"
    "inline device const float *lfm_frontier(device const float *rows,device const uint *parents,device const uint *depths,uint n,uint current,uint target,uint width){if(!rows||!parents||!depths||current>=n)return nullptr;uint row=current,steps=0u;while(depths[row]>target){if(++steps>n)return nullptr;row=parents[row];if(row==0xffffffffu||row>=n)return nullptr;}return depths[row]==target?rows+ulong(row)*width:nullptr;}\n"
    "inline device const float *lfm_attn_row(device const float *committed,device const float *tentative,device const uint *parents,device const uint *depths,uint rows,uint current,uint position,uint source,uint width,uint ring,uint cap){return position<source?lfm_committed(committed,position,source,width,ring,cap):lfm_frontier(tentative,parents,depths,rows,current,position-source,width);}\n"
    "kernel void lfm_attention_scores(constant T &a[[buffer(0)]],device const uint *sourcep[[buffer(1)]],device const uint *parents[[buffer(2)]],device const uint *depths[[buffer(3)]],device const float *queries[[buffer(4)]],device const float *committed[[buffer(5)]],device const float *tentative[[buffer(6)]],device float *scores[[buffer(7)]],device atomic_int *status[[buffer(8)]],uint tid[[thread_index_in_threadgroup]],uint3 group[[threadgroup_position_in_grid]]){uint ledger=group.x,task=ledger/a.score_workers,worker=ledger-task*a.score_workers,local=task/a.query_heads,head=task-local*a.query_heads;if(local>=a.wave_rows||a.wave_first+local>=a.rows)return;if(atomic_load_explicit(status,memory_order_relaxed)!=0)return;uint source=*sourcep,row=a.wave_first+local,position=source+depths[row],kvhead=head/a.heads_per_kv;if(position>=a.maximum_context){atomic_store_explicit(status,301,memory_order_relaxed);return;}device const float *query=queries+ulong(row)*a.query_width+ulong(head)*a.head_dim;device float *out=scores+ulong(task)*a.maximum_context;for(uint kp=worker*256u+tid;kp<=position;kp+=a.score_workers*256u){device const float *key=lfm_attn_row(committed,tentative,parents,depths,a.rows,row,kp,source,a.kv_width,a.kv_ring,a.kv_capacity);if(!key){atomic_store_explicit(status,302,memory_order_relaxed);continue;}key+=ulong(kvhead)*a.head_dim;float acc=0.0f;for(uint col=0;col<a.head_dim;col++){float product=query[col]*key[col];acc+=product;}out[kp]=acc*a.score_scale;}}\n"
    "kernel void lfm_attention_reduce(constant T &a[[buffer(0)]],device const uint *sourcep[[buffer(1)]],device const uint *parents[[buffer(2)]],device const uint *depths[[buffer(3)]],device const float *committed[[buffer(4)]],device const float *tentative[[buffer(5)]],device float *scores[[buffer(6)]],device float *denominators[[buffer(7)]],device float *output[[buffer(8)]],device atomic_int *status[[buffer(9)]],uint tid[[thread_index_in_threadgroup]],uint3 group[[threadgroup_position_in_grid]]){uint task=group.x,local=task/a.query_heads,head=task-local*a.query_heads;if(local>=a.wave_rows||a.wave_first+local>=a.rows)return;if(atomic_load_explicit(status,memory_order_relaxed)!=0)return;uint row=a.wave_first+local,source=*sourcep,position=source+depths[row],kvhead=head/a.heads_per_kv;if(position>=a.maximum_context){atomic_store_explicit(status,311,memory_order_relaxed);return;}device float *rs=scores+ulong(task)*a.maximum_context;threadgroup float maximum;threadgroup float denominator;threadgroup float probabilities[64];threadgroup float products[4096];if(tid==0u){float peak=-1.0f/0.0f;for(uint kp=0;kp<=position;kp++)if(rs[kp]>peak)peak=rs[kp];maximum=peak;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint kp=tid;kp<=position;kp+=256u)rs[kp]=salt_exp(rs[kp]-maximum);threadgroup_barrier(mem_flags::mem_device);if(tid==0u){float sum=0.0f;for(uint kp=0;kp<=position;kp++)sum+=rs[kp];denominator=sum;denominators[ulong(row)*a.query_heads+head]=sum;if(!(sum>0.0f)||!isfinite(sum))atomic_store_explicit(status,312,memory_order_relaxed);}threadgroup_barrier(mem_flags::mem_threadgroup);if(atomic_load_explicit(status,memory_order_relaxed)!=0)return;float acc=0.0f;for(uint base=0;base<=position;base+=64u){uint count=position-base+1u;if(count>64u)count=64u;if(tid<count)probabilities[tid]=rs[base+tid]/denominator;threadgroup_barrier(mem_flags::mem_threadgroup);uint total=count*a.head_dim;for(uint index=tid;index<total;index+=256u){uint lp=index/a.head_dim,col=index-lp*a.head_dim,kp=base+lp;device const float *value=lfm_attn_row(committed,tentative,parents,depths,a.rows,row,kp,source,a.kv_width,a.kv_ring,a.kv_capacity);if(!value){products[index]=0.0f;atomic_store_explicit(status,313,memory_order_relaxed);}else{value+=ulong(kvhead)*a.head_dim;products[index]=probabilities[lp]*value[col];}}threadgroup_barrier(mem_flags::mem_threadgroup);if(tid<a.head_dim)for(uint lp=0;lp<count;lp++)acc+=products[ulong(lp)*a.head_dim+tid];threadgroup_barrier(mem_flags::mem_threadgroup);}if(tid<a.head_dim){output[ulong(row)*a.query_width+ulong(head)*a.head_dim+tid]=acc;if(!isfinite(acc))atomic_store_explicit(status,314,memory_order_relaxed);}}\n";

static void lfm_metal_disable_fast_math_legacy(MTLCompileOptions *options) {
    [options setValue:@NO forKey:@"fastMathEnabled"];
}

static void lfm_metal_exact_math(MTLCompileOptions *options) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *))
        options.mathMode = MTLMathModeSafe;
    else
        lfm_metal_disable_fast_math_legacy(options);
#else
    lfm_metal_disable_fast_math_legacy(options);
#endif
}

static int lfm_metal_prepare(void *context,
        const SaltTextModelGpuPrepareArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLFunction> function;
    MTLCompileOptions *options;
    NSError *error = nil;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL || !args->device ||
        !state || state_bytes != sizeof *state || args->maximum_rows == 0u)
        return -1;
    memset(state, 0, sizeof *state);
    device = (id<MTLDevice>)args->device;
    options = [MTLCompileOptions new];
    if (!options) return -1;
    lfm_metal_exact_math(options);
    library = [device newLibraryWithSource:
        [NSString stringWithUTF8String:LFM_METAL_SOURCE]
        options:options error:&error];
    [options release];
    if (!library) return -1;
    function = [library newFunctionWithName:@"lfm_short_conv"];
    if (!function) {
        [library release];
        return -1;
    }
    state->conv = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->conv) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_q6_project"];
    if (!function) {
        [library release];
        return -1;
    }
    state->q6 = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!state->q6) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_q6_warp"];
    if (!function) {
        [library release];
        return -1;
    }
    state->q6_warp = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->q6_warp) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_f32_project"];
    if (!function) {
        [library release];
        return -1;
    }
    state->f32 = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!state->f32) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_bf16_project"];
    if (!function) {
        [library release];
        return -1;
    }
    state->bf16 = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!state->bf16) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_silu"];
    if (!function) {
        [library release];
        return -1;
    }
    state->activation = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->activation) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_router_scores"];
    if (!function) {
        [library release];
        return -1;
    }
    state->router_scores = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->router_scores) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_router_select"];
    if (!function) {
        [library release];
        return -1;
    }
    state->router = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->router) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_route_complete"];
    if (!function) {
        [library release];
        return -1;
    }
    state->route_complete = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->route_complete) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_routed_gather"];
    if (!function) {
        [library release];
        return -1;
    }
    state->routed_gather = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->routed_gather) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_q6_embedding"];
    if (!function) {
        [library release];
        return -1;
    }
    state->embedding_q6 = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->embedding_q6) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_attention_scores"];
    if (!function) {
        [library release];
        return -1;
    }
    state->attention_scores = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    if (!state->attention_scores) {
        [library release];
        return -1;
    }
    function = [library newFunctionWithName:@"lfm_attention_reduce"];
    if (!function) {
        [library release];
        return -1;
    }
    state->attention_reduce = [device newComputePipelineStateWithFunction:function
        error:&error];
    [function release];
    [library release];
    if (!state->attention_reduce || args->maximum_context == 0u ||
        lfm_attention_io_ledger_prepare(&state->attention,
            args->maximum_context) != 0)
        return -1;
    return 0;
}

static int range_ok(uint64_t offset, uint64_t bytes, size_t capacity) {
    return offset <= (uint64_t)capacity && bytes <= (uint64_t)capacity - offset;
}

static int lfm_metal_view_ok(const SaltTextModelGpuBufferView *view,
        uint64_t need) {
    id<MTLBuffer> buffer;
    if (!view || !(buffer = (id<MTLBuffer>)view->resource) ||
        need > view->bytes || view->offset > (uint64_t)[buffer length])
        return 0;
    return need <= (uint64_t)[buffer length] - view->offset;
}

static int lfm_metal_project(void *context,
        const SaltTextModelGpuProjectArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> weights, canonical;
    LfmMetalQ6Args q6;
    uint64_t input_bytes, output_bytes, elements;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || args->rows == 0u ||
        (args->tensor.encoding != SALT_TENSOR_ENCODING_F32 &&
         args->tensor.encoding != SALT_TENSOR_ENCODING_BF16 &&
         args->tensor.encoding != SALT_TENSOR_ENCODING_AFFINE_Q6) ||
        args->tensor.rows == 0u || args->tensor.cols == 0u ||
        !(weights = (id<MTLBuffer>)args->tensor.resource))
        return -1;
    input_bytes = (uint64_t)args->rows * args->tensor.cols * sizeof(float);
    output_bytes = (uint64_t)args->rows * args->tensor.rows * sizeof(float);
    elements = (uint64_t)args->rows * args->tensor.rows;
    if (!range_ok(args->input_offset, input_bytes, args->canonical_bytes) ||
        !range_ok(args->output_offset, output_bytes, args->canonical_bytes) ||
        elements > NSUIntegerMax)
        return -1;
    command = (id<MTLCommandBuffer>)args->command;
    canonical = (id<MTLBuffer>)args->canonical;
    memset(&q6, 0, sizeof q6);
    q6.value_offset = args->tensor.value_offset;
    q6.scale_offset = args->tensor.scale_offset;
    q6.bias_offset = args->tensor.bias_offset;
    q6.rows = args->tensor.rows;
    q6.cols = args->tensor.cols;
    q6.jobs = args->rows;
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:
        args->tensor.encoding == SALT_TENSOR_ENCODING_F32 ? state->f32 :
        args->tensor.encoding == SALT_TENSOR_ENCODING_BF16 ? state->bf16 :
        (LFM_METAL_Q6_WARP ? state->q6_warp : state->q6)];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->input_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->output_offset atIndex:2];
    [encoder setBytes:&q6 length:sizeof q6 atIndex:3];
    if (args->tensor.encoding == SALT_TENSOR_ENCODING_AFFINE_Q6 &&
        LFM_METAL_Q6_WARP)
        [encoder dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)args->tensor.rows + 3u) / 4u,
                        (NSUInteger)args->rows, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    else
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return 0;
}

static int lfm_metal_embedding(void *context,
        const SaltTextModelGpuEmbeddingArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> weights, canonical;
    LfmMetalEmbeddingArgs embedding;
    uint64_t token_bytes, output_bytes, elements;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || !state->embedding_q6 ||
        args->token_count == 0u ||
        args->tensor.encoding != SALT_TENSOR_ENCODING_AFFINE_Q6 ||
        args->tensor.rows != LFM_VOCAB || args->tensor.cols != LFM_HIDDEN ||
        !(weights = (id<MTLBuffer>)args->tensor.resource))
        return -1;
    token_bytes = (uint64_t)args->token_count * sizeof(int32_t);
    output_bytes = (uint64_t)args->token_count * LFM_HIDDEN * sizeof(float);
    elements = (uint64_t)args->token_count * LFM_HIDDEN;
    if (!range_ok(args->token_ids_offset, token_bytes, args->canonical_bytes) ||
        !range_ok(args->output_offset, output_bytes, args->canonical_bytes) ||
        elements > NSUIntegerMax)
        return -1;
    command = (id<MTLCommandBuffer>)args->command;
    canonical = (id<MTLBuffer>)args->canonical;
    memset(&embedding, 0, sizeof embedding);
    embedding.value_offset = args->tensor.value_offset;
    embedding.scale_offset = args->tensor.scale_offset;
    embedding.bias_offset = args->tensor.bias_offset;
    embedding.vocabulary = args->tensor.rows;
    embedding.width = args->tensor.cols;
    embedding.tokens = args->token_count;
    embedding.scale = args->scale;
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->embedding_q6];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->token_ids_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->output_offset atIndex:2];
    [encoder setBytes:&embedding length:sizeof embedding atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return 0;
}

static int lfm_metal_activation(void *context,
        const SaltTextModelGpuActivationArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> canonical;
    uint32_t count;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || !state->activation ||
        args->activation != SALT_ACT_SILU || args->elements == 0u ||
        args->elements > UINT32_MAX)
        return -1;
    count = (uint32_t)args->elements;
    if (!range_ok(args->gate_offset, (uint64_t)count * sizeof(float),
                  args->canonical_bytes) ||
        !range_ok(args->up_offset, (uint64_t)count * sizeof(float),
                  args->canonical_bytes) ||
        !range_ok(args->output_offset, (uint64_t)count * sizeof(float),
                  args->canonical_bytes))
        return -1;
    command = (id<MTLCommandBuffer>)args->command;
    canonical = (id<MTLBuffer>)args->canonical;
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->activation];
    [encoder setBuffer:canonical offset:(NSUInteger)args->gate_offset atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->up_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->output_offset atIndex:2];
    [encoder setBytes:&count length:sizeof count atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return 0;
}

static int lfm_metal_router(void *context,
        const SaltTextModelGpuRouterArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> canonical, bias;
    struct { uint32_t rows, experts, topk, normalize, hidden; float scale; } routing;
    uint64_t logits_bytes, chosen;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || !state->router_scores ||
        !state->router ||
        !state->route_complete || !state->routed_gather ||
        args->rows == 0u || args->experts != LFM_EXPERTS ||
        args->topk != LFM_TOP_K ||
        args->expert_bias.encoding != SALT_TENSOR_ENCODING_F32 ||
        args->expert_bias.rows != 1u ||
        args->expert_bias.cols != LFM_EXPERTS ||
        !(bias = (id<MTLBuffer>)args->expert_bias.resource))
        return -1;
    logits_bytes = (uint64_t)args->rows * args->experts * sizeof(float);
    chosen = (uint64_t)args->rows * args->topk;
    if (!range_ok(args->logits_offset, logits_bytes, args->canonical_bytes) ||
        !range_ok(args->selected_offset, chosen * sizeof(int32_t),
                  args->canonical_bytes) ||
        !range_ok(args->weights_offset, chosen * sizeof(float),
                  args->canonical_bytes) ||
        !range_ok(args->grouped_offset, chosen * sizeof(int32_t),
                  args->canonical_bytes) ||
        !range_ok(args->inverse_offset, chosen * sizeof(int32_t),
                  args->canonical_bytes) ||
        !range_ok(args->scratch_offset,
                  (uint64_t)args->rows * args->experts * sizeof(float),
                  args->canonical_bytes) ||
        !range_ok(args->source_offset,
                  (uint64_t)args->rows * args->hidden * sizeof(float),
                  args->canonical_bytes) ||
        !range_ok(args->routed_offset,
                  chosen * args->hidden * sizeof(float),
                  args->canonical_bytes))
        return -1;
    routing.rows = args->rows;
    routing.experts = args->experts;
    routing.topk = args->topk;
    routing.normalize = args->normalize_topk ? 1u : 0u;
    routing.hidden = args->hidden;
    routing.scale = args->router_scale;
    command = (id<MTLCommandBuffer>)args->command;
    canonical = (id<MTLBuffer>)args->canonical;
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->router_scores];
    [encoder setBuffer:canonical offset:(NSUInteger)args->logits_offset atIndex:0];
    [encoder setBuffer:bias offset:(NSUInteger)args->expert_bias.value_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->scratch_offset atIndex:2];
    [encoder setBytes:&routing length:sizeof routing atIndex:3];
    [encoder dispatchThreads:
         MTLSizeMake((NSUInteger)(args->rows * args->experts), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->router];
    [encoder setBuffer:canonical offset:(NSUInteger)args->logits_offset atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->scratch_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->selected_offset atIndex:2];
    [encoder setBuffer:canonical offset:(NSUInteger)args->weights_offset atIndex:3];
    [encoder setBytes:&routing length:sizeof routing atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(args->rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->route_complete];
    [encoder setBuffer:canonical offset:(NSUInteger)args->selected_offset atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->grouped_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->inverse_offset atIndex:2];
    [encoder setBuffer:canonical offset:(NSUInteger)args->scratch_offset atIndex:3];
    [encoder setBytes:&routing length:sizeof routing atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->routed_gather];
    [encoder setBuffer:canonical offset:(NSUInteger)args->grouped_offset atIndex:0];
    [encoder setBuffer:canonical offset:(NSUInteger)args->source_offset atIndex:1];
    [encoder setBuffer:canonical offset:(NSUInteger)args->routed_offset atIndex:2];
    [encoder setBytes:&routing length:sizeof routing atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)(chosen * args->hidden), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return 0;
}

static int lfm_metal_attention(void *context,
        const SaltTextModelGpuAttentionArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    uint64_t score_bytes, query_bytes, kv_bytes;
    uint32_t wave_first;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->physical_launches || !args->full_attention ||
        !state || state_bytes != sizeof *state ||
        !state->attention_scores || !state->attention_reduce ||
        state->attention.ready != 1u || args->rows == 0u ||
        args->maximum_context != state->attention.maximum_context ||
        args->query_width != LFM_HIDDEN ||
        args->kv_width != LFM_KV_HEADS * LFM_HEAD_DIM ||
        args->n_heads != state->attention.query_heads ||
        args->n_kv_heads != state->attention.kv_heads ||
        args->head_dim != state->attention.head_dim ||
        args->kv_capacity == 0u)
        return -1;
    if (args->kv_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
        args->kv_mode != SALT_TEXT_KV_PRIVATE_RING)
        return -1;
    score_bytes = (uint64_t)state->attention.score_task_capacity *
        state->attention.maximum_context * sizeof(float);
    query_bytes = (uint64_t)args->rows * args->query_width * sizeof(float);
    kv_bytes = (uint64_t)args->kv_capacity * args->kv_width * sizeof(float);
    if (!lfm_metal_view_ok(&args->source_position, sizeof(uint32_t)) ||
        !lfm_metal_view_ok(&args->parent_rows,
            (uint64_t)args->rows * sizeof(uint32_t)) ||
        !lfm_metal_view_ok(&args->depths,
            (uint64_t)args->rows * sizeof(uint32_t)) ||
        !lfm_metal_view_ok(&args->queries, query_bytes) ||
        !lfm_metal_view_ok(&args->committed_keys, kv_bytes) ||
        !lfm_metal_view_ok(&args->committed_values, kv_bytes) ||
        !lfm_metal_view_ok(&args->tentative_keys,
            (uint64_t)args->rows * args->kv_width * sizeof(float)) ||
        !lfm_metal_view_ok(&args->tentative_values,
            (uint64_t)args->rows * args->kv_width * sizeof(float)) ||
        !lfm_metal_view_ok(&args->score_workspace, score_bytes) ||
        !lfm_metal_view_ok(&args->output, query_bytes) ||
        !lfm_metal_view_ok(&args->status, sizeof(int)))
        return -1;
    command = (id<MTLCommandBuffer>)args->command;
    *args->physical_launches = 0u;
    for (wave_first = 0u; wave_first < args->rows;
         wave_first += state->attention.rows_per_wave) {
        LfmMetalAttentionArgs call;
        id<MTLComputeCommandEncoder> encoder;
        uint32_t wave_rows = args->rows - wave_first;
        uint32_t score_groups, reduce_groups;
        if (wave_rows > state->attention.rows_per_wave)
            wave_rows = state->attention.rows_per_wave;
        memset(&call, 0, sizeof call);
        call.maximum_context = state->attention.maximum_context;
        call.query_heads = state->attention.query_heads;
        call.kv_heads = state->attention.kv_heads;
        call.head_dim = state->attention.head_dim;
        call.heads_per_kv = state->attention.heads_per_kv;
        call.rows = args->rows;
        call.wave_first = wave_first;
        call.wave_rows = wave_rows;
        call.query_width = args->query_width;
        call.kv_width = args->kv_width;
        call.kv_ring = args->kv_mode == SALT_TEXT_KV_PRIVATE_RING ? 1u : 0u;
        call.kv_capacity = args->kv_capacity;
        call.score_workers = wave_rows == 1u
            ? state->attention.decode_score_workers : 1u;
        call.score_scale = args->score_scale;
        score_groups = wave_rows * call.query_heads * call.score_workers;
        reduce_groups = wave_rows * call.query_heads;

        encoder = [command computeCommandEncoder];
        if (!encoder) return -1;
        [encoder setComputePipelineState:state->attention_scores];
        [encoder setBytes:&call length:sizeof call atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)args->source_position.resource
                    offset:(NSUInteger)args->source_position.offset atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)args->parent_rows.resource
                    offset:(NSUInteger)args->parent_rows.offset atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)args->depths.resource
                    offset:(NSUInteger)args->depths.offset atIndex:3];
        [encoder setBuffer:(id<MTLBuffer>)args->queries.resource
                    offset:(NSUInteger)args->queries.offset atIndex:4];
        [encoder setBuffer:(id<MTLBuffer>)args->committed_keys.resource
                    offset:(NSUInteger)args->committed_keys.offset atIndex:5];
        [encoder setBuffer:(id<MTLBuffer>)args->tentative_keys.resource
                    offset:(NSUInteger)args->tentative_keys.offset atIndex:6];
        [encoder setBuffer:(id<MTLBuffer>)args->score_workspace.resource
                    offset:(NSUInteger)args->score_workspace.offset atIndex:7];
        [encoder setBuffer:(id<MTLBuffer>)args->status.resource
                    offset:(NSUInteger)args->status.offset atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(score_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        if (!encoder) return -1;
        [encoder setComputePipelineState:state->attention_reduce];
        [encoder setBytes:&call length:sizeof call atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)args->source_position.resource
                    offset:(NSUInteger)args->source_position.offset atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)args->parent_rows.resource
                    offset:(NSUInteger)args->parent_rows.offset atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)args->depths.resource
                    offset:(NSUInteger)args->depths.offset atIndex:3];
        [encoder setBuffer:(id<MTLBuffer>)args->committed_values.resource
                    offset:(NSUInteger)args->committed_values.offset atIndex:4];
        [encoder setBuffer:(id<MTLBuffer>)args->tentative_values.resource
                    offset:(NSUInteger)args->tentative_values.offset atIndex:5];
        [encoder setBuffer:(id<MTLBuffer>)args->score_workspace.resource
                    offset:(NSUInteger)args->score_workspace.offset atIndex:6];
        [encoder setBuffer:(id<MTLBuffer>)args->queries.resource
                    offset:(NSUInteger)args->queries.offset atIndex:7];
        [encoder setBuffer:(id<MTLBuffer>)args->output.resource
                    offset:(NSUInteger)args->output.offset atIndex:8];
        [encoder setBuffer:(id<MTLBuffer>)args->status.resource
                    offset:(NSUInteger)args->status.offset atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(reduce_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (*args->physical_launches > UINT32_MAX - 2u) return -1;
        *args->physical_launches += 2u;
    }
    return 0;
}

static int lfm_metal_encode(void *context,
        const SaltTextModelGpuEncodeArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> canonical, kernel;
    LfmMetalConvArgs conv;
    uint64_t scratch_bytes, recurrent_bytes, gated_offset;
    (void)context;
    if (!args || args->backend != SALT_TEXT_MODEL_GPU_METAL ||
        !args->command || !args->canonical || !args->project ||
        !args->backend_context || !state || state_bytes != sizeof *state ||
        !state->conv || args->hidden != LFM_HIDDEN || args->rows == 0u ||
        args->kernel.encoding != SALT_TENSOR_ENCODING_BF16 ||
        args->kernel.rows != LFM_HIDDEN || args->kernel.cols != LFM_CONV_L ||
        args->input_projection.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        args->input_projection.rows != 3u * LFM_HIDDEN ||
        args->input_projection.cols != LFM_HIDDEN ||
        args->output_projection.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        args->output_projection.rows != LFM_HIDDEN ||
        args->output_projection.cols != LFM_HIDDEN ||
        args->scratch_float_capacity < (size_t)args->rows * 4u * LFM_HIDDEN ||
        args->state_bytes != (size_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float))
        return -1;
    scratch_bytes = (uint64_t)args->rows * 4u * LFM_HIDDEN * sizeof(float);
    recurrent_bytes = (uint64_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float);
    if (!range_ok(args->normalized_offset,
            (uint64_t)args->rows * LFM_HIDDEN * sizeof(float),
            args->canonical_bytes) ||
        !range_ok(args->scratch_offset, scratch_bytes, args->canonical_bytes) ||
        !range_ok(args->state_offset, recurrent_bytes, args->canonical_bytes) ||
        !range_ok(args->output_offset,
            (uint64_t)args->rows * LFM_HIDDEN * sizeof(float),
            args->canonical_bytes))
        return -1;
    if (args->project(args->backend_context, &args->input_projection,
            args->normalized_offset, args->scratch_offset, args->rows) != 0)
        return -1;
    command = (id<MTLCommandBuffer>)args->command;
    canonical = (id<MTLBuffer>)args->canonical;
    kernel = (id<MTLBuffer>)args->kernel.resource;
    if (!kernel || args->scratch_offset % sizeof(float) != 0u ||
        args->state_offset % sizeof(float) != 0u)
        return -1;
    memset(&conv, 0, sizeof conv);
    conv.scratch_float = args->scratch_offset / sizeof(float);
    conv.state_float = args->state_offset / sizeof(float);
    conv.kernel_byte = args->kernel.value_offset;
    conv.rows = args->rows;
    conv.hidden = LFM_HIDDEN;
    conv.kernel_size = LFM_CONV_L;
    encoder = [command computeCommandEncoder];
    if (!encoder) return -1;
    [encoder setComputePipelineState:state->conv];
    [encoder setBuffer:canonical offset:0 atIndex:0];
    [encoder setBuffer:kernel offset:0 atIndex:1];
    [encoder setBytes:&conv length:sizeof conv atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(LFM_HIDDEN, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    gated_offset = args->scratch_offset +
        (uint64_t)args->rows * 3u * LFM_HIDDEN * sizeof(float);
    return args->project(args->backend_context, &args->output_projection,
        gated_offset, args->output_offset, args->rows);
}

static void lfm_metal_destroy(void *context, SaltTextModelGpuBackend backend,
        void *state_memory, size_t state_bytes) {
    LfmMetalState *state = (LfmMetalState *)state_memory;
    (void)context;
    if (backend != SALT_TEXT_MODEL_GPU_METAL || !state ||
        state_bytes != sizeof *state)
        return;
    if (state->conv) [state->conv release];
    if (state->q6) [state->q6 release];
    if (state->q6_warp) [state->q6_warp release];
    if (state->f32) [state->f32 release];
    if (state->bf16) [state->bf16 release];
    if (state->activation) [state->activation release];
    if (state->router_scores) [state->router_scores release];
    if (state->router) [state->router release];
    if (state->route_complete) [state->route_complete release];
    if (state->routed_gather) [state->routed_gather release];
    if (state->embedding_q6) [state->embedding_q6 release];
    if (state->attention_scores) [state->attention_scores release];
    if (state->attention_reduce) [state->attention_reduce release];
    memset(state, 0, sizeof *state);
}

static const SaltTextModelGpuOps LFM_METAL_OPS = {
    .state_bytes = sizeof(LfmMetalState),
    .prepare = lfm_metal_prepare,
    .project = lfm_metal_project,
    .embedding = lfm_metal_embedding,
    .activation = lfm_metal_activation,
    .router_topk = lfm_metal_router,
    .attention = LFM_METAL_TILED_ATTENTION ? lfm_metal_attention : NULL,
    .encode = lfm_metal_encode,
    .destroy = lfm_metal_destroy,
};

const SaltTextModelGpuOps *lfm_gpu_ops(void) {
    return &LFM_METAL_OPS;
}

int lfm_gpu_canonical_readback(void *host, const void *backend,
                               size_t offset, size_t bytes) {
    (void)host;
    (void)backend;
    (void)offset;
    (void)bytes;
    return 0;
}
