#ifndef SALT_ROTATION_ATTENTION_PRODUCT_TILE_H
#define SALT_ROTATION_ATTENTION_PRODUCT_TILE_H
/* Initialization-only source selection. The original attention kernel and every
 * math helper remain byte-for-byte unchanged in gpu_metal_ops_source.h. Select
 * its complete score/softmax prefix and its scalar product/add statements; only
 * the ownership of independent AV products changes. No alternate math helper. */
static NSString *rotation_attention_product_source(void) {
    NSString *source = [NSString stringWithUTF8String:salt_gpu_metal_ops_source];
    NSRange start = [source rangeOfString:@"kernel void attention_exact_view("];
    if (start.location == NSNotFound) return nil;
    NSRange end = [source rangeOfString:@"\n" options:0
        range:NSMakeRange(start.location, [source length] - start.location)];
    if (end.location == NSNotFound) return nil;
    NSString *original = [source substringWithRange:
        NSMakeRange(start.location, end.location - start.location)];
    NSRange av = [original rangeOfString:@"for(uint d=tid;d<a.head_dim;d+=nt){float acc=0.0f;"];
    NSRange multiply = [original rangeOfString:@"float w=scores[i]/shared[0];float product=w*vr[d];"];
    NSRange add = [original rangeOfString:@"acc+=product;"];
    if (av.location == NSNotFound || multiply.location == NSNotFound ||
        add.location == NSNotFound || multiply.location < av.location ||
        add.location < multiply.location) return nil;
    NSString *prefix = [[original substringToIndex:av.location]
        stringByReplacingOccurrencesOfString:@"attention_exact_view("
                                 withString:@"attention_exact_product_view("];
    NSString *argument = @"threadgroup float *shared[[threadgroup(1)]]";
    if ([prefix rangeOfString:argument].location == NSNotFound) return nil;
    prefix = [prefix stringByReplacingOccurrencesOfString:argument withString:
        @"threadgroup float *shared[[threadgroup(1)]],threadgroup float *products[[threadgroup(2)]]"];
    NSString *multiply_source = [original substringWithRange:multiply];
    NSString *add_source = [original substringWithRange:add];
    NSString *body = [NSString stringWithFormat:
        @"for(uint column_first=0u;column_first<a.head_dim;column_first+=64u){"
         "uint columns=min(64u,a.head_dim-column_first);float acc=0.0f;"
         "for(uint key_first=0u;key_first<count;key_first+=64u){"
         "uint positions=min(64u,count-key_first);"
         "for(uint job=tid;job<positions*64u;job+=nt){"
         "uint row=job/64u,col=job%%64u;"
         "if(col>=columns){products[job]=0.0f;continue;}"
         "uint i=key_first+row,p=uint(first)+i,d=column_first+col;"
         "device const float *vr=text_kv_row(pv,sv,tv,parents,depths,p,a.source,token,a.kvstride,a);"
         "if(!vr){products[job]=NAN;atomic_store_explicit(status,1u,memory_order_relaxed);continue;}"
         "vr+=ulong(kh)*a.head_dim;%@products[job]=product;}"
         "threadgroup_barrier(mem_flags::mem_threadgroup);"
         "if(tid<columns){for(uint row=0u;row<positions;row++){"
         "float product=products[row*64u+tid];%@}}"
         "threadgroup_barrier(mem_flags::mem_threadgroup);}"
         "if(tid<columns){dst[column_first+tid]=acc;"
         "if(!isfinite(acc))atomic_store_explicit(status,1u,memory_order_relaxed);}}}\n",
        multiply_source, add_source];
    return [source stringByAppendingString:[prefix stringByAppendingString:body]];
}
#endif
