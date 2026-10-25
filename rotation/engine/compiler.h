#ifndef SALT_COMPILER_H
#define SALT_COMPILER_H

/* Salt's ordinary sources remain C99. Thread-local storage and x86 feature
 * discovery are explicit compiler adapters rather than unmarked language
 * extensions. Clang and GCC are the qualified compiler families. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SALT_THREAD_LOCAL static _Thread_local
#elif defined(__clang__) || defined(__GNUC__)
#define SALT_THREAD_LOCAL __extension__ static __thread
#else
#error "Salt requires a C11 or Clang/GCC thread-local storage adapter"
#endif

#if (defined(__x86_64__) || defined(__i386__)) && \
        (defined(__clang__) || defined(__GNUC__))
#define SALT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define SALT_TARGET_AVX2
#endif

static inline int salt_compiler_avx2_available(void) {
#if (defined(__x86_64__) || defined(__i386__)) && \
        (defined(__clang__) || defined(__GNUC__))
    return (__builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("fma")) ? 1 : 0;
#else
    return 0;
#endif
}

#endif /* SALT_COMPILER_H */
