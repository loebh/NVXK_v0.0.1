#include <nvxk.h>
#include <immintrin.h>

bool XK_has_avx2_fma() {
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#define XK_TARGET_AVX2 __attribute__((target("avx2")))
#define XK_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define XK_TARGET_AVX2
#define XK_TARGET_AVX2_FMA
#endif

static XK_TARGET_AVX2 void vector_add_avx2_impl(const float* a, const float* b, float* c, size_t n) {
    size_t i=0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb=_mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va,vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    for(; i<n; ++i) c[i] = a[i]+b[i];
}

void vector_add_avx2(const float* a, const float* b, float* c, size_t n) {
    if (XK_has_avx2_fma()) {
        vector_add_avx2_impl(a, b, c, n);
    } else {
        for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
    }
}

void XK_naive(const float* A,const float* B, float* C,size_t N){
    for (size_t i = 0; i < N; ++i) {
        for(size_t j=0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k<N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i*N + j]=sum;
        }
    }
}

void XK_ikj(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N * N; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const float a_ik = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

static XK_TARGET_AVX2_FMA void XK_avx2_impl(const float* A, const float* B, float* C, size_t N) {
    for(size_t i = 0; i < N*N; ++i) C[i]=0.0f;

    for (size_t i = 0; i < N; ++i) {
        for(size_t k = 0; k < N; ++k) {
            __m256 a_ik = _mm256_set1_ps(A[i * N + k]);
            size_t j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 b_row = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_row = _mm256_loadu_ps(&C[i * N + j]);
                
                c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
                _mm256_storeu_ps(&C[i * N + j], c_row);
            }
            for(; j<N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void XK_avx2(const float* A, const float* B, float* C, size_t N) {
    if (XK_has_avx2_fma()) {
        XK_avx2_impl(A, B, C, N);
    } else {
        XK_ikj(A, B, C, N);
    }
}

static XK_TARGET_AVX2_FMA void XK_avx2_omp_impl(
    const float* A, const float* B, float* C, size_t N, int thread_count) {
    if (thread_count < 1) thread_count = 1;

    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0f;

        for (size_t k = 0; k < N; ++k) {
            __m256 a_ik = _mm256_set1_ps(A[i * N + k]);
            size_t j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 b_row = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_row = _mm256_loadu_ps(&C[i * N + j]);

                c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
                _mm256_storeu_ps(&C[i * N + j], c_row);
            }
            for (; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

static void XK_ikj_omp_impl(const float* A, const float* B, float* C, size_t N, int thread_count) {
    if (thread_count < 1) thread_count = 1;

    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0f;
        for (size_t k = 0; k < N; ++k) {
            const float a_ik = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

void XK_avx2_omp(const float* A, const float* B, float* C, size_t N, int thread_count) {
    if (XK_has_avx2_fma()) {
        XK_avx2_omp_impl(A, B, C, N, thread_count);
    } else {
        XK_ikj_omp_impl(A, B, C, N, thread_count);
    }
}

void XK_auto(const float* A, const float* B, float* C, size_t N) {
    XK_avx2(A, B, C, N);
}

void XK_omp(const float* A, const float* B, float* C, size_t N, int thread_count) {
    XK_avx2_omp(A, B, C, N, thread_count);
}

static XK_TARGET_AVX2_FMA void XK_blocked_impl(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N * N; ++i) C[i] = 0.0f;

    const size_t BLOCK=64;

    for(size_t i=0; i < N; i += BLOCK) {
        for (size_t k = 0; k < N; k += BLOCK) {
            for(size_t j=0; j < N; j += BLOCK) {
                
                for (size_t ii = i; ii < i + BLOCK && ii < N; ++ii) {
                    for (size_t kk = k; kk < k + BLOCK && kk < N; ++kk) {
                        
                        __m256 a_ik = _mm256_set1_ps(A[ii * N + kk]);
                        size_t jj=j;
                        
                        for (; jj + 7 < j + BLOCK && jj + 7 < N; jj += 8) {
                            __m256 b_row = _mm256_loadu_ps(&B[kk * N + jj]);
                            __m256 c_row = _mm256_loadu_ps(&C[ii * N + jj]);
                            
                            c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
                            _mm256_storeu_ps(&C[ii * N + jj], c_row);
                        }
                        
                    
                        for (; jj < j + BLOCK && jj < N; ++jj) {
                            C[ii * N + jj] += A[ii * N + kk] * B[kk * N + jj];
                        }
                    }
                }
                
            }
        }
    }
}

void XK_blocked(const float* A, const float* B, float* C, size_t N) {
    if (XK_has_avx2_fma()) {
        XK_blocked_impl(A, B, C, N);
    } else {
        XK_ikj(A, B, C, N);
    }
}