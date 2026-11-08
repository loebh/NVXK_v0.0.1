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

static XK_TARGET_AVX2 void vector_add_avx2_double_impl(
    const double* a, const double* b, double* c, size_t n) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        const __m256d va = _mm256_loadu_pd(&a[i]);
        const __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_add_pd(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

void vector_add_avx2(const double* a, const double* b, double* c, size_t n) {
    if (XK_has_avx2_fma()) {
        vector_add_avx2_double_impl(a, b, c, n);
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

static XK_TARGET_AVX2_FMA void XK_avx2_row(
    const float* A, const float* B, float* C, size_t N, size_t i) {
    for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0f;
    for (size_t k = 0; k < N; ++k) {
        const __m256 a_ik = _mm256_set1_ps(A[i * N + k]);
        size_t j = 0;
        for (; j + 7 < N; j += 8) {
            const __m256 b_row = _mm256_loadu_ps(&B[k * N + j]);
            __m256 c_row = _mm256_loadu_ps(&C[i * N + j]);
            c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
            _mm256_storeu_ps(&C[i * N + j], c_row);
        }
        for (; j < N; ++j) C[i * N + j] += A[i * N + k] * B[k * N + j];
    }
}

static void XK_avx2_omp_impl(
    const float* A, const float* B, float* C, size_t N, int thread_count) {
    if (thread_count < 1) thread_count = 1;
    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (size_t i = 0; i < N; ++i) XK_avx2_row(A, B, C, N, i);
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
    if (thread_count < 1) thread_count = 1;
    if (N < 256 || thread_count == 1) {
        XK_avx2(A, B, C, N);
        return;
    }
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

static XK_TARGET_AVX2_FMA void XK_gemm_blocked_impl(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N, size_t block_size) {
    const size_t output_count = xk_detail::checked_element_count(M, N);
    for (size_t i = 0; i < output_count; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; i += block_size) {
        const size_t i_end = i + (M - i < block_size ? M - i : block_size);
        for (size_t k = 0; k < K; k += block_size) {
            const size_t k_end = k + (K - k < block_size ? K - k : block_size);
            for (size_t j = 0; j < N; j += block_size) {
                const size_t j_end = j + (N - j < block_size ? N - j : block_size);
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        const __m256 a_ik = _mm256_set1_ps(A[ii * K + kk]);
                        size_t jj = j;
                        for (; jj + 7 < j_end; jj += 8) {
                            __m256 b_row = _mm256_loadu_ps(&B[kk * N + jj]);
                            __m256 c_row = _mm256_loadu_ps(&C[ii * N + jj]);
                            c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
                            _mm256_storeu_ps(&C[ii * N + jj], c_row);
                        }
                        for (; jj < j_end; ++jj) {
                            C[ii * N + jj] += A[ii * K + kk] * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

void XK_blocked(const float* A, const float* B, float* C, size_t N) {
    XK_blocked(A, B, C, N, 64);
}

void XK_blocked(const float* A, const float* B, float* C, size_t N, size_t block_size) {
    XK_gemm_blocked(A, B, C, N, N, N, block_size);
}

static XK_TARGET_AVX2_FMA void XK_gemm_avx2_row(
    const float* A, const float* B, float* C, size_t K, size_t N, size_t i) {
    for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        const __m256 a_ik = _mm256_set1_ps(A[i * K + k]);
        size_t j = 0;
        for (; j + 7 < N; j += 8) {
            const __m256 b_row = _mm256_loadu_ps(&B[k * N + j]);
            __m256 c_row = _mm256_loadu_ps(&C[i * N + j]);
            c_row = _mm256_fmadd_ps(a_ik, b_row, c_row);
            _mm256_storeu_ps(&C[i * N + j], c_row);
        }
        for (; j < N; ++j) C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
}

static XK_TARGET_AVX2_FMA void XK_gemm_avx2_impl(
    const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) XK_gemm_avx2_row(A, B, C, K, N, i);
}

static XK_TARGET_AVX2_FMA void XK_gemm_avx2_double_row(
    const double* A, const double* B, double* C, size_t K, size_t N, size_t i) {
    for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0;
    for (size_t k = 0; k < K; ++k) {
        const __m256d a_ik = _mm256_set1_pd(A[i * K + k]);
        size_t j = 0;
        for (; j + 3 < N; j += 4) {
            const __m256d b_row = _mm256_loadu_pd(&B[k * N + j]);
            __m256d c_row = _mm256_loadu_pd(&C[i * N + j]);
            c_row = _mm256_fmadd_pd(a_ik, b_row, c_row);
            _mm256_storeu_pd(&C[i * N + j], c_row);
        }
        for (; j < N; ++j) C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
}

static XK_TARGET_AVX2_FMA void XK_gemm_avx2_double_impl(
    const double* A, const double* B, double* C,
    size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) XK_gemm_avx2_double_row(A, B, C, K, N, i);
}

static void XK_gemm_omp_avx2_impl(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N, int thread_count) {
    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (size_t i = 0; i < M; ++i) XK_gemm_avx2_row(A, B, C, K, N, i);
}

void XK_gemm_avx2(const float* A, const float* B, float* C,
                  size_t M, size_t K, size_t N) {
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    if (!XK_has_avx2_fma()) {
        XK_gemm_ikj(A, B, C, M, K, N);
        return;
    }
    XK_gemm_avx2_impl(A, B, C, M, K, N);
}

void XK_gemm_avx2(const double* A, const double* B, double* C,
                  size_t M, size_t K, size_t N) {
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    if (!XK_has_avx2_fma()) {
        XK_gemm_ikj(A, B, C, M, K, N);
        return;
    }
    XK_gemm_avx2_double_impl(A, B, C, M, K, N);
}

void XK_gemm(const float* A, const float* B, float* C,
             size_t M, size_t K, size_t N) {
    XK_gemm_avx2(A, B, C, M, K, N);
}

void XK_gemm(const double* A, const double* B, double* C,
             size_t M, size_t K, size_t N) {
    XK_gemm_avx2(A, B, C, M, K, N);
}

void XK_gemm_i32_i64(const int32_t* A, const int32_t* B, int64_t* C,
                     size_t M, size_t K, size_t N) {
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<int64_t>(A[i * K + k]) *
                       static_cast<int64_t>(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

void XK_gemm_blocked(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N, size_t block_size) {
    if (block_size == 0) throw std::invalid_argument("block size must be positive");
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    if (XK_has_avx2_fma()) {
        XK_gemm_blocked_impl(A, B, C, M, K, N, block_size);
    } else {
        XK_gemm_ikj(A, B, C, M, K, N);
    }
}

void XK_gemm_omp(const float* A, const float* B, float* C,
                 size_t M, size_t K, size_t N, int thread_count) {
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    if (thread_count < 1) thread_count = 1;
    const size_t work_cutoff = static_cast<size_t>(256) * 256 * 256;
    if (M < 2 || K == 0 || N == 0 || thread_count == 1 ||
        M <= work_cutoff / K / N) {
        XK_gemm_avx2(A, B, C, M, K, N);
        return;
    }
    if (M < static_cast<size_t>(thread_count)) thread_count = static_cast<int>(M);
    if (XK_has_avx2_fma()) {
        XK_gemm_omp_avx2_impl(A, B, C, M, K, N, thread_count);
    } else {
        #pragma omp parallel for num_threads(thread_count) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) C[i * N + j] = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                const float a_ik = A[i * K + k];
                for (size_t j = 0; j < N; ++j) C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}