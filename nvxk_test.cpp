#include <nvxk.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <vector>

template<typename T>
void test_vector_add(const size_t* sizes, size_t size_count) {
    for (size_t s = 0; s < size_count; ++s) {
        const size_t count = sizes[s];
        std::vector<T> a(count), b(count), result(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = static_cast<T>(static_cast<int>(i % 23) - 11) / static_cast<T>(3);
            b[i] = static_cast<T>(static_cast<int>(i % 19) - 9) / static_cast<T>(5);
        }
        vector_add_avx2(a.data(), b.data(), result.data(), count);
        for (size_t i = 0; i < count; ++i) {
            const T expected = a[i] + b[i];
            const T scale = std::max(static_cast<T>(1), std::fabs(expected));
            const T tolerance = static_cast<T>(4) *
                                std::numeric_limits<T>::epsilon() * scale;
            assert(std::isfinite(result[i]));
            assert(std::fabs(result[i] - expected) <= tolerance);
        }
    }
}

int main() {
    const size_t vector_sizes[] = {0, 1, 3, 4, 7, 8, 9, 17};
    test_vector_add<float>(vector_sizes, sizeof(vector_sizes) / sizeof(vector_sizes[0]));
    test_vector_add<double>(vector_sizes, sizeof(vector_sizes) / sizeof(vector_sizes[0]));

    const size_t M = 3, K = 5, N = 11;
    float a[M * K], b[K * N], expected[M * N], result[M * N];
    for (size_t i = 0; i < M * K; ++i) a[i] = static_cast<float>(i % 7) - 3.0f;
    for (size_t i = 0; i < K * N; ++i) b[i] = static_cast<float>(i % 5) - 2.0f;

    XK_gemm_naive(a, b, expected, M, K, N);
    XK_gemm(a, b, result, M, K, N);
    for (size_t i = 0; i < M * N; ++i) assert(std::fabs(expected[i] - result[i]) < 1e-4f);

    XK_gemm_blocked(a, b, result, M, K, N, 7);
    for (size_t i = 0; i < M * N; ++i) assert(std::fabs(expected[i] - result[i]) < 1e-4f);

    XK_gemm_omp(a, b, result, M, K, N, 2);
    for (size_t i = 0; i < M * N; ++i) assert(std::fabs(expected[i] - result[i]) < 1e-4f);

    const int32_t ia[] = {100000, 100000};
    const int32_t ib[] = {100000, -100000, 100000, -100000};
    int64_t ic[2] = {};
    XK_gemm_i32_i64(ia, ib, ic, 1, 2, 2);
    assert(ic[0] == INT64_C(20000000000));
    assert(ic[1] == -INT64_C(20000000000));

}
