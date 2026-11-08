#ifndef NVXK_H
#define NVXK_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

void vector_add_avx2(const float* a,const float* b,float* c, size_t n);
void vector_add_avx2(const double* a, const double* b, double* c, size_t n);
bool XK_has_avx2_fma();
void XK_naive(const float* A, const float* B, float* C, size_t N);
void XK_ikj(const float* A, const float* B, float* C, size_t N);
void XK_avx2(const float* A,const float* B,float* C,size_t N);
void XK_avx2_omp(const float* A, const float* B, float* C, size_t N, int thread_count);
void XK_auto(const float* A, const float* B, float* C, size_t N);
void XK_omp(const float* A, const float* B, float* C, size_t N, int thread_count);

void XK_blocked(const float* A, const float* B, float* C, size_t N);
void XK_blocked(const float* A, const float* B, float* C, size_t N, size_t block_size);
void XK_gemm_avx2(const float* A, const float* B, float* C,
                  size_t M, size_t K, size_t N);
void XK_gemm_avx2(const double* A, const double* B, double* C,
                  size_t M, size_t K, size_t N);
void XK_gemm_blocked(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N, size_t block_size = 64);
void XK_gemm_omp(const float* A, const float* B, float* C,
                 size_t M, size_t K, size_t N, int thread_count);
void XK_gemm(const float* A, const float* B, float* C,
             size_t M, size_t K, size_t N);
void XK_gemm(const double* A, const double* B, double* C,
             size_t M, size_t K, size_t N);
void XK_gemm_i32_i64(const int32_t* A, const int32_t* B, int64_t* C,
                     size_t M, size_t K, size_t N);

namespace xk_detail {

template<typename T>
struct accumulator_type {
    typedef typename std::conditional<
        std::is_integral<T>::value && (sizeof(T) <= sizeof(int)),
        int64_t, T>::type type;
};

inline size_t checked_element_count(size_t rows, size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<size_t>::max() / columns) {
        throw std::length_error("matrix dimensions are too large");
    }
    return rows * columns;
}

template<typename T>
void store_result(typename accumulator_type<T>::type value, T& output,
                  typename std::enable_if<std::is_integral<T>::value>::type* = 0) {
    if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
        throw std::overflow_error("integer matrix result is out of range");
    }
    output = static_cast<T>(value);
}

template<typename T>
void store_result(typename accumulator_type<T>::type value, T& output,
                  typename std::enable_if<!std::is_integral<T>::value>::type* = 0) {
    output = static_cast<T>(value);
}

inline int64_t add_checked(int64_t sum, int64_t product) {
    if ((product > 0 && sum > std::numeric_limits<int64_t>::max() - product) ||
        (product < 0 && sum < std::numeric_limits<int64_t>::min() - product)) {
        throw std::overflow_error("integer matrix accumulation is out of range");
    }
    return sum + product;
}

template<typename T>
typename accumulator_type<T>::type multiply_add(
    typename accumulator_type<T>::type sum, T left, T right,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0) {
    const int64_t product = static_cast<int64_t>(left) * static_cast<int64_t>(right);
    return add_checked(sum, product);
}

template<typename T>
typename accumulator_type<T>::type multiply_add(
    typename accumulator_type<T>::type sum, T left, T right,
    typename std::enable_if<!std::is_integral<T>::value>::type* = 0) {
    return sum + left * right;
}

}

template<typename T>
void XK_gemm_naive(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    static_assert(std::is_floating_point<T>::value ||
                  (std::is_integral<T>::value && std::is_signed<T>::value &&
                   sizeof(T) <= sizeof(int)),
                  "XK_gemm supports floating-point and signed integer types up to int");
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    xk_detail::checked_element_count(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            typename xk_detail::accumulator_type<T>::type sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum = xk_detail::multiply_add(sum, A[i * K + k], B[k * N + j]);
            }
            xk_detail::store_result(sum, C[i * N + j]);
        }
    }
}

template<typename T>
void XK_gemm_ikj(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    static_assert(std::is_floating_point<T>::value ||
                  (std::is_integral<T>::value && std::is_signed<T>::value &&
                   sizeof(T) <= sizeof(int)),
                  "XK_gemm supports floating-point and signed integer types up to int");
    const size_t output_count = xk_detail::checked_element_count(M, N);
    xk_detail::checked_element_count(M, K);
    xk_detail::checked_element_count(K, N);
    std::vector<typename xk_detail::accumulator_type<T>::type> sums(output_count, 0);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const T a_ik = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                typename xk_detail::accumulator_type<T>::type& sum = sums[i * N + j];
                sum = xk_detail::multiply_add(sum, a_ik, B[k * N + j]);
            }
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        xk_detail::store_result(sums[i], C[i]);
    }
}

template<typename T>
void XK_gemm(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    XK_gemm_ikj(A, B, C, M, K, N);
}

#endif