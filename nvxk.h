#ifndef NVXK_H
#define NVXK_H

#include <cstddef>


void vector_add_avx2(const float* a,const float* b,float* c, size_t n);
bool XK_has_avx2_fma();
void XK_naive(const float* A, const float* B, float* C, size_t N);
void XK_ikj(const float* A, const float* B, float* C, size_t N);
void XK_avx2(const float* A,const float* B,float* C,size_t N);
void XK_avx2_omp(const float* A, const float* B, float* C, size_t N, int thread_count);
void XK_auto(const float* A, const float* B, float* C, size_t N);
void XK_omp(const float* A, const float* B, float* C, size_t N, int thread_count);

void XK_blocked(const float* A, const float* B, float* C, size_t N);

#endif 