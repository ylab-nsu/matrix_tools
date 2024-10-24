#ifndef HABR_ARTICLE_PROJECT_HABR_DGEMM_H
#define HABR_ARTICLE_PROJECT_HABR_DGEMM_H

void gemm_v0(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v1(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v2(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v3(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v4(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v5(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v6(int M, int N, int K, const float *A, const float *B, float *C);

void gemm_v7(int M, int N, int K, const float *A, const float *B, float *C);

// Add paralleling with OpenMP to gemm_v7
void gemm_v8(int M, int N, int K, const float *A, const float *B, float *C);

#endif //HABR_ARTICLE_PROJECT_HABR_DGEMM_H
