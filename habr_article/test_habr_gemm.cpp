#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "common.h"

typedef void(*gemm_interface)(int M, int N, int K, const float *A, const float *B, float *C);

void test_gemm(gemm_interface gemm, int M, int N, int K) {
    std::vector<float> A, B, C;
    gen_data(M, N, K, A, B, C);

    std::cout << "M, N, K, duration" << std::endl;
    auto duration = min_funcTime(50, gemm, M, N, K, A.data(), B.data(), C.data());
    std::cout << M << ", " << N << ", " << K << ", " << duration << " ms" << std::endl;
}

void gemm_v0(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v0) {
    test_gemm(gemm_v0, 1152, 1152, 1152);
}

void gemm_v1(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v1) {
    test_gemm(gemm_v1, 1152, 1152, 1152);
}

void gemm_v2(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v2) {
    test_gemm(gemm_v2, 1152, 1152, 1152);
}

void gemm_v3(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v3) {
    test_gemm(gemm_v3, 1152, 1152, 1152);
}

void gemm_v4(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v4) {
    test_gemm(gemm_v4, 1152, 1152, 1152);
    test_gemm(gemm_v4, 1152, 1152, 115200);
}

void gemm_v5(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v5) {
    test_gemm(gemm_v5, 1152, 1152, 1152);
    test_gemm(gemm_v4, 1152, 1152, 115200);
}

void gemm_v6(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v6) {
    test_gemm(gemm_v6, 1152, 1152, 1152);
    test_gemm(gemm_v4, 1152, 1152, 115200);
}

void gemm_v7(int M, int N, int K, const float *A, const float *B, float *C);

TEST(habr_dgemm, gemm_v7) {
    test_gemm(gemm_v7, 1152, 1152, 1152);
    test_gemm(gemm_v4, 1152, 1152, 115200);
}
