#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "common.h"
#include "habr_dgemm.h"
#include <cblas.h>

typedef void(*gemm_interface)(int M, int N, int K, const float *A, const float *B, float *C);

struct matrice_sizes {
    int m, n, k;
};

struct gemm_function {
    std::string name;
    gemm_interface func;
};

void test_gemm(gemm_function &gemm, matrice_sizes &sizes, int cnt_iter = 50) {
    std::vector<float> A, B, C;
    gen_data(sizes.m, sizes.n, sizes.k, A, B, C);

    auto duration = min_funcTime(cnt_iter, gemm.func, sizes.m, sizes.n, sizes.k, A.data(), B.data(), C.data());
    long long number_of_operations = 2LL * sizes.m * sizes.n * sizes.k;
    std::cout << gemm.name << ", " << std::flush;
    std:: cout << duration << ", " << (double) number_of_operations / (duration / 1000) / 1e9 << std::endl;
}

void gemm_openblas(int M, int N, int K, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, M, B, K, 0.0, C, M);
}

#define name_and_func(func) #func, func

gemm_function functions[] = {{name_and_func(gemm_v0)},
                             {name_and_func(gemm_v1)},
                             {name_and_func(gemm_v2)},
                             {name_and_func(gemm_v3)},
                             {name_and_func(gemm_v4)},
                             {name_and_func(gemm_v5)},
                             {name_and_func(gemm_v6)},
                             {name_and_func(gemm_v7)},
                             {name_and_func(gemm_v8)},
                             {name_and_func(gemm_openblas)},
};

matrice_sizes small_test{1152, 1152, 1152}, big_test{1152, 1152, 115200};



TEST(habr_dgemm, small_test) {
    openblas_set_num_threads(1);

    std::cout << "Test with product of matrix of size " << small_test.m << "x" << small_test.k << " by "
              << small_test.k << "x" << small_test.n << std::endl;

    std::cout << "-------\n" << "function name, duration (ms), GFLOPS" << std::endl;

    for (auto &gemm_function: functions) {
        test_gemm(gemm_function, small_test);
    }
}

TEST(habr_dgemm, big_test) {
    openblas_set_num_threads(1);

    std::cout << "Test with product of matrix of size " << big_test.m << "x" << big_test.k << " by "
              << big_test.k << "x" << big_test.n << std::endl;

    std::cout << "-------\n" << "function name, duration (ms), GFLOPS" << std::endl;

    for (auto &gemm_function: functions) {
        if (gemm_function.func == gemm_v0) {
//            test_gemm(gemm_function, big_test, 1);
        } else if (gemm_function.func != gemm_openblas){
            test_gemm(gemm_function, big_test, 10);
        } else {
            test_gemm(gemm_function, big_test);
        }
    }
}
