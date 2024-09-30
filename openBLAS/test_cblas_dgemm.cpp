#include <cblas.h>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <random>
#include "common.h"


TEST(OpenBlas, dgemm) {
    int maxN = 1000;
    std::vector<double> A, B, C;
    gen_data(maxN, A, B, C);

    std::vector<int> n = {10, 50, 100, 200, 300, 500, 700, 800, 1000};
    std::cout << "mat_size, duration" << std::endl;
    for (int i: n) {
        auto duration = min_funcTime(5, cblas_dgemm, CblasColMajor, CblasNoTrans, CblasNoTrans, i, i, i, 1.0, A.data(),
                                     i, B.data(), i, 1.0, C.data(), i);
        std::cout << i << ", " << duration << " ms" << std::endl;
    }
}
