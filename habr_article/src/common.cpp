#include <random>
#include "common.h"


void gen_data(int M, int N, int K, std::vector<float> &A, std::vector<float> &B, std::vector<float> &C) {
//    std::random_device rd;
    std::mt19937_64 gen(132123);

    A.resize(M * K), B.resize(K * N), C.resize(M * N);
    std::uniform_real_distribution<float> unif(-100, 100);
    for (float &i: A) {
        i = unif(gen);
    }
    for (float &i: B) {
        i = unif(gen);
    }
    for (float &i: C) {
        i = unif(gen);
    }
}