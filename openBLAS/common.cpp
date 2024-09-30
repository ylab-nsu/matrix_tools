#include <random>
#include "common.h"


void gen_data(int N, std::vector<double> &A, std::vector<double> &B, std::vector<double> &C) {
    std::random_device rd;
    std::mt19937_64 gen(rd());

    A.resize(N * N), B.resize(N * N), C.resize(N * N);
    std::uniform_real_distribution<double> unif(-100, 100);
    for (double &i: A) {
        i = unif(gen);
    }
    for (double &i: B) {
        i = unif(gen);
    }
    for (double &i: C) {
        i = unif(gen);
    }
}