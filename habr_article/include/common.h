#ifndef GEMM_PROJECT_COMMON_H
#define GEMM_PROJECT_COMMON_H

#include <time.h>
#include <chrono>

template<typename F, typename... Args>
extern double funcTime(F func, Args &&... args);

template<typename F, typename... Args>
double funcTime(F func, Args &&... args) {
#ifdef _WIN64
    const auto &start = std::chrono::high_resolution_clock::now();
    std::forward<decltype(func)>(func)(std::forward<decltype(args)>(args)...);
    const auto &stop = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
#else
    struct timespec tp1, tp2;
    clock_gettime(CLOCK_MONOTONIC, &tp1) ;
    std::forward<decltype(func)>(func)(std::forward<decltype(args)>(args)...);
    clock_gettime(CLOCK_MONOTONIC, &tp2) ;
    time_t duration = ((tp2.tv_sec-tp1.tv_sec)*(1000*1000*1000) + (tp2.tv_nsec-tp1.tv_nsec)) / 1000 ;
    return (double)duration * 1e-3;
#endif
};

template<typename F, typename... Args>
extern double min_funcTime(F func, Args &&... args);

template<typename F, typename... Args>
double min_funcTime(int cnt_iter, F func, Args &&... args) {
    double result = funcTime(func, std::forward<decltype(args)>(args)...);
    for (int i = 1; i < cnt_iter; ++i) {
        double duration = funcTime(func, std::forward<decltype(args)>(args)...);
        result = std::min(result, duration);
    }
    return result;
}

void gen_data(int M, int N, int K, std::vector<float> &A, std::vector<float> &B, std::vector<float> &C);

#endif //GEMM_PROJECT_COMMON_H