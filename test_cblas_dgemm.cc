#include <cblas.h>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <random>

std::random_device rd;
std::mt19937_64 gen(rd());
	

TEST(OpenBlas, dgemm)
{
	int maxN = 1000;
	std::vector<double> A(maxN * maxN), B(maxN * maxN), C(maxN * maxN);
	std::uniform_real_distribution<double> unif(-100, 100);
	for (int i = 0; i < A.size(); ++i) {
		A[i] = unif(gen);
	}
	for (int i = 0; i < B.size(); ++i) {
		B[i] = unif(gen);
	}
	for (int i = 0; i < C.size(); ++i) {
		C[i] = unif(gen);
	}
	using std::chrono::high_resolution_clock;
    	using std::chrono::duration_cast;
    	using std::chrono::duration;
    	using std::chrono::milliseconds;

	std::vector<int> n = {10, 50, 100, 200, 300, 500, 700, 800, 1000};
	std::cout << "mat_size, duration" << std::endl;
	for (int i: n) {
    		auto t1 = high_resolution_clock::now();
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, i, i, i,
			1.0, A.data(), i, B.data(), i, 1.0, C.data(), i);
		auto t2 = high_resolution_clock::now();

		duration<double, std::milli> ms_double = t2 - t1;

		std::cout << i << ", " << ms_double.count() << "ms" << std::endl;
	}
}
