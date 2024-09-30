void
slow_dgemm(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c[j * ldc + i] *= beta;
            for (int w = 0; w < k; ++w) {
                c[j * ldc + i] += a[w * lda + i] * b[j * ldb + w] * alpha;
            }
        }
    }
}
