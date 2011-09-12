/**
 *  \file local_mm_wrapper.c
 *  \brief C interface for when local_mm is implemented in Fortran
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdlib.h>
#include <stdio.h>

extern void local_mm_(const int *m, const int *n, const int *k,
    const double *alpha, const double *A, const int *lda, const double *B,
    const int *ldb, const double *beta, double *C, const int *ldc);

void local_mm(const int m, const int n, const int k, const double alpha,
    const double *A, const int lda, const double *B, const int ldb,
    const double beta, double *C, const int ldc) {

  local_mm_(&m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  return;

}
