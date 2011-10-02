/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

/**
 *
 *  Local Matrix Multiply
 *   Computes C = alpha * A * B + beta * C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 *  alpha and beta are double-precision scalars
 *
 *  A, B, and C are matrices of double-precision elements
 *  stored in column-major format 
 *
 *  The output is stored in C
 *  A and B are not modified during computation
 *
 *
 *  m - number of rows of matrix A and rows of C
 *  n - number of columns of matrix B and columns of C
 *  k - number of columns of matrix A and rows of B
 * 
 *  lda, ldb, and ldc specifies the size of the first dimension of the matrices
 *
 **/
void local_mm(const int m, const int n, const int k, const double alpha,
    const double *A, const int lda, const double *B, const int ldb,
    const double beta, double *C, const int ldc) {

  int row, col;

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);


  #pragma omp parallel shared (n, m, k, lda, ldb, ldc, A, B, C, alpha, beta)
  {
      /* Iterate over the columns of C */

      #pragma omp for default (none) private (col)
      for (col = 0; col < n; col++) {

          /* Iterate over the rows of C */

          #pragma omp for default (none) private (row)
          for (row = 0; row < m; row++) {

              int k_iter;
              double dotprod = 0.0; /* Accumulates the sum of the dot-product */

              /* Iterate over column of A, row of B */

              #pragma omp for default (none) reduction (+:dotprod) private (k_iter)
              for (k_iter = 0; k_iter < k; k_iter++) {
                  int a_index, b_index;
                  a_index = (k_iter * lda) + row; /* Compute index of A element */
                  b_index = (col * ldb) + k_iter; /* Compute index of B element */
                  dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
              } /* k_iter */

              int c_index = (col * ldc) + row;
              C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
          } /* row */
      } /* col */

  }
}
