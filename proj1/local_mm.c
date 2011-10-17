/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define BM 1
#define BK 1
#define BN 1

    void
report_num_threads(int level)
{ 
#pragma omp single 
    {
        printf("Level %d: number of threads in the team - %d\n", level, omp_get_num_threads()); 
    }
}

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

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

#ifdef USE_MKL

  printf("Using MKL...\n");
  char transa[1] = {'N'};
  char transb[1] = {'N'};

  dgemm(transa,
          transb,
          &m,
          &n,
          &k,
          &alpha,
          A,
          &lda,
          B,
          &ldb,
          &beta,
          C,
          &ldc);

#else

#ifdef USE_NAIVE
  {
      int row, col;
      double dotprod;

      /* Iterate over the columns of C */

      /*#pragma omp parallel for reduction (+:dotprod)*/
      for (col = 0; col < n; col++) {

          /* Iterate over the rows of C */

          /*#pragma omp parallel for reduction (+:dotprod)*/
          for (row = 0; row < m; row++) {

              int k_iter;
              dotprod = 0.0; /* Accumulates the sum of the dot-product */

              /* Iterate over column of A, row of B */

              /*#pragma omp parallel for*/
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
#else

{
    int row;
    int col;
    int i;
    int j;
    int l;

    double tmpSum = 0.0;
    for(row = 0; row < m; row += BM)
    {
        for(col = 0; col < n; col += BN)
        {
            for(i = row; i < MIN(i + BM, m); ++i)
            {
                for(j = 0; j < MIN(j + BN, n); ++j)
                {
                    tmpSum = 0.0;
                    
                    for(l = 0; l < k; ++l)
                    {
                        int a_index, b_index;
                        a_index = (l * lda) + i; /* Compute index of A element */
                        b_index = (j * ldb) + l; /* Compute index of B element */
                        tmpSum += A[a_index] * B[b_index]; /* Compute product of A and B */
                    }
                    
                    int c_index = (j * ldc) + i;
                    C[c_index] = (alpha * tmpSum) + (beta * C[c_index]);
                }
            }
        }

    }




}


#endif

#endif

}
