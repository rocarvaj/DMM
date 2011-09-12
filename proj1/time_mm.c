/**
 *  \file time_mm.c
 *  \brief code for timing local_mm()
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "matrix_utils.h"
#include "local_mm.h"

#define NUM_TRIALS 25 /*!< Number of timing trials */

/**
 * Test the multiplication of two matrices of all ones
 **/
void random_multiply(int m, int n, int k, int iterations) {
  int iter;
  double *A, *B, *C;
  double t_start, t_elapsed;

  printf("Timing Matrix Multiply m=%d n=%d k=%d iterations=%d....", m, n, k,
      iterations);

  /* Allocate matrices */
  A = random_matrix(m, k);
  B = random_matrix(k, n);
  C = random_matrix(m, n);

  t_start = MPI_Wtime(); /* Start timer */

  /* perform several Matric Mulitplies back-to-back */
  for (iter = 0; iter < iterations; iter++) {
    /* C = (1.0/k)*(A*B) + 0.0*C */
    local_mm(m, n, k, 1.0, A, m, B, k, 1.0, C, m);
  } /* iter */

  t_elapsed = MPI_Wtime() - t_start; /* Stop timer */

  /* deallocate memory */
  deallocate_matrix(A);
  deallocate_matrix(B);
  deallocate_matrix(C);

  printf("total_time=%lf, per_iteration=%lf\n", t_elapsed, t_elapsed
      / iterations);
}

int main(int argc, char *argv[]) {

  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME + 1];
  int namelen = 0;

  MPI_Init(&argc, &argv); /* starts MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */
  MPI_Comm_size(MPI_COMM_WORLD, &np); /* Get number of processes */
  MPI_Get_processor_name(hostname, &namelen); /* Get hostname of node */
  printf("[Using Host:%s -- Rank %d out of %d]\n", hostname, rank, np);

  if (rank == 0) {
    random_multiply(512, 512, 512, NUM_TRIALS);
  }

  MPI_Finalize();
  return 0;
}

