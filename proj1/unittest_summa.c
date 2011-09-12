/**
 *  \file unittest_summa.c
 *  \brief unittests for summa() (summa.c/summa.f90)s
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "summa.h"

void random_matrix_test(int m, int n, int k, int px, int py, int panel_size) {
  int num_procs = px * py;
  int rank = 0;
  double *A, *B, *C, *CC, *A_block, *B_block, *C_block, *CC_block;

  A = NULL;
  B = NULL;
  C = NULL;
  CC = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */

  if (rank == 0) {
    printf("random_matrix_test m=%d n=%d k=%d px=%d py=%d............", m, n,
        k, px, py);

    /* Allocate matrices */
    A = random_matrix(m, k);
    B = random_matrix(k, n);
    C = zeros_matrix(m, n);

    /* Stores the solution */
    CC = zeros_matrix(m, n);

    /* 
     * Solve the problem locally and store the
     *  solution in CC
     */
    local_mm(m, n, k, 1.0, A, m, B, k, 0.0, CC, m);
  }

  /* 
   * Allocate memory for matrix blocks 
   */
  A_block = malloc(sizeof(double) * (m * k) / num_procs);
  assert(A_block);

  B_block = malloc(sizeof(double) * (k * n) / num_procs);
  assert(B_block);

  C_block = malloc(sizeof(double) * (m * n) / num_procs);
  assert(C_block);

  CC_block = malloc(sizeof(double) * (m * n) / num_procs);
  assert(CC_block);

  /* Distrute the matrices */
  distribute_matrix(px, py, m, k, A, A_block, rank);
  distribute_matrix(px, py, k, n, B, B_block, rank);
  distribute_matrix(px, py, m, n, C, C_block, rank);
  distribute_matrix(px, py, m, n, CC, CC_block, rank);

  if (rank == 0) {

    /* deallocate memory */
    deallocate_matrix(A);
    deallocate_matrix(B);
    deallocate_matrix(C);
    deallocate_matrix(CC);

    printf("passed\n");
  }

  /* 
   *
   * Call SUMMA
   *
   */

  summa(m, n, k, A_block, B_block, C_block, px, py, 1);

  MPI_Barrier( MPI_COMM_WORLD);

  /* Verfiy the results */
  verify_matrix(m / px, n / py, C_block, CC_block);

  MPI_Barrier(MPI_COMM_WORLD);

  free(A_block);
  free(B_block);
  free(C_block);
}

/** Program start */
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

  /* These tests use 16 processes */
  if (np != 16) {
    printf("Error: np=%d. Please use 16 processes\n", np);
  }

  /** Test different sizes */
  random_matrix_test(16, 16, 16, 4, 4, 1);
  random_matrix_test(32, 32, 32, 4, 4, 1);
  random_matrix_test(128, 128, 128, 4, 4, 1);

  /* Test different shapes */
  random_matrix_test(128, 32, 128, 4, 4, 1);
  random_matrix_test(64, 32, 128, 4, 4, 1);

  /* Test different process grids */
  random_matrix_test(128, 128, 128, 8, 2, 1);
  random_matrix_test(128, 128, 128, 2, 8, 1);
  random_matrix_test(128, 128, 128, 1, 16, 1);
  random_matrix_test(128, 128, 128, 16, 1, 1);

  MPI_Finalize();
  return 0;
}
