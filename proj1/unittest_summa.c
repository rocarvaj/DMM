/**
 *  \file unittest_summa.c
 *  \brief unittests for summa()
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

#include "matrix_utils.h"
#include "local_mm.h"
#include "summa.h"

#define true 1
#define false 0
#define bool char

#define EPS 0.0001

/** 
 * Similar to verify_matrix(),
 *  this function verifies that each element of A
 *  matches the corresponding element of B
 *
 *  returns true if A and B are equal
 */
bool verify_matrix_bool(int m, int n, double *A, double *B) {

  /* Loop over every element of A and B */
  int row, col;
  for (col = 0; col < n; col++) {
    for (row = 0; row < m; row++) {
      int index = (col * m) + row;
      double a = A[index];
      double b = B[index];

      if (a < b - EPS) {
        return false;
      }
      if (a > b + EPS) {
        return false;
      }
    } /* row */
  }/* col */

  return true;
}


/**
 * Creates random A, B, and C matrices and uses summa() to
 *  calculate the product. Output of summa() is compared 
 *  to CC, the true solution.
 **/
bool random_matrix_test(int m, int n, int k, int px, int py, int panel_size) {
  int proc = 0, passed_test = 0, group_passed = 0;
  int num_procs = px * py;
  int rank = 0;
  double *A, *B, *C, *CC, *A_block, *B_block, *C_block, *CC_block;

  A = NULL;
  B = NULL;
  C = NULL;
  CC = NULL;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */

  if (rank == 0) {
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

    /* 
     * blocks of A, B, C, and CC have been distributed to
     * each of the processes, now we can safely deallocate the 
     * matrices
     */
    deallocate_matrix(A);
    deallocate_matrix(B);
    deallocate_matrix(C);
    deallocate_matrix(CC);
  }

#ifdef DEBUG
  /* flush output and synchronize the processes */
  fflush(stdout);
  sleep(1);
  MPI_Barrier( MPI_COMM_WORLD);
#endif

  /* 
   *
   * Call SUMMA
   *
   */

  summa(m, n, k, A_block, B_block, C_block, px, py, 1);

#ifdef DEBUG
  /* flush output and synchronize the processes */
  fflush(stdout);
  sleep(1);
  MPI_Barrier( MPI_COMM_WORLD);
#endif

#ifdef DEBUG
  /* Verify each C_block sequentially */
  for (proc=0; proc < num_procs; proc++) {

    if (rank == proc) {

      bool isCorrect = verify_matrix_bool(m / px, n / py, C_block, CC_block);

      if (isCorrect) {
        printf("CBlock on rank=%d is correct\n",rank);
        fflush(stdout);
      } else {
        printf("**\tCBlock on rank=%d is wrong\n",rank);

        printf("CBlock on rank=%d is\n",rank);
        print_matrix(m / px, n / py, C_block);

        printf("CBlock on rank=%d should be\n",rank);
        print_matrix(m / px, n / py, CC_block);

        printf("**\n\n");
        fflush(stdout);

        passed_test = 1;
        sleep(1);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD); /* keep all processes synchronized */
  }/* proc */

#else

  /* each process will verify its C_block in parallel */
  if (verify_matrix_bool(m / px, n / py, C_block, CC_block) == false) {
    passed_test = 1;
  }

#endif

  /* free A_block, B_block, C_block, and CC_block */
  free(A_block);
  free(B_block);
  free(C_block);
  free(CC_block);

  /*
   *
   *  passed_test == 0 if the process PASSED the test
   *  passed_test == 1 if the process FAILED the test
   *  
   *  therefore a MPI_Reduce of passed_test will count the
   *   number of processes that failed
   *  
   *  After the MPI_Reduce/MPI_Scatter, if group_passed == 0 then every process passed
   */

  MPI_Reduce(&passed_test, &group_passed, 1, MPI_INT, MPI_SUM, 0,
      MPI_COMM_WORLD);
  MPI_Bcast(&group_passed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0 && group_passed == 0) {
    printf(
        "random_matrix_test m=%d n=%d k=%d px=%d py=%d pb=%d............PASSED\n",
        m, n, k, px, py, panel_size);
  }

  if (rank == 0 && group_passed != 0) {
    printf(
        "random_matrix_test m=%d n=%d k=%d px=%d py=%d pb=%d............FAILED\n",
        m, n, k, px, py, panel_size);
  }

  /* If group_passed==0 then every process passed the test*/
  if (group_passed == 0) {
    return true;
  } else {
    return false;
  }
}

#ifdef DEBUG
#  define exit_on_fail(passed) if (passed == false) { goto finalize; }
#else
#  define exit_on_fail(passed) (passed)
#endif

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
  exit_on_fail( random_matrix_test(16, 16, 16, 4, 4, 1));
  exit_on_fail( random_matrix_test(32, 32, 32, 4, 4, 1));
  exit_on_fail( random_matrix_test(128, 128, 128, 4, 4, 1));
  
  /* Test different shapes */
  exit_on_fail( random_matrix_test(128, 32, 128, 4, 4, 1));
  exit_on_fail( random_matrix_test(64, 32, 128, 4, 4, 1));

  /* Test different process grids */
  exit_on_fail( random_matrix_test(128, 128, 128, 8, 2, 1));
  exit_on_fail( random_matrix_test(128, 128, 128, 2, 8, 1));
  exit_on_fail( random_matrix_test(128, 128, 128, 1, 16, 1));
  exit_on_fail( random_matrix_test(128, 128, 128, 16, 1, 1));
  
finalize: MPI_Finalize();
  return 0;
}
