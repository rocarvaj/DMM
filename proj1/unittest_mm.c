/**
 *  \file unittest_mm.c
 *  \brief unittests for testing local_mm()
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "matrix_utils.h"
#include "local_mm.h"

void print_matrix_types() {

  double *random, *ones, *zeros, *identity, *tri;

  /* Allocate matrices */
  random = random_matrix(6, 3);
  identity = identity_matrix(5, 5);
  ones = ones_matrix(4, 2);
  zeros = zeros_matrix(2, 4);
  tri = lowerTri_matrix(5, 5);

  printf("\n\t\tMatrix Types\n");

  printf("6x3 Random Matrix\n");
  print_matrix(6, 3, random);
  printf("\n\n");

  printf("5x5 Identity Matrix\n");
  print_matrix(5, 5, identity);
  printf("\n\n");

  printf("4x2 Ones Matrix\n");
  print_matrix(4, 2, ones);
  printf("\n\n");

  printf("2x4 Zeros Matrix\n");
  print_matrix(2, 4, zeros);
  printf("\n\n");

  printf("5x5 Lower Triangular Matrix\n");
  print_matrix(5, 5, tri);
  printf("\n\n");

  /* deallocate memory */
  deallocate_matrix(random);
  deallocate_matrix(ones);
  deallocate_matrix(zeros);
  deallocate_matrix(identity);
  deallocate_matrix(tri);

}

/**
 * Verify that a matrix times the identity is itself
 **/
void identity_test(int n) {
  double *A, *B, *C;

  printf("identity_test n=%d............", n);

  /* Allocate matrices */
  A = random_matrix(n, n);
  B = identity_matrix(n, n);
  C = zeros_matrix(n, n);

  /* C = 1.0*(A*B) + 0.0*C */
  local_mm(n, n, n, 1.0, A, n, B, n, 5.0, C, n);

  /* Verfiy the results */
  verify_matrix(n, n, A, C);

  /* Backwards C = 1.0*(B*A) + 0.0*C */
  local_mm(n, n, n, 1.0, B, n, A, n, 0.0, C, n);

  /* Verfiy the results */
  verify_matrix(n, n, A, C);

  /* deallocate memory */
  deallocate_matrix(A);
  deallocate_matrix(B);
  deallocate_matrix(C);

  printf("passed\n");
}

/**
 * Test the multiplication of two matrices of all ones
 **/
void ones_test(int m, int n, int k) {
  double *A, *B, *C, *C_ones, *C_zeros;

  printf("ones_test m=%d n=%d k=%d............", m, n, k);

  /* Allocate matrices */
  A = ones_matrix(m, k);
  B = ones_matrix(k, n);
  C = ones_matrix(m, n);

  C_ones = ones_matrix(m, n);
  C_zeros = zeros_matrix(m, n);

  /* C = (1.0/k)*(A*B) + 0.0*C */
  local_mm(m, n, k, (1.0 / k), A, m, B, k, 0.0, C, m);

  /* Verfiy the results */
  verify_matrix(m, n, C, C_ones);

  /* C = (1.0/k)*(A*B) + -1.0*C */
  local_mm(m, n, k, (1.0 / k), A, m, B, k, -1.0, C, m);

  /* Verfiy the results */
  verify_matrix(m, n, C, C_zeros);

  /* deallocate memory */
  deallocate_matrix(A);
  deallocate_matrix(B);
  deallocate_matrix(C);

  deallocate_matrix(C_ones);
  deallocate_matrix(C_zeros);

  printf("passed\n");
}

/**
 * Test the multiplication of a lower triangular matrix
 **/
void lower_triangular_test(int n) {

  int i;
  double *A, *B, *C;

  printf("lower_triangular_test n=%d ............", n);

  /* Allocate matrices */
  A = lowerTri_matrix(n, n);
  B = ones_matrix(n, 1);
  C = ones_matrix(n, 1);

  /* C = 1.0*(A*B) + 0.0*C */
  local_mm(n, 1, n, 1.0, A, n, B, n, 0.0, C, n);

  /* Loops over every element in C */
  for (i = 0; i < n; i++) {
    /* The elements of C should be [1 2 3 4...] */
    assert(C[i] == (double) (i + 1.0));
  }

  /* C = 0.0*(A*B) + 1.0*C */
  local_mm(n, 1, n, 0.0, A, n, B, n, 1.0, C, n);

  /* Loops over every element in C */
  for (i = 0; i < n; i++) {
    /* The elements of C should be [1 2 3 4...] */
    assert(C[i] == (double) (i + 1.0));
  }

  /* C = 1.0*(A*B) + 1.0*C */
  local_mm(n, 1, n, 3.0, A, n, B, n, 1.0, C, n);

  /* Loops over every element in C */
  for (i = 0; i < n; i++) {
    /* The elements of C should be [4 8 16 32...] */
    assert(C[i] == (double) ((i + 1) * 4.0));
  }

  /* C = 0.0*(A*B) + 0.0*C */
  local_mm(n, 1, n, 0.0, A, n, B, n, 0.0, C, n);

  /* Loops over every element in C */
  for (i = 0; i < n; i++) {
    /* The elements of C should be 0 */
    assert(C[i] == 0.0);
  }

  /* deallocate memory */
  deallocate_matrix(A);
  deallocate_matrix(B);
  deallocate_matrix(C);

  printf("passed\n");
}

int main() {

  printf("Hello World\n");

  identity_test(16);
  identity_test(37);
  identity_test(512);
  ones_test(32, 32, 32);
  ones_test(61, 128, 123);
  lower_triangular_test(8);
  lower_triangular_test(92);
  lower_triangular_test(128);

  return 0;
}

