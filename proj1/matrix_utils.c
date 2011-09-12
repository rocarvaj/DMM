/**
 *  \file matrix_utils.c
 *  \brief Matrix Utility Functions for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

/**
 * Matrix Utility Functions
 *  
 * rows - the number of rows
 * cols - the number of cols
 * mat - a rows by cols matrix of double-precision elements
 *   in column-major order  
 **/

/**
 * Allocates a matrix
 **/
double *allocate_matrix(int rows, int cols) {
  double *mat = NULL;
  mat = malloc(sizeof(double) * rows * cols);
  assert(mat != NULL);
  return (mat);
}

/**
 * Deallocates a matrix
 **/
void deallocate_matrix(double *mat) {
  free(mat);
}

/**
 * Print the elements of the matrix
 **/
void print_matrix(int rows, int cols, double *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      printf("%.1lf ", mat[index]);
    } /* c */
    printf("\n");
  } /* r */
}

/**
 * Set the elements of the matrix to random values
 **/
double *random_matrix(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = round(10.0 * rand() / (RAND_MAX + 1.0));
    } /* r */
  } /* c */

  return mat;
}

/**
 * Set the elements of the matrix to random values
 **/
double *random_matrix_bin(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = round(rand() / (RAND_MAX + 1.0));
    } /* r */
  } /* c */

  return mat;
}

/**
 * Sets each element of the matrix to 1
 **/
double *ones_matrix(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = 1.0;
    } /* r */
  } /* c */

  return mat;
}

/**
 * Sets each element of the matrix to 1
 **/
double *zeros_matrix(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = 0.0;
    } /* r */
  } /* c */

  return mat;
}

/**
 * Sets each element of the diagonal to 1, 0 otherwise
 **/
double *identity_matrix(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      if (r == c) {
        mat[index] = 1.0;
      } else {
        mat[index] = 0.0;
      }
    } /* r */
  } /* c */

  return mat;
}

/**
 * Sets each element of the diagonal and every element
 *  under the diagonal to 1, 0 otherwise
 **/
double *lowerTri_matrix(int rows, int cols) {

  int r, c;
  double *mat = allocate_matrix(rows, cols);

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      if (r >= c) {
        mat[index] = 1.0;
      } else {
        mat[index] = 0.0;
      }
    } /* r */
  } /* c */

  return mat;
}

/**
 * Write a matrix to a CSV file
 **/
void write_csv(int rows, int cols, double *mat, char *filename) {

  FILE *fp = NULL;
  int r, c;

  /* Open a file for writing */
  fp = fopen(filename, "w");
  assert(fp != NULL);

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      fprintf(fp, "%lf,", mat[index]);
      fflush(fp);
    } /* c */
    printf("\n");
  } /* r */

  fclose(fp); /* Close file */
}

/**
 * Copy a block of a matrix mat to dest
 *  
 * mat is a m by n matrix
 * block size is determined by procGridX and procGridY
 * rank is used to pick the block to copy 
 */
void copy_block(int procGridX, int procGridY, int rank, int n, int m,
    double *mat, double *dest) {

  int row, col;
  int block_index = 0;

  int block_rows = n / procGridX;
  int block_cols = m / procGridY;

  int proc_x = rank % procGridX;
  int proc_y = (rank - proc_x) / procGridX;

  /**
   * Assume matrix dimensions are divisible by
   *  process grid dimensions 
   **/
  assert(n % procGridX == 0);
  assert(m % procGridY == 0);

  /* Loop over the columns in the block*/
  for (col = proc_y * block_cols; col < (proc_y + 1) * block_cols; col++) {

    /* Loop over the rows in the block*/
    for (row = proc_x * block_rows; row < (proc_x + 1) * block_rows; row++) {
      int mat_index = (col * n) + row;
      dest[block_index] = mat[mat_index];
      block_index++;
    } /* row */
  } /* col */
}

/**
 * Reoder a matrix so that block elements are contiguous 
 * 
 * src is the original matrix 
 * dest is the reordered matrix
 */
void reorder_matrix(int procGridX, int procGridY, int n, int m, double *src,
    double *dest) {

  int block;
  int num_blocks = procGridX * procGridY;
  int block_size = m * n / num_blocks;

  /**
   * Assume matrix dimensions are divisible by
   *  process grid dimensions 
   **/
  assert(n % procGridX == 0);
  assert(m % procGridY == 0);

  /* Loop over all blocks */
  for (block = 0; block < num_blocks; block++) {
    copy_block(procGridX, procGridY, block, n, m, src, &(dest[block_size
        * block]));
  } /* block */
}

/**
 * Distributes the a blocks of the matrix 
 *  to each process
 * 
 * The full matrix (mat) starts on proc=0,
 *  MPI_Scatter is used to deliver the
 *  appropriate block to each process
 *  
 * The appropiate block of the matrix
 *  is saved to the block buffer
 */
void distribute_matrix(int procGridX, int procGridY, int n, int m, double *mat,
    double *block, int rank) {

  double *buffer = NULL;

  int num_procs = procGridX * procGridY;
  int block_size = m * n / num_procs;

  if (rank == 0) {
    /* Allocate a buffer for the reordered matrix */
    buffer = malloc(sizeof(double) * m * n);
    assert(buffer != NULL);

    reorder_matrix(procGridX, procGridY, n, m, mat, buffer);
  }

  MPI_Scatter(buffer, block_size, MPI_DOUBLE, block, block_size, MPI_DOUBLE, 0,
      MPI_COMM_WORLD);

  if (rank == 0) {
    free(buffer);
  }
}

#define EPSILON 0.00001

/**
 * Verfies that two numbers are REASONABLY close
 **/
void verify_element(double a, double b) {

  assert(a < (b + EPSILON));
  assert(a > (b - EPSILON));
}

/**
 * Verfies that each element in A is REASONABLY close
 *  to the corresponding element in B
 **/
void verify_matrix(int m, int n, double *A, double *B) {
  int i;

  for (i = 0; i < m * n; i++) {
    verify_element(A[i], B[i]);
  }
}

/**
 * Allocate buffers and distribute the matrix
 **/
void allocate_and_distribute(double *mat, double *block, int m, int n,
    int procGridX, int procGridY, int rank) {

  int num_procs = procGridX * procGridY;

  block = malloc(sizeof(double) * (m * n) / num_procs);
  assert(block);

  /* Use MPI to distribute the matrix */
  distribute_matrix(procGridX, procGridY, m, n, mat, block, rank);
}
