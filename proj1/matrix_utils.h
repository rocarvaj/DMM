/**
 *  \file matrix_utils.h
 *  \brief Matrix Utility Functions for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>
 */


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
double *allocate_matrix(int rows, int cols);

/**
 * Deallocates a matrix
 **/
void deallocate_matrix(double *mat);

/**
 * Print the elements of the matrix
 **/
void print_matrix(int rows, int cols, double *mat);

/**
 * Set the elements of the matrix to random values
 **/
double *random_matrix(int rows, int cols);

/**
 * Set the elements of the matrix to random values
 **/
double *random_matrix_bin(int rows, int cols);

/**
 * Sets each element of the matrix to 1
 **/
double *ones_matrix(int rows, int cols);

/**
 * Sets each element of the matrix to 1
 **/
double *zeros_matrix(int rows, int cols);

/**
 * Sets each element of the diagonal to 1, 0 otherwise
 **/
double *identity_matrix(int rows, int cols);

/**
 * Sets each element of the diagonal and every element
 *  under the diagonal to 1, 0 otherwise
 **/
double *lowerTri_matrix(int rows, int cols);

/**
 * Write matrix to a csv file
 */
void write_csv(int rows, int cols, double *mat, char *filename);

/**
 * Copy a block of a matrix mat to dest
 *  
 * mat is a m by n matrix
 * block size is determined by procGridX and procGridY
 * rank is used to pick the block to copy 
 */
void copy_block(int procGridX, int procGridY, int rank, int n, int m,
    double *mat, double *dest);

/**
 * Reoder a matrix so that block elements are contiguous 
 * 
 * src is the original matrix 
 * dest is the reordered matrix
 */
void reorder_matrix(int procGridX, int procGridY, int n, int m, double *src,
    double *dest);

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
    double *block, int rank);

/**
 * Verifies that two numbers are REASONABLY close
 **/
void verify_element(double a, double b);

/**
 * Verifies that each element in A is REASONABLY close
 *  to the corresponding element in B
 **/
void verify_matrix(int m, int n, double *A, double *B);
