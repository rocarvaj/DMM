/**
 *  \file summa.c
 *  \brief Implementation of Scalable Universal 
 *    Matrix Multiplication Algorithm for Proj1
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#include "local_mm.h"

/**
 * Distributed Matrix Multiply using the SUMMA algorithm
 *  Computes C = A*B + C
 * 
 *  This function uses procGridX times procGridY processes
 *   to compute the product
 *  
 *  A is a m by k matrix, each process starts
 *	with a block of A (aBlock) 
 *  
 *  B is a k by n matrix, each process starts
 *	with a block of B (bBlock) 
 *  
 *  C is a n by m matrix, each process starts
 *	with a block of C (cBlock)
 *
 *  The resulting matrix is stored in C.
 *  A and B should not be modified during computation.
 * 
 *  Ablock, Bblock, and CBlock are stored in
 *   column-major format  
 *
 *  pb is the Panel Block Size
 **/
void summa(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
		int procGridX, int procGridY, int pb) {



	MPI_Barrier(MPI_COMM_WORLD);

}
