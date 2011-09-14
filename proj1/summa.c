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

    int i;
    int p;
    int rank = 0;
    int indexX = rank % procGridX;
    int indexY = (rank - indexX) / procGridX;
    int rowGroupIndex[procGridY];
    int colGroupIndex[procGridX];
    int whoseTurnRow;
    int whoseTurnCol;

    MPI_Group originalGroup, rowGroup, colGroup;
    MPI_Comm rowComm, colComm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &originalGroup);

    for(p = 0; p < procGridY; ++p)
        rowGroupIndex[p] = p * procGridX + indexX;

    for(p = 0; p < procGridX; ++p)
        colGroupIndex[p] = indexY * procGridX + p;

    /* Create groups */
    MPI_Group_incl(originalGroup, procGridY, rowGroupIndex, &rowGroup);
    MPI_Group_incl(originalGroup, procGridX, colGroupIndex, &colGroup);

    /* Create communicators */
    MPI_Comm_create(MPI_COMM_WORLD, rowGroup, &rowComm);
    MPI_Comm_create(MPI_COMM_WORLD, colGroup, &colComm);

    for(i = 0; i < k/pb; ++i)
    {
        whoseTurnRow = (int) i * pb * procGridX / k;
        whoseTurnCol = (int) i * pb * procGridY / k;


        /* Row */
        if(rank == whoseTurnRow)
        {
            int buffer = 1;
            MPI_Bcast(&buffer, sizeof(int), MPI_INT, whoseTurnRow, rowComm);

            printf("I'm proc: %d, and sent message!\n", rank);
        }
        else
        {
            int buffer;
            MPI_Bcast(&buffer, sizeof(int), MPI_INT, whoseTurnRow, rowComm);
            printf("I'm proc: %d, and got message: %d\n", rank, buffer);

        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

}
