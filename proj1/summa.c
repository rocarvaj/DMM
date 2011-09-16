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
    int rankRow = 0;
    int rankCol = 0;
    int indexX;
    int indexY;
    int rowGroupIndex[procGridY];
    int colGroupIndex[procGridX];
    int whoseTurnRow;
    int whoseTurnCol;

    MPI_Group originalGroup, rowGroup, colGroup;
    MPI_Comm rowComm, colComm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &originalGroup);

    fprintf(stderr, "[Rank %d] New call to summa function...\n", rank);

    indexX = rank % procGridX;
    indexY = (rank - indexX)/procGridX;

    fprintf(stderr, "[Rank %d] indexX = %d, indexY = %d\n", rank, indexX, indexY);

    fprintf(stderr, "[Rank %d] Ranks in row group: ", rank);
    for(p = 0; p < procGridY; ++p)
    {
        rowGroupIndex[p] = p * procGridX + indexX;
        fprintf(stderr, "%d, ", rowGroupIndex[p]);
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "[Rank %d] Ranks in col group: ", rank);
    for(p = 0; p < procGridX; ++p)
    {
        colGroupIndex[p] = indexY * procGridX + p;
        fprintf(stderr, "%d, ", colGroupIndex[p]);
    }

    fprintf(stderr, "\n");

    /* Create groups */
    if(MPI_Group_incl(originalGroup, procGridY, rowGroupIndex, &rowGroup))
    {
        fprintf(stderr, "Error creating group\n");
        MPI_Finalize();
    }
    if(MPI_Group_incl(originalGroup, procGridX, colGroupIndex, &colGroup))
    {
        fprintf(stderr, "Error creating group\n");
        MPI_Finalize();
    }

    /* Create communicators */
    if(MPI_Comm_create(MPI_COMM_WORLD, rowGroup, &rowComm))
    {
        fprintf(stderr, "Error creating group\n");
        MPI_Finalize();
    }
     
    if(MPI_Comm_create(MPI_COMM_WORLD, colGroup, &colComm))
    {
        fprintf(stderr, "Error creating group\n");
        MPI_Finalize();
    }

    fprintf(stderr, "[Rank %d] Created communicators...\n", rank);

    for(i = 0; i < k/pb; ++i)
    {
        whoseTurnRow = (int) i * pb * procGridY / k;
        whoseTurnCol = (int) i * pb * procGridX / k;

        fprintf(stderr, "[Rank %d, i = %d] Senders: %d (row) / %d (col)\n", rank, i,  whoseTurnRow, whoseTurnCol);

        /* Broadcast column to Row */
        if(indexY == whoseTurnRow)
        {
            int buffer = rank;
            fprintf(stderr, "[Rank %d, i = %d] Value of indexY before Bcast1 = %d\n", rank, i, indexY);
            MPI_Bcast(&buffer, sizeof(int), MPI_INT, whoseTurnRow, rowComm);

            fprintf(stderr, "[Rank %d, i = %d] Value of indexY after Bcast1 = %d\n", rank, i, indexY);

            fprintf(stderr, "[Rank %d, i = %d] I'm proc: %d, and sent message! (whoseTurnRow = %d)\n", rank, i, rank, whoseTurnRow);
        }
        else
        {
            int buffer;
            fprintf(stderr, "[Rank %d, i = %d] Value of indexY before Bcast2 = %d\n", rank, i, indexY);
            MPI_Bcast(&buffer, sizeof(int), MPI_INT, whoseTurnRow, rowComm);
            fprintf(stderr, "[Rank %d, i = %d] Value of indexY after Bcast2 = %d\n", rank, i, indexY);
            
            fprintf(stderr, "[Rank %d, i = %d] I'm proc: %d, and got message: %d (whoseTurnRow = %d, indexY = %d)\n", rank, i, rank, buffer, whoseTurnRow, indexY);
        }

    }

    /* MPI_Barrier(MPI_COMM_WORLD); */

}
