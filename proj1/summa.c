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

#define DEBUG_INFO 0

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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
    int indexX;
    int indexY;
    int rowGroupIndex[procGridY];
    int colGroupIndex[procGridX];
    int whoseTurnRow;
    int whoseTurnCol;

    int indexRowCnt = 0;
    int indexColCnt = 0;
    int localRowCnt = 0;
    int localColCnt = 0;
    
    int sizeStripA = pb * m/procGridX;
    int sizeStripB = pb * n/procGridY;
    
    double *bufferA;
    double *bufferB;

    MPI_Group originalGroup, rowGroup, colGroup;
    MPI_Comm rowComm, colComm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &originalGroup);

    if(DEBUG_INFO) fprintf(stderr, "[Rank %d] New call to summa function...\n", rank);

    indexX = rank % procGridX;
    indexY = (rank - indexX)/procGridX;

    if(DEBUG_INFO) fprintf(stderr, "[Rank %d] indexX = %d, indexY = %d\n", rank, indexX, indexY);

    if(DEBUG_INFO) fprintf(stderr, "[Rank %d] Ranks in row group: ", rank);
    for(p = 0; p < procGridY; ++p)
    {
        rowGroupIndex[p] = p * procGridX + indexX;
        if (DEBUG_INFO) fprintf(stderr, "%d, ", rowGroupIndex[p]);
    }

    if(DEBUG_INFO) fprintf(stderr, "\n");

    if(DEBUG_INFO) fprintf(stderr, "[Rank %d] Ranks in col group: ", rank);
    for(p = 0; p < procGridX; ++p)
    {
        colGroupIndex[p] = indexY * procGridX + p;
        if(DEBUG_INFO) fprintf(stderr, "%d, ", colGroupIndex[p]);
    }

    if(DEBUG_INFO) fprintf(stderr, "\n");

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

    if(DEBUG_INFO) fprintf(stderr, "[Rank %d] Created communicators...\n", rank);

    for(i = 0; i < k/pb; ++i)
    {
        if(pb > k/procGridX || pb > k/procGridY)
        {
            int b;
            int panelRowCnt = 0;
            int panelColCnt = 0;

            int nBlocksRow = pb / (k / procGridY) + (pb % (k / procGridY) > 0)? 1 : 0;
            int nBlocksCol = pb / (k / procGridX) + (pb % (k / procGridX) > 0)? 1 : 0;

            whoseTurnRow = (int) indexRowCnt / (k / procGridY);
            whoseTurnCol = (int) indexColCnt / (k / procGridX);

            bufferA = (double *) malloc(sizeStripA * sizeof(double));
            bufferB = (double *) malloc(sizeStripB * sizeof(double));

            /* Rows */

            for(b = 0; b < nBlocksRow; ++b)
            {
                int c;
                int lengthBand = MIN(k / procGridY - localRowCnt, pb);
                double *localBufferA = (double *) malloc(lengthBand * (m / procGridX) * sizeof(double));

                if(rank == whoseTurnRow)
                {
                    /* Fill local buffer */
                    int l;

                    /* Copy A's coefficients to buffer */
                    for(l = 0; l < lengthBand * (m / procGridX); ++l)
                    {
                        localBufferA[l] = Ablock[localRowCnt * (m / procGridX) + l]; 
                    }

                    if(MPI_Bcast(localBufferA, lengthBand * (m/ procGridX), MPI_DOUBLE, whoseTurnRow, rowComm))
                    {
                        fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                        MPI_Finalize();
                    }

                }
                else
                {
                    if(MPI_Bcast(localBufferA, lengthBand * (m/ procGridX), MPI_DOUBLE, whoseTurnRow, rowComm))
                    {
                        fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                        MPI_Finalize();
                    }

                }


                /* Fill up bufferA */
                for(c = 0; c < lengthBand * (m / procGridX); ++c)
                {
                    bufferA[panelRowCnt * (k / procGridX) + c] = localBufferA[c];

                }

                indexRowCnt += lengthBand;
                panelRowCnt += lengthBand;
                
                if(lengthBand == k / procGridY)
                    localRowCnt = 0;
                else
                    localRowCnt += lengthBand; 


                free(localBufferA);

            } /* End for(b) */


            /* Columns */

            for(b = 0; b < nBlocksCol; ++b)
            {
                int c;
                int r;
                int cnt = 0;
                int lengthBand = MIN(k / procGridX - localColCnt, pb);
                double *localBufferB = (double *) malloc(lengthBand * (n / procGridY) * sizeof(double));

                if(rank == whoseTurnCol)
                {
                    /* Fill local buffer */
                    
                    /* Copy B's coefficients to buffer */
                    for(c = 0; c < n / procGridY; ++c)
                    {
                        for(r = 0; r < lengthBand; ++r)
                        {
                            localBufferB[cnt] = Bblock[c * k / procGridX  + localColCnt + r];
                            ++cnt;
                        }
                    }

                    if(MPI_Bcast(localBufferB, lengthBand * (n / procGridY), MPI_DOUBLE, whoseTurnCol, colComm))
                    {
                        fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                        MPI_Finalize();
                    }

                }
                else
                {
                    if(MPI_Bcast(localBufferB, lengthBand * (n / procGridY), MPI_DOUBLE, whoseTurnCol, colComm))
                    {
                        fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                        MPI_Finalize();
                    }

                }


                /* Fill up bufferB */
                cnt = 0;
                for(c = 0; c < n / procGridY; ++c)
                {
                    for(r = 0; r < lengthBand; ++r)
                    {
                        bufferB[c * k / procGridX  + panelColCnt + r] = localBufferB[cnt];
                        ++cnt;
                    }
                }

                indexColCnt += lengthBand;
                panelColCnt += lengthBand;

                if(lengthBand == k / procGridX)
                    localColCnt = 0;
                else
                    localColCnt += lengthBand; 


                free(localBufferB);

            } /* End for(b) */



        } 
        else
        {
            /** Case pb <= min(k/procGridX, k/procGridY) **/
        
            whoseTurnRow = (int) i * pb * procGridY / k;
            whoseTurnCol = (int) i * pb * procGridX / k;

            if(DEBUG_INFO) fprintf(stderr, "[Rank %d, i = %d] whoseTurnRow: %d, whoseTurnCol: %d\n", rank, i, whoseTurnRow, whoseTurnCol);

            bufferA = (double *) malloc(sizeStripA * sizeof(double));
            bufferB = (double *) malloc(sizeStripB * sizeof(double));

            /*fprintf(stderr, "[Rank %d, i = %d] Senders: %d (row) / %d (col)\n", rank, i,  whoseTurnRow, whoseTurnCol);*/

            /* Broadcast column to Row */
            if(indexY == whoseTurnRow)
            {
                int l;
                int currentCol = i % (k / (procGridY * pb));

                /* Copy A's coefficients to buffer */
                for(l = 0; l < sizeStripA; ++l)
                {
                    bufferA[l] = Ablock[currentCol * sizeStripA + l]; 
                }



                if(MPI_Bcast(bufferA, sizeStripA, MPI_DOUBLE, whoseTurnRow, rowComm))
                {
                    fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                    MPI_Finalize();
                }

                if(DEBUG_INFO) fprintf(stderr, "[Rank %d, i = %d] Bcast to row!\n", rank, i);

            }
            else
            {
                if(MPI_Bcast(bufferA, sizeStripA, MPI_DOUBLE, whoseTurnRow, rowComm))
                {
                    fprintf(stderr, "[Rank %d, i = %d] Error receiving!", rank, i);
                    MPI_Finalize();
                }
            } /* Row group */

            /* Broadcast row to Column Group */
            if(indexX == whoseTurnCol)
            {
                int c, r, cnt = 0;
                int currentRow = i % (k / (procGridX * pb));

                /* Copy B's coefficients to buffer */
                for(c = 0; c < n/procGridY; ++c)
                {
                    for(r = 0; r < pb; ++r)
                    {
                        bufferB[cnt] = Bblock[c * k / procGridX  + currentRow * pb + r];
                        ++cnt;
                    }
                }

                if(MPI_Bcast(bufferB, sizeStripB, MPI_DOUBLE, whoseTurnCol, colComm))
                {
                    fprintf(stderr, "[Rank %d, i = %d] Error!", rank, i);
                    MPI_Finalize();
                }

                if(DEBUG_INFO) fprintf(stderr, "[Rank %d, i = %d] Bcast to col!\n", rank, i);

                /*fprintf(stderr, "[Rank %d, i = %d] I'm proc: %d, and sent message! (whoseTurnCol = %d)\n", rank, i, rank, whoseTurnCol);*/
            }
            else
            {
                if(MPI_Bcast(bufferB, sizeStripB, MPI_DOUBLE, whoseTurnCol, colComm))
                {
                    fprintf(stderr, "[Rank %d, i = %d] Error receiving!", rank, i);
                    MPI_Finalize();
                }

                /*fprintf(stderr, "[Rank %d, i = %d] I'm proc: %d, and got message: %f (whoseTurnCol = %d, indexX = %d)\n", rank, i, rank, bufferB[0], whoseTurnCol, indexX);*/
            } /* Col group */

            if(DEBUG_INFO)
            {
                int p, q;

                fprintf(stderr, "[Rank %d, i = %d] bufferA = [", rank, i);
                for(p = 0; p < m/procGridX; ++p)
                {
                    for(q = 0; q < pb; ++q)
                        fprintf(stderr, "%f ", bufferA[q * m/procGridX + p]);
                    fprintf(stderr, ";\n");
                }
                fprintf(stderr, "]\n");

                fprintf(stderr, "[Rank %d, i = %d ]bufferB = [", rank, i);
                for(p = 0; p < pb; ++p)
                {
                    for(q = 0; q < n/procGridY; ++q)
                        fprintf(stderr, "%f ", bufferB[q * pb + p]);
                    fprintf(stderr, ";\n");
                }
                fprintf(stderr, "]\n");

            }

        } /* else (pb shorter than block) */

        /* Multiply */

        local_mm(m/procGridX, n/procGridY, pb, 1.0, bufferA, m/procGridX, bufferB, pb, 1.0, Cblock, m/procGridX);

        if(DEBUG_INFO) fprintf(stderr, "[Rank %d, i = %d] Local A: %f, local B: %f (Bblock[0]: %f). Result: %f\n", rank, i, bufferA[0], bufferB[0], Bblock[0], Cblock[0]);


        free(bufferA);
        free(bufferB);

    }

    /* MPI_Barrier(MPI_COMM_WORLD); */

}
