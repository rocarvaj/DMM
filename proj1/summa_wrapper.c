/**
 *  \file summa_wrapper.c
 *  \brief C interface for when summa is implemented in Fortran
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdlib.h>
#include <stdio.h>

extern void summa_(int *m, int *n, int *k, double *Ablock, double *Bblock,
    double *Cblock, int *procGridX, int *procGridY, int *blockSize);

void summa(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
    int procGridX, int procGridY, int blockSize) {

  summa_(&m, &n, &k, Ablock, Bblock, Cblock, &procGridX, &procGridY, &blockSize);

}
