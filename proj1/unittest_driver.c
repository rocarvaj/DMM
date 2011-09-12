/**
 *  \file lab1.c
 *  \brief Driver file for Lab 1: MPI ping-pong microbenchmark
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MIN_BUFLEN 1 /*!< Smallest message length to try, in words */
#define MAX_BUFLEN (1 << 21) /*!< Largest message length to try, in words */
#define NUM_TRIALS 1000 /*!< Number of timing trials */

/**
 *  Performs a ping-pong exchange: YOU need to implement this
 *  function, in either C or Fortran, by editing `pingpong.c` or
 *  `pingpong.f90`, respectively.
 *
 *  \param msgbuf  Message data buffer to use during the volley.
 *  \param p_len   Read-only pointer to length of msgbuf, in words.
 *
 *  \note The length is passed as a pointer, so that the C and Fortran
 *  interfaces can be the same.
 */
extern void pingpong_ (int* msgbuf, const int* p_len);

/** Program start */
int
main (int argc, char *argv[])
{
  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME+1];
  int namelen = 0;

  FILE *fp = NULL; /* output file, only valid on rank 0 */

  int* msgbuf = NULL;
  int len = 0;

  MPI_Init (&argc, &argv);	/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* Get process id */
  MPI_Comm_size (MPI_COMM_WORLD, &np);	/* Get number of processes */
  MPI_Get_processor_name (hostname, &namelen); /* Get hostname of node */
  printf ("Hello, world! [Host:%s -- Rank %d out of %d]\n", hostname, rank, np);

  /* Ping-pong takes place only between two processes */
  assert (np == 2);

  /* Open a file for writing resutls */
  if (rank == 0) {
    fp = fopen("results.dat", "w");
    assert (fp != NULL);
  }

  /* Create a buffer large enough to hold the largest message */
  msgbuf = (int *)malloc (MAX_BUFLEN * sizeof (int));
  assert (msgbuf);

  /* Iterates over power-of-two message sizes */
  for (len = 1; len <= MAX_BUFLEN; len *= 2) {
    int trial;
    double t_start, t_elapsed;

    /* Main timing loop */
    MPI_Barrier (MPI_COMM_WORLD); /* Synchronize the nodes */
    t_start = MPI_Wtime (); /* Start timer */
    for (trial = 0; trial < NUM_TRIALS; ++trial) {
      pingpong_ (msgbuf, &len);
    } /* trial */
    t_elapsed = MPI_Wtime () - t_start; /* Stop timer */
	
    if (rank == 0) {
      /* Write the one-way transfer time data to results.dat */
      int num_bytes = len * sizeof (int);
      double t_msg = t_elapsed / NUM_TRIALS / 2;
      fprintf (fp, "%d\t%.10f\n", num_bytes, t_msg);
      fflush (fp);
    }
  } /* len */

  free (msgbuf);

  if (rank == 0) {
    fclose (fp); /* Close results.dat */
  }
  MPI_Finalize ();
  return 0;
}
