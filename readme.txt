% CSE 6230, Fall 2011: Project 1: Distributed Matrix Multiply

* This page: <http://bit.ly/gtcse6230-fa11-proj1>
* Text-only: [readme.txt](readme.txt), in pandoc markdown format

In this project, you will write a distributed matrix multiply. The project is
broken into six checkpoints, each one focusing on a different aspect of the
project. By the end, you will have written a complete, high-performing, code.

* Checkpoint A: A distributed algorithm (SUMMA)
* Checkpoint B: Leveraging multicores with OpenMP
* Checkpoint C: Using BLAS
* Checkpoint D: Cache blocking + copying; recursive and explicitly tiled implementations
* Checkpoint E: SIMD vectorization and other low-level tuning techniques
* Checkpoint F: CUBLAS and your own GPU implementations

Teams
------------------

You may work in teams of two. We encourage you to work with the same
group during the entire project, but you are allowed to change teams
("amiable divorce") from one checkpoint to the next.

What to turn in
------------------

For each checkpoint you will be asked to submit

* `code.tar.gz` - A tar of your code. This should include a Makefile,
job script, and all source code.  Your code must compile and run on
Jinx.  Include a README if your code requires any special
instructions.

* `writeup.pdf` - Each checkpoint will involve an experiment, this
file should contain a plot of the results and a brief analysis. Be
sure to include the names of each group member.

T-Square will accept sumbissions until 11:55pm on the due date. Late
assignments will not be accepted.

Checkpoint A: Implementing SUMMA (Due Sept 20)
=======================

For this checkpoint, you will be using MPI to implement the SUMMA
algorithm. The goal is to compute $C = A \times B + C$ where $A$, $B$,
and $C$ are large matrices of double precision floating-point
values. Consult course resources for a explanation of the SUMMA
algorithm.

We have provided scaffolding code to help you get started. You can
download this scaffolding here:

* <http://bit.ly/gtcse6230-fa11-proj1a-tgz>

You need to implement the `summa()` function in `summa.c`.

 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.c}

/**
 * Distributed Matrix Multiply using the SUMMA algorithm
 *  Computes C = A*B + C
 * 
 *  This function uses procGridX times procGridY processes
 *   to compute the product
 *  
 *  A is a m by k matrix, each process starts
 *	with a block of A (Ablock) 
 *  
 *  B is a k by n matrix, each process starts
 *	with a block of B (Bblock) 
 *  
 *  C is a n by m matrix, each process starts
 *	with a block of C (Cblock)
 *
 *  The resulting matrix is stored in C.
 *  A and B should not be modified during computation.
 * 
 *  Ablock, Bblock, and CBlock are stored in
 *   column-major format  
 *
 *  panelSize is the Panel Block Size
 **/
void 
summa(int m, int n, int k,
	double *Ablock, double *Bblock, double *Cblock,
	int procGridX, int procGridY, int panelSize) {

	int rank, proc_x, proc_y;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get process id */

	/** 
	 * Processes are arranged in a process grid
	 * based on a column-major numbering
	 */
	proc_x = rank % procGridX;
	proc_y = (rank - proc_x) / procGridX;

}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For simplicity, assume that the number of rows and columns in the matrices are
always multiples of the dimensions of the process grid. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.c}
assert(m % procGridX == 0);
assert(k % procGridY == 0);

assert(k % procGridX == 0);
assert(n % procGridY == 0);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Use the `local_mm()` in `local_mm.c` to compute the local matrix
multiply.


We have provided a few unit tests to verify that your implementation is
correct. We will use similar tests when grading this checkpoint.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.bash}
make unittest_summa 		# compile the unit tests
make run--unittest_summa 	# submit a job that runs the unit tests  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unit tests generate several random input matrices, multiply them with your
`summa()` implementation, then compare the output to the correct solution.  If
your implementation works, you should get the following

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.bash}
random_matrix_test m=16 n=16 k=16 px=4 py=4............passed
random_matrix_test m=32 n=32 k=32 px=4 py=4............passed
random_matrix_test m=128 n=128 k=128 px=4 py=4............passed
random_matrix_test m=128 n=32 k=128 px=4 py=4............passed
random_matrix_test m=64 n=32 k=128 px=4 py=4............passed
random_matrix_test m=128 n=128 k=128 px=8 py=2............passed
random_matrix_test m=128 n=128 k=128 px=2 py=8............passed
random_matrix_test m=128 n=128 k=128 px=16 py=1............passed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


We also provide code to time your implementation, as we did in
Hands-on Lab 1. You will want to build upon this to facilitate data
collection for the write-up.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.bash}
make time_summa			# compile an evaluation program
make run--time_summa  	# submit a job that times summa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


If you choose to complete this project in Fortran, change the `LANG` macro in the `Makefile`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LANG = FORTRAN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Writeup
------------------

The SUMMA performance model that we derived in class assumes a square process
grid and square matrices.  Generalize that model to work for non-square process
grids and non-square matrices.

Evaluate your implementation on the Jinx cluster.  Use eight nodes with eight
processes per node, for a total of 64 processes.  Experiment with different
configurations (process grids, matrix dimensions, panel block sizes).
 
* Process Grid: $1 \times 64$, $2 \times 32$, $4 \times 16$, and $8 \times 8$

* Matrix Dimensions: {m=256,n=256,k=256}, {m=1024,n=256,k=256}, {m=256,n=256,k=1024}, {m=1024,n=1024,k=1024}

* Block Sizes: 4, 16, 64, 256

Does performance improve as you increase the panel block size? Provide a plot
to demonstrate. Does the impact of the panel block size match the performance
model dervied in class? Which configuration yields the best performance
(flop/s)? 

Lastly, do your results match your generalized performance model?
