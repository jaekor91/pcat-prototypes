// This file is used to demonstrate how the block hashing---that is, how given
// x, y, blocksize of an object, the program determines to which block the object
// belongs.



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>

// Define global dimensions
#define AVX_CACHE 16 // Number of floats that can fit into AVX512
#define NPIX_div2 12
#define MARGIN1 0 // Margin width of the block
#define MARGIN2 NPIX_div2 // Half of PSF
#define REGION 8 // Core proposal region 
#define BLOCK (REGION + 2 * (MARGIN1 + MARGIN2))
#define NUM_BLOCKS_PER_DIM 4	// Note that if the image size is too big, then the computer may not be able to hold. 
								// +1 for the extra padding. We only consider the inner blocks.
								// Sqrt(Desired block number x 4). For example, if 256 desired, then 32. If 64 desired, 16.
#define INCREMENT 1 // Block loop increment
#define NUM_PAD_BLOCK_PER_SIDE 0
#define NUM_BLOCKS_PER_DIM_W_PAD (NUM_BLOCKS_PER_DIM+(2*NUM_PAD_BLOCK_PER_SIDE)) // Note that if the image size is too big, then the computer may not be able to hold. 
#define NITER_BURNIN 50000 // Number of burn-in to perform
#define NITER (10000+NITER_BURNIN) // Number of iterations
#define BYTES 4 // Number of byte for int and float.
#define MAX_STARS 1000 // Maximum number of stars to try putting in. // Note that if the size is too big, then segfault will ocurr
#define IMAGE_WIDTH (NUM_BLOCKS_PER_DIM_W_PAD * BLOCK)
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_WIDTH)



int generate_offset(int min, int max)
{
	// Return a random number [min, max)
	int i;
	int diff = max-min;
	if (max>0){
		i = (rand() % diff) + min;
	}
	return i;	
}



int main(int argc, char *argv[])
{	
	// ----- Setting random seed though given the concurrency this is less meaningful. ----- //
	srand(123); 

	int i, j; // Initialization and NITER Loop variables	

	// Print basic parameters of the problem.
	int size_of_DATA = IMAGE_SIZE;
	// printf("Image size: %d\n", size_of_DATA);
	printf("NITER: %d\n", (NITER-NITER_BURNIN));
	printf("Block width: %d\n", BLOCK);
	printf("MARGIN 1/2: %d/%d\n", MARGIN1, MARGIN2);
	printf("Image width: %d\n", IMAGE_WIDTH);
	printf("Number of blocks per dim: %d\n", NUM_BLOCKS_PER_DIM);
	printf("Number of blocks processed per step: %d\n", NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM);
	int stack_size = kmp_get_stacksize_s() / 1e06;
	printf("Stack size being used: %dMB\n", stack_size);	



	// ----- Declare global, shared variables ----- //
	// Object array. Each object gets AVX_CACHE space or 16 floats.
	__attribute__((aligned(64))) float OBJS[AVX_CACHE * MAX_STARS];
	// Block ID of each object
	__attribute__((aligned(64))) int OBJS_BID[MAX_STARS]; 


	// ----- Main computation begins here ----- //
	double start, end, dt, dt_per_iter; // For timing purpose.
	// For each number of stars.
	dt = 0; // Time accumulator
	// Start of the loop
	for (j=0; j<NITER; j++){

		// Initialize block ids to -1. 
		#pragma omp parallel
		{	
			#pragma omp for simd
			for (i=0; i<MAX_STARS; i++){
				OBJS_BID[i] = -1;
			}
		
		// Generate (in parallel the positions of) each object.
			#pragma omp for
			for (i=0; i<MAX_STARS; i++){
				int idx = i*AVX_CACHE;
				OBJS[idx] = rand() % IMAGE_WIDTH;
				OBJS[idx+1] = rand() % IMAGE_WIDTH;
			}
		}

		// Generating offsets
		int offset_X = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		int offset_Y = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		// printf("Offset X, Y: %d, %d\n", offset_X, offset_Y);



		// Need to think about what is happening exactly...




		start = omp_get_wtime(); // Timing starts here 
		// ----- Model evaluation, followed by acceptance or rejection. ----- //
		// Iterating through all the blocks.
		// IMPORTANT: X is the row direction and Y is the column direction.
		#pragma omp parallel
		{
			int ibx, iby; // Block idx
			// Recall that we only consider the center blocks. That's where the extra 1 come from
			#pragma omp for collapse(2)
			for (iby=NUM_PAD_BLOCK_PER_SIDE; iby< (NUM_BLOCKS_PER_DIM_W_PAD- NUM_PAD_BLOCK_PER_SIDE); iby+=INCREMENT){ // Column direction				
				for (ibx=NUM_PAD_BLOCK_PER_SIDE; ibx< (NUM_BLOCKS_PER_DIM_W_PAD- NUM_PAD_BLOCK_PER_SIDE); ibx+=INCREMENT){ // Row direction
					int k, l, m; // private loop variables
					int block_ID = (ibx * NUM_BLOCKS_PER_DIM_W_PAD) + iby; // (0, 0) corresponds to block 0, (0, 1) block 1, etc.
					// printf("Block ID: %3d, (bx, by): %3d, %3d\n", block_ID, ibx, iby); // Used to check whether all the loops are properly addressed.

					// ------ Read into cache ----- //
					// AVX_CACHE_VERSION
					// __attribute__((aligned(64))) float p_OJBS[AVX_CACHE];;
					// Start index for X, Y, F and dX, dY
					// #pragma omp simd
					// for (k=0; k<ns; k++){ // You only need ns
					// 	p_X[k] = X[k];
					// 	p_Y[k] = Y[k];
					// }

				} // End of y block loop
			} // End of x block loop
		}// End of OMP parallel section

		end = omp_get_wtime();
		// Update time only if burn in has passed.
		if (j>NITER_BURNIN){
			dt += (end-start);
		}
	} // End of NITER loop

	// Calculatin the time took.
	dt_per_iter = (dt / (NITER-NITER_BURNIN)) * (1e06); // Burn-in	
	// dt_per_iter = (dt / NITER) * (1e06); // Actual	
	printf("Elapsed time per iter (us): %.3f\n", dt_per_iter);
}




