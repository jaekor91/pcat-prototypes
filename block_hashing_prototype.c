// This file is used to demonstrate how the block hashing---that is, how given
// x, y, blocksize of an object, the program determines to which block the object
// belongs.

// Note: Be careful about the random number generation. This may require more serious thinking. 
// Currently, I am simply using different seed for each thread.

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
#define AVX_CACHE2 16 
#define NPIX_div2 12
#define MARGIN1 0 // Margin width of the block
#define MARGIN2 NPIX_div2 // Half of PSF
#define REGION 8 // Core proposal region 
#define BLOCK (REGION + 2 * (MARGIN1 + MARGIN2))
#define NUM_BLOCKS_PER_DIM 16	// Note that if the image size is too big, then the computer may not be able to hold. 
								// +1 for the extra padding. We only consider the inner blocks.
								// Sqrt(Desired block number x 4). For example, if 256 desired, then 32. If 64 desired, 16.
#define INCREMENT 1 // Block loop increment
#define NITER_BURNIN 1000 // Number of burn-in to perform
#define NITER (100+NITER_BURNIN) // Number of iterations
#define BYTES 4 // Number of byte for int and float.
#define MAX_STARS 102 * (NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM) // Maximum number of stars to try putting in. // Note that if the size is too big, then segfault will ocurr
#define IMAGE_WIDTH (NUM_BLOCKS_PER_DIM* BLOCK)
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
	printf("MAX_STARS: %d\n", MAX_STARS);	
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
		    int p_seed = omp_get_thread_num(); // Each thread gets its own random seed.
			#pragma omp for
			for (i=0; i<MAX_STARS; i++){
				int idx = i*AVX_CACHE;
				OBJS[idx] = rand_r(&p_seed) % IMAGE_WIDTH;
				OBJS[idx+1] = rand_r(&p_seed) % IMAGE_WIDTH;
			}
		}

		// Generating offsets
		int offset_X = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		int offset_Y = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		// printf("Offset X, Y: %d, %d\n", offset_X, offset_Y);

		// Note that image is padded with BLOCK/2 on every side.
		// The mesh size is the same as the image size. It's shifted in each iteration.
		// Positive offset corresponds to adding offset_X, offset_Y for getting the 
		// relevant DATA and MODEL elements but subtracting when computing the block id.

		start = omp_get_wtime(); // Timing starts here 

		// Determine block id using all the threads.
		// Each thread checks out one obj at a time. 
		// Read in x, y and see if it falls within intended region.
		// If the objects are within the proposal region,
		// then update the corresponding block id array element. 
		// Otherwise, do nothing.
		#pragma omp parallel
		{
			#pragma omp for
			for (i=0; i<MAX_STARS; i++){
				// Get x, y of the object.
				// Offset is for the mesh offset.
				int idx = i*AVX_CACHE;
				int x = floor(OBJS[idx] - offset_X);
				int y = floor(OBJS[idx+1] - offset_Y);

				int b_idx = x / BLOCK;
				int b_idy = y / BLOCK;
				int x_in_block = x - b_idx * BLOCK;
				int y_in_block = y - b_idy * BLOCK;
				// Check if the object falls in the right region.
				// If yes, update.
				if ((x_in_block > (MARGIN1+MARGIN2)) & (x_in_block < (MARGIN1+MARGIN2+REGION)) &
					(y_in_block > (MARGIN1+MARGIN2)) & (y_in_block < (MARGIN1+MARGIN2+REGION))){
					// Caculate the block index
					OBJS_BID[i] = (b_idx * NUM_BLOCKS_PER_DIM) + b_idy;					
				}
				// For debugging
				// printf("OBJS x/y: %.1f/%.1f\n", OBJS[idx], OBJS[idx+1]);								
				// printf("b_id x/y: %d, %d\n", b_idx, b_idy);
				// printf("x/y_in_block: %d, %d\n", x_in_block, y_in_block);				
				// printf("OBJS_BID: %d\n\n", OBJS_BID[i]);								
			}// End of parallel region
		}// End of BID assign parallel region


		end = omp_get_wtime();
		// Update time only if burn in has passed.
		if (j>NITER_BURNIN){
			dt += (end-start);
		}// End compute time.



		// // Checking the answer. Need to set NITER 1 so as not to flood command prompt.
		// printf("Check answer.\n");
		// for (i=0; i<MAX_STARS; i++){
		// 	int idx = i*AVX_CACHE;	
		// 	int bid = OBJS_BID[i];
		// 	if (bid != -1){
		// 		printf("OBJS x/y: %.1f/%.1f\n", OBJS[idx], OBJS[idx+1]);
		// 		printf("OBJS_BID: %d\n\n", bid);
		// 	}
		// }// End check answer.


	} // End of NITER loop

	// Calculatin the time took.
	dt_per_iter = (dt / (NITER-NITER_BURNIN)) * (1e06); // Burn-in	
	// dt_per_iter = (dt / NITER) * (1e06); // Actual	
	printf("Elapsed time per iter (us), t_eff: %.3f, %.3f\n", dt_per_iter, (dt_per_iter/(NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM)));
}




