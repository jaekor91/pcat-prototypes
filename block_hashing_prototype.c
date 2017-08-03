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
#define NUM_BLOCKS_PER_DIM 32
#define INCREMENT 1 // Block loop increment
#define NITER_BURNIN 1000 // Number of burn-in to perform
#define NITER (1000+NITER_BURNIN) // Number of iterations
#define BYTES 4 // Number of byte for int and float.
#define STAR_DENSITY_PER_BLOCK ((int) (0.1 * BLOCK * BLOCK)) 
#define MAX_STARS (STAR_DENSITY_PER_BLOCK * (NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM)) // Maximum number of stars to try putting in. // Note that if the size is too big, then segfault will ocurr
#define DATA_WIDTH (NUM_BLOCKS_PER_DIM * BLOCK)
#define IMAGE_WIDTH ((NUM_BLOCKS_PER_DIM+1) * BLOCK) // Extra BLOCK is for padding with haf block on each side
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
	printf("Data width: %d\n", DATA_WIDTH);
	printf("Number of blocks per dim: %d\n", NUM_BLOCKS_PER_DIM);
	printf("Number of blocks processed per step: %d\n", NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM);
	printf("MAX_STARS: %d\n", MAX_STARS);	
	int stack_size = kmp_get_stacksize_s() / 1e06;
	printf("Stack size being used: %dMB\n", stack_size);	
	printf( "Number of processors available: %d\n", omp_get_num_procs( ));
	printf( "Number of threads: %d\n", omp_get_max_threads( ));	



	// ----- Declare global, shared variables ----- //
	// Object array. Each object gets AVX_CACHE space or 16 floats.
	__attribute__((aligned(64))) float OBJS[AVX_CACHE * MAX_STARS];
	// Block ID of each object
	__attribute__((aligned(64))) int OBJS_BID[MAX_STARS]; 


	// ----- Main computation begins here ----- //
	double start, end, dt, dt_per_iter; // For timing purpose.
	// For each number of stars.
	dt = 0; // Time accumulator

	// Initializing random seed for the whole program.
	srand(123);
	int time_seed; // Every time parallel region is entered, reset this seed as below.

	// Start of the loop
	for (j=0; j<NITER; j++){
		// printf("Time seed %d\n", time_seed);

		// Initialize block ids to -1.
		time_seed = (int) (time(NULL)) * rand();
		#pragma omp parallel
		{	
			#pragma omp for simd
			for (i=0; i<MAX_STARS; i++){
				OBJS_BID[i] = -1;
			}
           
			// Generate (in parallel the positions of) each object.
			int p_seed = time_seed * (1+omp_get_thread_num()); // Note that this seeding is necessary
			#pragma omp for
			for (i=0; i<MAX_STARS; i++){
				int idx = i*AVX_CACHE;
				OBJS[idx] = (rand_r(&p_seed) % (IMAGE_WIDTH-BLOCK)) + BLOCK/2;
				OBJS[idx+1] = (rand_r(&p_seed) % (IMAGE_WIDTH-BLOCK)) + BLOCK/2;
			}
		}

		// ------- Generating offsets ------ //
		// Note that image is padded with BLOCK/2 on every side.
		// The mesh size is the same as the image size. It's shifted in each iteration.
		// Positive offset corresponds to adding offset_X, offset_Y for getting the 
		// relevant DATA and MODEL elements but subtracting when computing the block id.		
		// int offset_X = 0; 
		// int offset_Y = 0; 
		
		int offset_X = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		int offset_Y = generate_offset(-BLOCK/4, BLOCK/4) * 2;
		// printf("Offset X, Y: %d, %d\n", offset_X, offset_Y);

		
		start = omp_get_wtime(); // Timing starts here 

		// Determine block id using all the threads.
		// Each thread checks out one obj at a time. 
		// Read in x, y and see if it falls within intended region.
		// If the objects are within the proposal region,
		// then update the corresponding block id array element. 
		// Otherwise, do nothing.
		time_seed = (int) (time(NULL)) * rand();				
		#pragma omp parallel
		{
			#pragma omp for
			for (i=0; i<MAX_STARS; i++){
				// Get x, y of the object.
				// Offset is for the mesh offset.
				int idx = i*AVX_CACHE;
				int x = floor(OBJS[idx] - offset_X - BLOCK/2);
				int y = floor(OBJS[idx+1] - offset_Y - BLOCK/2);

				int b_idx = x / BLOCK;
				int b_idy = y / BLOCK;
				int x_in_block = x - b_idx * BLOCK;
				int y_in_block = y - b_idy * BLOCK;
				// Check if the object falls in the right region.
				// If yes, update.
				if ((x_in_block > (MARGIN1+MARGIN2)) &  (y_in_block > (MARGIN1+MARGIN2)) &
					(x_in_block < (MARGIN1+MARGIN2+REGION)) & (y_in_block < (MARGIN1+MARGIN2+REGION))){
					// Caculate the block index
					OBJS_BID[i] = (b_idx * NUM_BLOCKS_PER_DIM) + b_idy;					
					// // For debugging
					// printf("OBJS x/y after cut: %d/%d\n", x, y);								
					// printf("b_id x/y: %d, %d\n", b_idx, b_idy);
					// printf("x/y_in_block: %d, %d\n", x_in_block, y_in_block);				
					// printf("OBJS_BID: %d\n\n", OBJS_BID[i]);							
				}					
			}// End of parallel region
		}// End of BID assign parallel region


		// ----- Model evaluation, followed by acceptance or rejection. ----- //
		// Iterating through all the blocks.
		// IMPORTANT: X is the row direction and Y is the column direction.
		#pragma omp parallel 
		{
			int ibx, iby; // Block idx
			// Recall that we only consider the center blocks. That's where the extra 1 come from
			#pragma omp for collapse(2) 
			for (iby=0; iby < NUM_BLOCKS_PER_DIM; iby+=INCREMENT){ // Column direction				
				for (ibx=0; ibx < NUM_BLOCKS_PER_DIM; ibx+=INCREMENT){ // Row direction
					int k, l, m; // private loop variables
					int block_ID = (ibx * NUM_BLOCKS_PER_DIM) + iby; // (0, 0) corresponds to block 0, (0, 1) block 1, etc.
					// printf("Block ID: %3d, (bx, by): %3d, %3d\n", block_ID, ibx, iby); // Used to check whether all the loops are properly addressed.


					// ----- Pick objs that lie in the proposal region ----- //
					int p_nobjs=0; // Number of objects within the proposal region of the block
					int p_objs_idx[AVX_CACHE2]; // The index of objects within the proposal region of the block
												// Necessary to keep in order to update after the iteration 
					float p_objs[AVX_CACHE * AVX_CACHE2]; //Array for the object information.
					int idx_helper[MAX_STARS];
					// Find out which objects belong to the block
					// Possibly most expensive step in the proposal part of the algorithm.
					for (k=0; k<MAX_STARS; k++){
						if (OBJS_BID[k] == block_ID){
							p_objs_idx[p_nobjs] = k;
							p_nobjs++;
						}
					}

					// Read in object information
					#pragma omp simd collapse(2)
					for (k=0; k<p_nobjs; k++){
						for (l=0; l<AVX_CACHE; l++){
							p_objs[AVX_CACHE*k+l] = OBJS[p_objs_idx[k]*AVX_CACHE+l];
						}
					}

					// ----- Implement perturbation ----- //
					// Draw random numbers to be used


					// Propose flux changes

					// Propose position changes

					// Compute dX matrix

					//

				} // End of y block loop
			} // End of x block loop
		}// End of OMP parallel section


		end = omp_get_wtime();
		// Update time only if burn in has passed.
		if (j>NITER_BURNIN){
			dt += (end-start);
		}// End compute time.
	} // End of NITER loop

	// Calculatin the time took.
	dt_per_iter = (dt / (NITER-NITER_BURNIN)) * (1e06); // Burn-in	
	// dt_per_iter = (dt / NITER) * (1e06); // Actual	
	printf("Elapsed time per iter (us), t_eff: %.3f, %.3f\n", dt_per_iter, (dt_per_iter/(NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM)));
}




