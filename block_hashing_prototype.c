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
#define NUM_BLOCKS_TOTAL (NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM)
#define MAXCOUNT 8 // Max number of objects to be "collected" by each thread when computing block id for each object.
#define INCREMENT 1 // Block loop increment
#define NITER_BURNIN 1000// Number of burn-in to perform
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
	printf("Number of processors available: %d\n", omp_get_num_procs());
	int max_num_threads = omp_get_max_threads();
	printf("Number of max threads: %d\n", max_num_threads);


	// ----- Declare global, shared variables ----- //
	// Object array. Each object gets AVX_CACHE space or 16 floats.
	__attribute__((aligned(64))) float OBJS[AVX_CACHE * MAX_STARS];
	// Array that tells which objects belong which arrays. See below for usage.
	__attribute__((aligned(64))) int OBJS_IN_BLOCK[MAXCOUNT * max_num_threads * NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM]; 
	// Block counter for each thread
	__attribute__((aligned(64))) int BLOCK_COUNT_THREAD[max_num_threads * NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM]; 


	double start, end, dt, dt_per_iter; // For timing purpose.
	// For each number of stars.
	dt = 0; // Time accumulator
	int i, j; // Initialization and NITER Loop variables	

	// Initializing random seed for the whole program.
	srand(123);
	int time_seed; // Every time parallel region is entered, reset this seed as below.

	// Start of the loop
	printf("\nLoop starts here.\n");
	for (j=0; j<NITER; j++){

		// ----- Initialize object array ----- //
		#pragma omp parallel for simd
		for (i=0; i< AVX_CACHE * MAX_STARS; i++){
			OBJS[i] = -1; // Can't set it to zero since 0 is a valid object number.
		}				

		time_seed = (int) (time(NULL)) * rand(); // printf("Time seed %d\n", time_seed);		
        #pragma omp parallel 
        {
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

		
		// ----- Main computation begins here ----- //
		start = omp_get_wtime(); // Timing starts here 

		// Initialize hashing variables	
		#pragma omp parallel for simd
		for (i=0; i<MAXCOUNT * max_num_threads * NUM_BLOCKS_TOTAL; i++){
			OBJS_IN_BLOCK[i] = -1; // Can't set it to zero since 0 is a valid object number.
		}
		
		// Set the counter to zero
		#pragma omp parallel for simd
		for (i=0; i < max_num_threads * NUM_BLOCKS_TOTAL; i++){
			BLOCK_COUNT_THREAD[i] = 0;
		}
	
		// For each block, allocate an array of length MAXCOUNT * numthreads 
		// Within each MAXCOUNT chunk, save the indices found by a particular thread.
		// Determine block id using all the threads.
		// Each thread checks out one obj at a time. 
		// Read in x, y and see if it falls within intended region.
		// If the objects are within the proposal region,
		// then update the corresponding block objs array element. 
		// Otherwise, do nothing.

		#pragma omp parallel shared(BLOCK_COUNT_THREAD)
		{
			int i;
			int t_id = omp_get_thread_num(); // Get thread number			
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
				if ((x_in_block >= (MARGIN1+MARGIN2)) &  (y_in_block >= (MARGIN1+MARGIN2)) &
					(x_in_block < (MARGIN1+MARGIN2+REGION)) & (y_in_block < (MARGIN1+MARGIN2+REGION)))
				{
					int b_id = (b_idx * NUM_BLOCKS_PER_DIM) + b_idy; // Compute block id of the object.
					OBJS_IN_BLOCK[MAXCOUNT * (max_num_threads * b_id + t_id) + BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]] = i; // Deposit the object number.
					BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]+=1; // Update the counts
					// Caculate the block index
					// OBJS_BID[i] = (b_idx * NUM_BLOCKS_PER_DIM) + b_idy;					

					// // For debugging
					// printf("OBJS x/y after cut: %d/%d\n", x, y);								
					// printf("OBJS number: %d\n", i);
					// printf("Block count: %d\n", BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]);					
					// printf("b_id x/y: %d, %d\n", b_idx, b_idy);
					// printf("x/y_in_block: %d, %d\n", x_in_block, y_in_block);				
					// printf("OBJS_BID: %d\n\n", b_id);							
				}//	

			}// End of parallel region
		}// End of BID assign parallel region

		// // Debug: How many stars were collected in total?
		// int counter = 0;
		// int i;
		// for (i=0; i<MAXCOUNT * max_num_threads * NUM_BLOCKS_TOTAL; i++){
		// 	if (OBJS_IN_BLOCK[i]>-1){
		// 		// printf("%d\n", OBJS_IN_BLOCK[i]);
		// 		counter++;
		// 	}
		// }
		// printf("\nCounter value: %d\n", counter);


		// ----- Model evaluation, followed by acceptance or rejection. ----- //
		// Iterating through all the blocks.
		// IMPORTANT: X is the row direction and Y is the column direction.
		time_seed = (int) (time(NULL)) * rand();		
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

					// Sift through the relevant regions of OBJS_IN_BLOCK to find objects that belong to the
					// proposal region of the block.
					int start_idx = block_ID * MAXCOUNT * max_num_threads;
					for (k=0; k < (MAXCOUNT * max_num_threads); k++){
						int tmp = OBJS_IN_BLOCK[start_idx+k]; // See if an object is deposited.
						if (tmp>-1){ // if yes, then collect it.
							p_objs_idx[p_nobjs] = tmp;
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

					// // Debug: Looking at objects selected for change. Must mach objects
					// // identified up-stream
					// if (block_ID == 160){
					// 	for (k=0; k<p_nobjs; k++){
					// 		float x = p_objs[AVX_CACHE*k] - BLOCK/2;
					// 		float y = p_objs[AVX_CACHE*k+1] - BLOCK/2;
					// 		printf("objs %2d: (x, y) = (%.1f, %.1f)\n", k, x, y);
					// 	}
					// }
					// // Debug: Check number of objects
					// // printf("Number of objects in the block: %d\n", p_nobjs);


					// ----- Implement perturbation ----- //
					// Draw random numbers to be used


					// Propose flux changes

					// Propose position changes

					// Compute dX matrix

					//

				// printf("End of Block %d computation.\n\n", block_ID);
				} // End of y block loop
			} // End of x block loop
			// printf("-------- End of iteration %d --------\n\n", j);
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




