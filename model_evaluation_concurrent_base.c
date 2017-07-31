// Model evaluation and update prototype
// General strategy: Use each thread to update a small block (48 x 48 or 32 x 32). One thread can work on 
// one region at a time. (However, we may experiment with a thread spawning addition threads
// of its own in the future.)


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
#define INNER 10 // Inner dimension of the matrix multplication.
#define AVX_CACHE 16 // Number of floats that can fit into AVX512
#define NPIX 25 // PSF single dimension
#define NPIX2 (NPIX*NPIX) // 25 x 25 = 625
#define MARGIN 4 // Margin width of the block
#define REGION 8 // Core proposal region
#define BLOCK (REGION + (2 * MARGIN))
#define NUM_BLOCKS_PER_DIM 2	// Note that if the image size is too big, then the computer may not be able to hold. 
								// +1 for the extra padding. We only consider the inner blocks.
								// Sqrt(Desired block number x 4). For example, if 256 desired, then 32. If 64 desired, 16.
#define NUM_BLOCKS_PER_DIM_W_PAD (NUM_BLOCKS_PER_DIM+2) // Note that if the image size is too big, then the computer may not be able to hold. 
#define NITER_BURNIN 1000 // Number of burn-in to perform
#define NITER (1000+NITER_BURNIN) // Number of iterations
#define LARGE_LOGLIKE 100 // Large loglike value filler.
#define BYTES 4 // Number of byte for int and float.
#define MAX_STARS 1000 // Maximum number of stars to try putting in. // Note that if the size is too big, then segfault will ocurr
#define IMAGE_WIDTH (NUM_BLOCKS_PER_DIM_W_PAD * BLOCK)
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_WIDTH)
#define BLOCK_LOGLIKE (BLOCK + (2 * MARGIN)+ REGION) // BLOCK_LOGLIKE is twice larger in size in each dimension.
#define HASHING REGION // HASHING = 0 if we want to explore performance gain with the technique. Otherwise set to MARGIN.


void init_mat_float(float* mat, int size, float fill_val, int rand_fill)
{
	// If random 1, then fill the matrix with random float values.
	// If random 0, then fill the matrix with fill values.
	int i;

	if (rand_fill)
	{
		for (i=0; i<size; i++){
			mat[i] = rand();
		}
	}
	else{
		for (i=0; i<size; i++){
			mat[i] = fill_val;
		}
	}

	return;
}

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


void init_mat_int(int* mat, int size, int min, int max)
{
	// Fill the integer matrix with random number in [min, max)
	int i;
	int diff = max-min;
	if (max>0){
		for (i=0; i<size; i++){
			mat[i] = (rand() % diff) + min;
		}
	}
	if (max==0){
		for (i=0; i<size; i++){
			mat[i] = 0;
		}
	}
	return;
}

void generate_parity(int* parity_vec, int vec_size){
	// Given the vector parity_vec of size NITER,
	// draw random 0 or 1 for NITER times and save.
	int i;
	for (i=0; i<vec_size; i++){
		parity_vec[i] = flip_coin();
	}
	return;
}

int flip_coin(){
	// Generate 0 or 1 randomly.
	int half_point = RAND_MAX / 2;
	int coin;
	int x = rand();
	if (x>half_point){
		coin = 1;
	}
	else{
		coin = 0;
	}
	return coin;
}

void print_mat_int(int* mat, int size){
	int i;
	for (i=0; i<size-1; i++){
		printf("%d, ", mat[i]);
	}
	printf("%d\n", mat[size-1]);
}



int main(int argc, char *argv[])
{	
	// ----- Setting random seed though given the concurrency this is less meaningful. ----- //
	srand(123); 

	// ----- Declare global, shared variables ----- //
	// Number of stars to perturb/add.
	int size_of_nstar = 9;
	int nstar[9] = {0, 1, 2, 3, 4, 8, 16, 160, MAX_STARS};

	// * Pre-allocate image DATA, MODEL, design matrix, num_stars, and loglike
	int size_of_DATA = IMAGE_SIZE;
	int size_of_A = NPIX2 * INNER;
	// int size_of_LOGLIKE = NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD; // Padding so as to not worry about margin. Only working on the inner region.
	// AVX_CACHE_VERSION
	int size_of_LOGLIKE = NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD * AVX_CACHE; // Padding so that no data contention occurs.	
	__attribute__((aligned(64))) float DATA[size_of_DATA]; // Generate positive test data. 64 bytes aligned.
	__attribute__((aligned(64))) float MODEL[size_of_DATA]; // Allocate model image. 64 bytes aligned.
	__attribute__((aligned(64))) float WEIGHT[size_of_DATA]; // Inverse variance map
	__attribute__((aligned(64))) float A[size_of_A]; // Design matrix
	__attribute__((aligned(64))) float LOGLIKE[size_of_LOGLIKE]; // loglike without padding. Turns out this is better.
	// printf("Image size: %d\n", size_of_DATA);
	printf("NITER: %d\n", (NITER-NITER_BURNIN));
	printf("Image width: %d\n", IMAGE_WIDTH);
	printf("Block width: %d\n", BLOCK);
	printf("Loglike bock width: %d\n", BLOCK_LOGLIKE); 		
	printf("Number of blocks per dim: %d\n", NUM_BLOCKS_PER_DIM);
	printf("Number of blocks processed per step: %d\n", NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM / 4);

	// ----- Initialize global, shared variables ----- //
	init_mat_float(DATA, size_of_DATA, 0.0, 1); // Fill data with random values
	init_mat_float(MODEL, size_of_DATA, 0.0, 0); // Fill data with zero values
	init_mat_float(MODEL, size_of_DATA, 0.0, 1); // Fill data with random values	
	init_mat_float(A, size_of_A, 0.0, 1); // Fill data with random values
	init_mat_float(LOGLIKE, size_of_LOGLIKE, LARGE_LOGLIKE, 0); // Fill data with random values


	// ----- Pre-allocate memory for within-loop shared variables ----- //
	// ----- Pre-draw parity variables ----- //
	int parity_X[NITER];
	int parity_Y[NITER];
	generate_parity(parity_X, NITER);
	generate_parity(parity_Y, NITER);
	// print_mat_int(parity_X, NITER);
	
	// ----- Main computation begins here ----- //
	int i, j; // Loop variables
	int ibx, iby; // Block idx
	int ns; // Number of stars
	int par_X, par_Y; // current parrity
	double start, end, dt, dt_per_iter;
	// For each number of stars.
	for (i=0; i<size_of_nstar; i++){
		ns = nstar[i]; // current star number
		dt = 0; // Time accumulator
		// Start of the loop
		for (j=0; j<NITER; j++){
			// Pick parity
			par_X = parity_X[j];
			par_Y = parity_Y[j];
			// printf("Parity: (%d, %d)\n", par_X, par_Y);

			// * Pre-allocate space for X, Y, dX, parity_X, parity_Y.
			// int size_of_XYF = (NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD) * ns; // 
			// int size_of_dX = (NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD) * ns * INNER; // Each block gets ns * INNER. Note, however, only the first 10 elements matter.
			// AVX_CACHE_VERSION
			int ns_AVX_CACHE;			
			if ((ns % AVX_CACHE) == 0){
				ns_AVX_CACHE = ns;
			}
			else{
				ns_AVX_CACHE = ((ns/AVX_CACHE)+1) * AVX_CACHE;
			}
			int size_of_XYF = (NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD) * ns_AVX_CACHE; // 
			int size_of_dX = (NUM_BLOCKS_PER_DIM_W_PAD * NUM_BLOCKS_PER_DIM_W_PAD) * ns * AVX_CACHE; // Each block gets ns * INNER. Note, however, only the first 10 elements matter.

			__attribute__((aligned(64))) int X[size_of_XYF]; // Assume 4 bytes integer
			__attribute__((aligned(64))) int Y[size_of_XYF];
			// __attribute__((aligned(64))) float F[size_of_XYF]; // The flux variable is not used 
			__attribute__((aligned(64))) float dX[size_of_dX];


			// Randomly generate X, Y, dX, dY
			init_mat_float(dX, size_of_dX, 0.0, 1); 
			// init_mat_float(F, size_of_XYF, 0.0, 1); 
			init_mat_int(X, size_of_XYF, 0, HASHING); 
			init_mat_int(Y, size_of_XYF, 0, HASHING);

			// For experimentign with offsets.
			// int offset_X = generate_offset(0, REGION);
			// int offset_Y = generate_offset(0, REGION);

			// print_mat_int(X, size_of_XYF); // Used to check the values of the matrix X, Y.

			start = omp_get_wtime(); // Timing starts here 
			// ----- Model evaluation, followed by acceptance or rejection. ----- //
			// Iterating through all the blocks.
			// IMPORTANT: X is the row direction and Y is the column direction.
			#pragma omp parallel
			{
				// Recall that we only consider the center blocks. That's where the extra 1 come from
				#pragma omp for collapse(2)
				for (iby=1+par_Y; iby< (NUM_BLOCKS_PER_DIM_W_PAD-1); iby+=2){ // Column direction				
					for (ibx=1+par_X; ibx< (NUM_BLOCKS_PER_DIM_W_PAD-1); ibx+=2){ // Row direction
						int k, l, m; // private loop variables
						int block_ID = (ibx * NUM_BLOCKS_PER_DIM_W_PAD) + iby; // (0, 0) corresponds to block 0, (0, 1) block 1, etc.
						// printf("Block ID: %3d, (bx, by): %3d, %3d\n", block_ID, ibx, iby); // Used to check whether all the loops are properly addressed.

						// Read into cache
						// Manual pre-fetching might be bad...
						// __attribute__((aligned(64))) float p_dX[INNER * ns];
						// __attribute__((aligned(64))) int p_X[ns]; // Really you only need ns
						// __attribute__((aligned(64))) int p_Y[ns];						
						// __attribute__((aligned(64))) int hash[REGION*REGION]; // Hashing variable
						// AVX_CACHE_VERSION
						__attribute__((aligned(64))) float p_dX[AVX_CACHE * ns];
						__attribute__((aligned(64))) int p_X[ns_AVX_CACHE]; // Really you only need ns
						__attribute__((aligned(64))) int p_Y[ns_AVX_CACHE];						
						__attribute__((aligned(64))) int hash[REGION*REGION]; // Hashing variable

						// Intermediate add point						
						__attribute__((aligned(64))) float model_diff[BLOCK_LOGLIKE*BLOCK_LOGLIKE]; 


						// Storage for PSF // With the intermediate storage, there is no longer a need for PSF storage.
						// __attribute__((aligned(64))) float PSF[ns * NPIX2];
						// AVX_CACHE_VERSION: Since PSF is called from cache ignore this.


						// Start index for X, Y, F and dX, dY
						// int idx_XYF = block_ID * ns;
						// int idx_dX = block_ID * ns * INNER;
						// AVX_CACHE_VERSION
						int idx_XYF = block_ID * ns_AVX_CACHE;
						int idx_dX = block_ID * ns * AVX_CACHE;						

						#pragma omp simd
						for (k=0; k<ns; k++){ // You only need ns
							p_X[k] = X[idx_XYF+k];
							p_Y[k] = Y[idx_XYF+k];
						}
						#pragma omp simd 
						for (k=0; k<ns; k++){
							for (m=0; m<INNER; m++){
								p_dX[INNER*k+m] = dX[idx_dX+k*INNER+m];
							}
						}


						// This actually seems to slow down the program.
						// __attribute__((aligned(64))) float p_A[size_of_A]; // Private copy of A. 
						// #pragma omp simd
						// for (k=0; k<size_of_A; k++){
						//      p_A[k] = A[k];
						// }


						// Update the model by inserting ns stars into model diff and then adding to MODEL.
						// Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX.

						// Hashing. This steps reduces number of PSFs that need to be evaluated.
					    for (k=0; k<REGION*REGION; k++) { hash[k] = -1; }
					    int jstar = 0; // Number of stars after coalescing.
						int istar;
						int xx, yy;
					    for (istar = 0; istar < ns; istar++)
					    {
					        xx = p_X[istar];
					        yy = p_Y[istar];
					        int idx = yy*REGION+xx;
					        if (hash[idx] != -1) {
					        	#pragma omp simd
					            for (l=0; l<INNER; l++) { p_dX[hash[idx]*INNER+l] += p_dX[istar*INNER+l]; }
					        }
					        else {
					            hash[idx] = jstar;
					            #pragma omp simd
					            for (l=0; l<INNER; l++) { p_dX[hash[idx]*INNER+l] = p_dX[istar*INNER+l]; }
					            p_X[jstar] = p_X[istar];
					            p_Y[jstar] = p_Y[istar];
					            jstar++;
					        }
					    }

						// row and col location of the star based on X, Y values.
						int idx_row; 
						int idx_col;					    

						// Calculate PSF and then add to model_diff
						for (k=0; k<jstar; k++){
							idx_row = (2*MARGIN) + (REGION/2) + p_X[k];
							idx_col = (2*MARGIN) + (REGION/2) + p_Y[k];							
							// Compute PSF and store
							#pragma omp simd
							for (l=0; l<NPIX2; l++){
								#pragma omp simd
								for (m=0; m<INNER; m++){
									model_diff[(idx_row+(l/NPIX))*BLOCK + (idx_col+(l%NPIX))] += p_dX[k*INNER+m] * A[m*NPIX2+l];
								} 
							}// End of PSF calculation for K-th star
						}

						#pragma omp simd
						for (l=0; l<BLOCK_LOGLIKE * BLOCK_LOGLIKE; l++){
							model_diff[l] = 0;
						}
						// Add to the model image
						idx_row = ibx * BLOCK - 2*MARGIN - (REGION/2);
						idx_col = iby * BLOCK - 2*MARGIN - (REGION/2);
						#pragma omp simd
						for (l=0; l<BLOCK_LOGLIKE; l++){						
							for (k=0; k<BLOCK_LOGLIKE; k++){
								MODEL[(idx_row+l)*IMAGE_WIDTH + (idx_col+k)] += model_diff[(idx_row+l)*BLOCK_LOGLIKE + (idx_col+k)];
							}
						}


						// ----- Compute the new likelihood based on the updated model. ----- //
						// Sum regions in BLOCK_LOGLIKE
						// float b_loglike = LOGLIKE[block_ID];// Loglikelihood corresponding to the block.
						// AVX_CACHE_VERSION
						float b_loglike = LOGLIKE[block_ID * AVX_CACHE];// Loglikelihood corresponding to the block.


						float p_loglike = 0; // Proposed move's loglikehood

						//simd reduction
						idx_row = ibx * BLOCK - MARGIN - (REGION/2);
						idx_col = iby * BLOCK - MARGIN - (REGION/2);

						__attribute__((aligned(64))) float loglike_temp[AVX_CACHE];
						#pragma omp simd
						for (k=0; k<AVX_CACHE; k++){
							loglike_temp[k] = 0;
						}

						// 
						int idx_start, idx;
						#pragma omp simd
						for (l=0; l < BLOCK_LOGLIKE; l++){ // 32
							for (m=0; m < BLOCK_LOGLIKE/AVX_CACHE; m++){
								idx_start = (idx_row+l)*IMAGE_WIDTH + (idx_col+m*AVX_CACHE);
								#pragma omp simd								
								for (k=0; k<AVX_CACHE; k++){
									idx = idx_start+k;
									loglike_temp[k] += WEIGHT[idx]*(MODEL[idx]-DATA[idx])*(MODEL[idx]-DATA[idx]);
								}
							}
						}
						// Sum AVX_CACHE number
						for (k=0; k<AVX_CACHE; k++){
							p_loglike += loglike_temp[k];
						}

						// ----- Compare to the old likelihood and if the new value is smaller then update the loglike and continue.
						// If bigger then undo the addition by subtracting what was added to the model image.						
						if (flip_coin()){ // Currently, use flip coin.
							#pragma omp simd
							for (l=0; l<BLOCK_LOGLIKE; l++){						
								for (k=0; k<BLOCK_LOGLIKE; k++){
									MODEL[(idx_row+l)*IMAGE_WIDTH + (idx_col+k)] -= model_diff[(idx_row+l)*BLOCK_LOGLIKE + (idx_col+k)];
								}
							}
						}
						else{
							// Accept the proposal
							LOGLIKE[block_ID] = p_loglike;// Loglikelihood corresponding to the block.							
						}
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
	double dt_eff = dt_per_iter/ (NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM / 4);
	// dt_per_iter = (dt / NITER) * (1e06); // Actual	
	printf("ns =%5d, elapsed time per iter (us): %.3f, t_eff (us): %.3f\n", ns, dt_per_iter, dt_eff);
	} // End of nstar loop
}
