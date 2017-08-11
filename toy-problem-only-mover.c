// Toy PCAT problem where the number of objects (and types of objects are fixed).
// Note: If MARGIN1 size is too small, then seg fault error will occur during run time
// as there is not enough space in local block copy of MODEL.

// Note: If MAXCOUNT_BLOCK and MAXCOUNT is too small the algorithm to determine which object
// belongs to which block will fail as the buffer overflows.

// Note: Be careful about the random number generation. This may require more serious thinking. 
// Currently, I am simply using different seed for each thread.

// Note: If any of the MODEL values in the center region becomes 0, then loglike becomes nan.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>


#define GENERATE_NEW_MOCK 1 // If true, generate new mock. If false, then read in already generated image.
// Number of threads, ieration, and debug
#define NUM_THREADS 4 // Number of threads used for execution.
#define PERIODIC_MODEL_RECOMPUTE 0// If 1, at the end of each loop recompute the model from scatch to avoid accomulation of numerical error. 
#define MODEL_RECOMPUTE_PERIOD 1 // Recompute the model after 1000 iterations.
#define SERIAL_DEBUG 0 // Only to be used when NUM_THREADS 0
#define DEBUG 0// Set to 1 when debugging.
#define BLOCK_ID_DEBUG 2
#define OFFSET 1 // If 1, blocks are offset by a random amount in each iteration.
#if DEBUG
	// General strategy 
	// Debug first in serial mode, commenting out OMP directives as appropriate.
	// One thread, one block, one iteration.
	// One thread, one block, multiplie iterations.
	// One thread, multiple blocks, multiplie iterations.
	// Multiple threads, multiple blocks, multiple iterations.
	#define NLOOP 1000 // Number of times to loop before sampling
	#define NSAMPLE 2 // Numboer samples to collect
#else // If in normal mode
	#define NLOOP 100// Number of times to loop before sampling
	#define NSAMPLE 10000// Numboer samples to collect
#endif 
#define PRINT_PERF 1// If 1, print peformance after every sample.
#define RANDOM_WALK 0 // If 1, all proposed changes are automatically accepted.
#define COMPUTE_LOGLIKE 1 // If 1, loglike based on the current model is computed when collecting the sample.
#define SAVE_CHAIN 1 // If 1, save the chain for x, y, f, loglike.
#define SAVE_MODEL 1 // If 1, save the model corresponding to each sample as well as the initial.

// Define global dimensions
#define AVX_CACHE2 16
#define AVX_CACHE AVX_CACHE2
#define NPIX_div2 12
#define INNER 10
#define NPIX 25 // PSF single dimension
#define NPIX2 (NPIX*NPIX) // 25 x 25 = 625
#define MARGIN1 2 // Margin width of the block
#define MARGIN2 NPIX_div2 // Half of PSF
#define REGION 4// Core proposal region 
#define BLOCK (REGION + 2 * (MARGIN1 + MARGIN2))
#define NUM_BLOCKS_PER_DIM 2
#define NUM_BLOCKS_TOTAL (NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM)

#define MAXCOUNT_BLOCK 32 // Maximum number of objects expected to be found in a proposal region. 
#define MAXCOUNT MAXCOUNT_BLOCK// Max number of objects to be "collected" by each thread when computing block id for each object.
							// If too small, the hashing algorithm won't work as one thread will be overstepping into another's region.
#define INCREMENT 1 // Block loop increment
#define BYTES 4 // Number of byte for int and float.
#define DATA_WIDTH (NUM_BLOCKS_PER_DIM * BLOCK)
#define PADDED_DATA_WIDTH ((NUM_BLOCKS_PER_DIM+1) * BLOCK) // Extra BLOCK is for padding with haf block on each side
#define DATA_SIZE (DATA_WIDTH * DATA_WIDTH)
#define IMAGE_SIZE (PADDED_DATA_WIDTH * PADDED_DATA_WIDTH)

#define STAR_DENSITY_PER_BLOCK ((int) (0.1 * BLOCK * BLOCK))  // 102.4 x (36/1024) ~ 4
#define NUM_TRUE_STARS (STAR_DENSITY_PER_BLOCK * NUM_BLOCKS_TOTAL) // Maximum number of stars to try putting in. // Note that if the size is too big, then segfault will ocurr
#define MAX_STARS ((int) (NUM_TRUE_STARS * 0.8)) // The number of stars to use to model the image.
#define ONE_STAR_DEBUG 0 // Use only one star. NUM_BLOCKS_PER_DIM and MAX_STARS shoudl be be both 1.


// Bit number of objects within 
#define BIT_X 0
#define BIT_Y 1
#define BIT_FLUX 2

#define GAIN 1.0 // ADU to photoelectron gain factor. MODEL and DATA are given in ADU units. Flux is proportional to ADU.
#define TRUE_MIN_FLUX 250.0
#define TRUE_ALPHA 2.00
#define TRUE_BACK 179.0
#define SET_UPPER_FLUX_LIMIT 0 // If 1, the above limit is applied.
#define FLUX_UPPER_LIMIT 10000.0 // If the proposed flux values become greater than this, then set it to this value.
#define FREEZE_XY 0 // If 1, freeze the X, Y positins of the objs.
#define FREEZE_F 0 // If 1, free the flux
#define FLUX_DIFF_RATE 10.0

// Some MACRO functions
 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a < _b ? _a : _b; })   


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

int flip_coin_biased(float up_fraction){
	// Generate 0 or 1 randomly.
	float decision_point = RAND_MAX * (1-up_fraction);
	int coin;
	int x = rand();
	if (x>decision_point){
		coin = 1;
	}
	else{
		coin = 0;
	}
	return coin;
}


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
		#pragma omp parallel for simd
		for (i=0; i<size; i++){
			mat[i] = fill_val;
		}
	}

	return;
}


void print_float_vec(float* mat, int size){
	// Print out float vector
	int i;
	for (i=0; i<size-1; i++){
		printf("%.2f, ", mat[i]);
	}
	printf("%2f\n", mat[size-1]);
}



int main(int argc, char *argv[])
{	
	// Print basic parameters of the problem.
	printf("WARNING: Please be warned that the number of blocks must be greater than the number of threads.\n\n\n");
	printf("Number of sample to collect: %d\n", NSAMPLE);
	printf("Thinning rate: %d\n", NLOOP);
	printf("Total number of parallel iterations: %d, (%d K)\n", (NSAMPLE * NLOOP), (NSAMPLE * NLOOP) / (1000));
	printf("Total number of serial iterations: %.2f M\n", (NSAMPLE * NLOOP * NUM_BLOCKS_TOTAL) / (1e06));
	printf("Block width: %d\n", BLOCK);
	printf("MARGIN 1/2: %d/%d\n", MARGIN1, MARGIN2);
	printf("Proposal region width: %d\n", REGION);
	printf("Data width: %d\n", DATA_WIDTH);
	printf("Number of blocks per dim: %d\n", NUM_BLOCKS_PER_DIM);
	printf("Number of blocks processed per step: %d\n", NUM_BLOCKS_TOTAL);
	printf("MAX_STARS: %d\n", MAX_STARS);	
	printf("Obj density: %.2f per pixel\n", (float) MAX_STARS/ (float) DATA_SIZE);
	int stack_size = kmp_get_stacksize_s() / 1e06;
	printf("Stack size being used: %dMB\n", stack_size);	
	printf("Number of processors available: %d\n", omp_get_num_procs());
	printf("Number of thread used: %d\n", NUM_THREADS);


	// Set the number of threads to be used through out the program
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(NUM_THREADS); 

	srand(123); // Initializing random seed for the whole program.
	int i, j, s, k, l, m; // Initialization and loop variables. s for sample.
	int jstar = 0; // Number of stars after coalescing.
	int istar;
	int xx, yy;	
	int time_seed; // Every time parallel region is entered, reset this seed as below.


	// ----- Declare global, shared variables ----- //
	// Image DATA, MODEL, design matrix
	int size_of_A = NPIX2 * INNER;
	__attribute__((aligned(64))) float DATA[IMAGE_SIZE]; // Generate positive test data. 64 bytes aligned.
	__attribute__((aligned(64))) float MODEL[IMAGE_SIZE]; // Allocate model image. 64 bytes aligned.
	__attribute__((aligned(64))) float A[size_of_A]; // Design matrix [INNER, NPIX2]

	// Initialize to flat values.
	init_mat_float(DATA, IMAGE_SIZE, TRUE_BACK, 0); // Fill data with a flat value including the padded region
	init_mat_float(MODEL, IMAGE_SIZE, TRUE_BACK, 0); // Fill data with a flat value including the padded region
	init_mat_float(A, size_of_A, 1e-03, 0); // Fill data with small values

	// Read in the psf design matrix A
	FILE *fpA = NULL;
	// fpA = fopen("A_sdss.bin", "rb");
	fpA = fopen("A_gauss.bin", "rb");	
	fread(&A, sizeof(float), size_of_A, fpA);
	// print_float_vec(A, size_of_A); // Debug
	// printf("A[312]: %.3f\n", A[312]); // Should be 0.2971158 (based on Gaussian psf) // Debug
	fclose(fpA);


	// ------ Initialize MODEL matrix according to the random draws. ------ //
	// Object array. Each object gets AVX_CACHE space or 16 floats.
	__attribute__((aligned(64))) float OBJS[AVX_CACHE * MAX_STARS];
	// Array that tells which objects belong which arrays. See below for usage.
	__attribute__((aligned(64))) int OBJS_HASH[MAXCOUNT * NUM_THREADS * NUM_BLOCKS_TOTAL]; 
	// Block counter for each thread
	__attribute__((aligned(64))) int BLOCK_COUNT_THREAD[NUM_THREADS * NUM_BLOCKS_TOTAL]; 

	// ----- Initialize object array ----- //
	#pragma omp parallel for simd shared(OBJS)
	for (i=0; i< AVX_CACHE * MAX_STARS; i++){
		OBJS[i] = -1; // 
	}
	time_seed = (int) (time(NULL)) * rand(); // printf("Time seed %d\n", time_seed);	
    #pragma omp parallel shared(OBJS)
    {
		unsigned int p_seed = time_seed * (1+omp_get_thread_num()); // Note that this seeding is necessary
		#pragma omp for
		for (i=0; i<MAX_STARS; i++){
			int idx = i*AVX_CACHE;
			#if ONE_STAR_DEBUG
				OBJS[idx+BIT_X] = BLOCK+0.1; // x
				OBJS[idx+BIT_Y] = BLOCK+0.1; // y
				OBJS[idx+BIT_FLUX] = TRUE_MIN_FLUX * 5.;
			#else
				OBJS[idx+BIT_X] = (rand_r(&p_seed) / (float) RAND_MAX) * DATA_WIDTH + (BLOCK/2); // x
				OBJS[idx+BIT_Y] = (rand_r(&p_seed) / (float) RAND_MAX) * DATA_WIDTH + (BLOCK/2); // y
				float u = rand_r(&p_seed)/(RAND_MAX + 1.0);
				#if SET_UPPER_FLUX_LIMIT
					OBJS[idx+BIT_FLUX] = min(FLUX_UPPER_LIMIT, TRUE_MIN_FLUX * exp(-log(u) * (TRUE_ALPHA-1.0))); // flux. Impose an upper limit.							
				#else
					// OBJS[idx+BIT_FLUX] = TRUE_MIN_FLUX * exp(-log(u) * (TRUE_ALPHA-1.0)); // flux.
				#endif
	            OBJS[idx+BIT_FLUX] = TRUE_MIN_FLUX * 2; // Constant flux values for all the stars. Still an option.
			#endif
		}
	}
	// Initialize hashing variable	
	#pragma omp parallel for simd shared(OBJS_HASH)
	for (i=0; i< MAXCOUNT * NUM_THREADS * NUM_BLOCKS_TOTAL; i++){
		OBJS_HASH[i] = -1; // Can't set it to zero since 0 is a valid object number.
	}	

	// Gather operation
	__attribute__((aligned(64))) float init_x[MAX_STARS];
	__attribute__((aligned(64))) float init_y[MAX_STARS];
	__attribute__((aligned(64))) float init_f[MAX_STARS];
	for (i=0; i<MAX_STARS; i++){
		int idx = i * AVX_CACHE;
		float x = OBJS[idx + BIT_X];
		float y = OBJS[idx + BIT_Y];
		float f = OBJS[idx + BIT_FLUX];
		init_x[i] = x;
		init_y[i] = y;
		init_f[i] = f;
	}	

	// Calculating dX for each star.
	__attribute__((aligned(64))) int init_ix[MAX_STARS];
	__attribute__((aligned(64))) int init_iy[MAX_STARS];					
	#pragma omp parallel for simd
	for (k=0; k< MAX_STARS; k++){
		init_ix[k] = ceil(init_x[k]); // Padding width is already accounted for.
		init_iy[k] = ceil(init_y[k]);
	} // end of ix, iy computation
	
	// For vectorization, compute dX^T [AVX_CACHE2, MAX_STARS] and transpose to dX [MAXCOUNT, AVX_CACHE2]
	__attribute__((aligned(64))) float init_dX_T[AVX_CACHE2 * MAX_STARS];

	#pragma omp parallel for simd
	for (k=0; k < MAX_STARS; k++){
		// Calculate dx, dy						
		float px = init_x[k];
		float py = init_y[k];
		float dpx = init_ix[k]-px;
		float dpy = init_iy[k]-py;

		// flux values
		float pf = init_f[k];

		// Compute dX * f
		init_dX_T[k] = pf; //
		// dx
		init_dX_T[MAX_STARS + k] = dpx * pf; 
		// dy
		init_dX_T[MAX_STARS * 2+ k] = dpy * pf; 
		// dx*dx
		init_dX_T[MAX_STARS * 3+ k] = dpx * dpx * pf; 
		// dx*dy
		init_dX_T[MAX_STARS * 4+ k] = dpx * dpy * pf; 
		// dy*dy
		init_dX_T[MAX_STARS * 5+ k] = dpy * dpy * pf; 
		// dx*dx*dx
		init_dX_T[MAX_STARS * 6+ k] = dpx * dpx * dpx * pf; 
		// dx*dx*dy
		init_dX_T[MAX_STARS * 7+ k] = dpx * dpx * dpy * pf; 
		// dx*dy*dy
		init_dX_T[MAX_STARS * 8+ k] = dpx * dpy * dpy * pf; 
		// dy*dy*dy
		init_dX_T[MAX_STARS * 9+ k] = dpy * dpy * dpy * pf; 
	} // end of dX computation 
	#if SERIAL_DEBUG
		printf("Computed dX.\n");
	#endif
	
	// Transposing the matrices: dX^T [AVX_CACHE2, MAX_STARS] to dX [MAXCOUNT, AVX_CACHE2]
	// Combine current and proposed arrays. 
	__attribute__((aligned(64))) float init_dX[AVX_CACHE2 * MAX_STARS];
	#pragma omp parallel for collapse(2)
	for (k=0; k<MAX_STARS; k++){
		for (l=0; l<INNER; l++){
			init_dX[k*AVX_CACHE2+l] = init_dX_T[MAX_STARS*l+k];
		}
	}// end of transpose
	#if SERIAL_DEBUG
		printf("Finished transposing dX.\n");
	#endif


	// ----- Hashing ----- //
	// This steps reduces number of PSFs that need to be evaluated.					
	__attribute__((aligned(64))) int init_hash[IMAGE_SIZE];
	// Note: Objs may fall out of the inner proposal region. However
	// it shouldn't go too much out of it. So as long as MARGIN1 is 
	// 1 or 2, there should be no problem. 
	#pragma omp parallel for simd // Explicit vectorization
    for (k=0; k<IMAGE_SIZE; k++) { init_hash[k] = -1; }
	#if SERIAL_DEBUG
		printf("Initialized hashing variable.\n");
	#endif

    jstar = 0; // Number of stars after coalescing.
    for (istar = 0; istar < MAX_STARS; istar++) // This must be a serial operation.
    {
        xx = init_ix[istar];
        yy = init_iy[istar];
        int idx = xx*PADDED_DATA_WIDTH+yy;
        if (init_hash[idx] != -1) {
        	#pragma omp simd // Compiler knows how to unroll. But it doesn't seem to effective vectorization.
            for (l=0; l<INNER; l++) { init_dX[init_hash[idx]*AVX_CACHE2+l] += init_dX[istar*AVX_CACHE2+l]; }
        }
        else {
            init_hash[idx] = jstar;
            #pragma omp simd // Compiler knows how to unroll.
            for (l=0; l<INNER; l++) { init_dX[init_hash[idx]*AVX_CACHE2+l] = init_dX[istar*AVX_CACHE2+l]; }
            init_ix[jstar] = xx;
            init_iy[jstar] = yy;
            jstar++;
        }
    }
    #if SERIAL_DEBUG
		printf("Finished hashing.\n");
	#endif

	// row and col location of the star based on X, Y values.
	// Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX.
	// Calculate PSF and then add to model proposed
	for (k=0; k<jstar; k++){
		int idx_x = init_ix[k]; // Note that ix and iy are already within block position.
		int idx_y = init_iy[k];
		#if SERIAL_DEBUG
			printf("Proposed %d obj's ix, iy: %d, %d\n", k, idx_x, idx_y);
		#endif
		#pragma omp simd collapse(2)
		for (l=0; l<NPIX2; l++){
			for (m=0; m<INNER; m++){
				// AVX_CACHE_VERSION
				MODEL[(idx_x+(l/NPIX)-NPIX_div2)*PADDED_DATA_WIDTH + (idx_y+(l%NPIX)-NPIX_div2)] += init_dX[k*AVX_CACHE2+m] * A[m*NPIX2+l];
			} 
		}// End of PSF calculation for K-th star
	}
	// ----- End of initialization of the MODEL ------ //	



	// ------ Initialize DATA matrix by either reading in an old data or generating a new mock ----- //
	#if GENERATE_NEW_MOCK 
		__attribute__((aligned(64))) float OBJS_TRUE[AVX_CACHE * NUM_TRUE_STARS];
		__attribute__((aligned(64))) float mock_x[NUM_TRUE_STARS];
		__attribute__((aligned(64))) float mock_y[NUM_TRUE_STARS];
		__attribute__((aligned(64))) float mock_f[NUM_TRUE_STARS];
		// Initialize 
		#pragma omp parallel for simd shared(OBJS_TRUE)
		for (i=0; i< AVX_CACHE * NUM_TRUE_STARS; i++){
			OBJS_TRUE[i] = -1; // 
		}
		time_seed = (int) (time(NULL)) * rand(); // printf("Time seed %d\n", time_seed);		
	    #pragma omp parallel shared(OBJS_TRUE)
	    {
			unsigned int p_seed = time_seed * (1+omp_get_thread_num()); // Note that this seeding is necessary
			#pragma omp for
			for (i=0; i<NUM_TRUE_STARS; i++){
				int idx = i*AVX_CACHE;
				#if ONE_STAR_DEBUG
					OBJS_TRUE[idx+BIT_X] = BLOCK-0.5; // x
					OBJS_TRUE[idx+BIT_Y] = BLOCK-1.5; // y
					OBJS_TRUE[idx+BIT_FLUX] = TRUE_MIN_FLUX * 10000.0; // Constant flux values for all the stars. Still an option.
				#else
					OBJS_TRUE[idx+BIT_X] = (rand_r(&p_seed) / (float) RAND_MAX) * DATA_WIDTH + (BLOCK/2); // x
					OBJS_TRUE[idx+BIT_Y] = (rand_r(&p_seed) / (float) RAND_MAX) * DATA_WIDTH + (BLOCK/2); // y
					float u = rand_r(&p_seed)/(RAND_MAX + 1.0);					
					#if SET_UPPER_FLUX_LIMIT
						OBJS_TRUE[idx+BIT_FLUX] = min(FLUX_UPPER_LIMIT, TRUE_MIN_FLUX * exp(-log(u) * (TRUE_ALPHA-1.0))); // flux. Impose an upper limit.							
					#else
						OBJS_TRUE[idx+BIT_FLUX] = TRUE_MIN_FLUX * exp(-log(u) * (TRUE_ALPHA-1.0)); // flux.
					#endif					
				#endif
			}
		}
		// Saving mock x, y, f of underlying objects into files. 
		// Note that htis part cannot be parallelized.
		// Also, coalescing x, y, f into 1D array each.
		FILE *fpx_mock = NULL;
		FILE *fpy_mock = NULL;
		FILE *fpf_mock = NULL;
		fpx_mock = fopen("mock_chain_x.bin", "wb");
		fpy_mock = fopen("mock_chain_y.bin", "wb");
		fpf_mock = fopen("mock_chain_f.bin", "wb");	
		for (i=0; i<NUM_TRUE_STARS; i++){
			int idx = i * AVX_CACHE;
			float x = OBJS_TRUE[idx + BIT_X];
			float y = OBJS_TRUE[idx + BIT_Y];
			float f = OBJS_TRUE[idx + BIT_FLUX];
			fwrite(&x, sizeof(float), 1, fpx_mock);
			fwrite(&y, sizeof(float), 1, fpy_mock);
			fwrite(&f, sizeof(float), 1, fpf_mock);
			mock_x[i] = x;
			mock_y[i] = y;
			mock_f[i] = f;
		}
		fclose(fpx_mock);
		fclose(fpy_mock);
		fclose(fpf_mock);
		// Calculating dX for each star.
		__attribute__((aligned(64))) int mock_ix[NUM_TRUE_STARS];
		__attribute__((aligned(64))) int mock_iy[NUM_TRUE_STARS];					
		#pragma omp parallel for simd
		for (k=0; k< NUM_TRUE_STARS; k++){
			mock_ix[k] = ceil(mock_x[k]); // Padding width is already accounted for.
			mock_iy[k] = ceil(mock_y[k]);
		} // end of ix, iy computation
		
		// For vectorization, compute dX^T [AVX_CACHE2, NUM_TRUE_STARS] and transpose to dX [MAXCOUNT, AVX_CACHE2]
		__attribute__((aligned(64))) float mock_dX_T[AVX_CACHE2 * NUM_TRUE_STARS];

		#pragma omp parallel for simd
		for (k=0; k < NUM_TRUE_STARS; k++){
			// Calculate dx, dy						
			float px = mock_x[k];
			float py = mock_y[k];
			float dpx = mock_ix[k]-px;
			float dpy = mock_iy[k]-py;

			// flux values
			float pf = mock_f[k];

			// Compute dX * f
			mock_dX_T[k] = pf; //
			// dx
			mock_dX_T[NUM_TRUE_STARS + k] = dpx * pf; 
			// dy
			mock_dX_T[NUM_TRUE_STARS * 2+ k] = dpy * pf; 
			// dx*dx
			mock_dX_T[NUM_TRUE_STARS * 3+ k] = dpx * dpx * pf; 
			// dx*dy
			mock_dX_T[NUM_TRUE_STARS * 4+ k] = dpx * dpy * pf; 
			// dy*dy
			mock_dX_T[NUM_TRUE_STARS * 5+ k] = dpy * dpy * pf; 
			// dx*dx*dx
			mock_dX_T[NUM_TRUE_STARS * 6+ k] = dpx * dpx * dpx * pf; 
			// dx*dx*dy
			mock_dX_T[NUM_TRUE_STARS * 7+ k] = dpx * dpx * dpy * pf; 
			// dx*dy*dy
			mock_dX_T[NUM_TRUE_STARS * 8+ k] = dpx * dpy * dpy * pf; 
			// dy*dy*dy
			mock_dX_T[NUM_TRUE_STARS * 9+ k] = dpy * dpy * dpy * pf; 
		} // end of dX computation 
		#if SERIAL_DEBUG
			printf("Computed dX.\n");
		#endif
		
		// Transposing the matrices: dX^T [AVX_CACHE2, NUM_TRUE_STARS] to dX [MAXCOUNT, AVX_CACHE2]
		// Combine current and proposed arrays. 
		__attribute__((aligned(64))) float mock_dX[AVX_CACHE2 * NUM_TRUE_STARS];
		#pragma omp parallel for collapse(2)
		for (k=0; k<NUM_TRUE_STARS; k++){
			for (l=0; l<INNER; l++){
				mock_dX[k*AVX_CACHE2+l] = mock_dX_T[NUM_TRUE_STARS*l+k];
			}
		}// end of transpose
		#if SERIAL_DEBUG
			printf("Finished transposing dX.\n");
		#endif


		// ----- Hashing ----- //
		// This steps reduces number of PSFs that need to be evaluated.					
		__attribute__((aligned(64))) int mock_hash[IMAGE_SIZE];
		// Note: Objs may fall out of the inner proposal region. However
		// it shouldn't go too much out of it. So as long as MARGIN1 is 
		// 1 or 2, there should be no problem. 
		#pragma omp parallel for simd // Explicit vectorization
	    for (k=0; k<IMAGE_SIZE; k++) { mock_hash[k] = -1; }
    	#if SERIAL_DEBUG
			printf("Mock: Initialized hashing variable.\n");
		#endif

		jstar = 0;
	    for (istar = 0; istar < NUM_TRUE_STARS; istar++) // This must be a serial operation.
	    {
	        xx = mock_ix[istar];
	        yy = mock_iy[istar];
	        #if SERIAL_DEBUG
				printf("xx, yy: %d, %d\n", xx, yy);
			#endif
	        int idx = xx*PADDED_DATA_WIDTH+yy;
	        if (mock_hash[idx] != -1) {
	        	#pragma omp simd // Compiler knows how to unroll. But it doesn't seem to effective vectorization.
	            for (l=0; l<INNER; l++) { mock_dX[mock_hash[idx]*AVX_CACHE2+l] += mock_dX[istar*AVX_CACHE2+l]; }
	        }
	        else {
	            mock_hash[idx] = jstar;
	            #pragma omp simd // Compiler knows how to unroll.
	            for (l=0; l<INNER; l++) { mock_dX[mock_hash[idx]*AVX_CACHE2+l] = mock_dX[istar*AVX_CACHE2+l]; }
	            mock_ix[jstar] = xx;
	            mock_iy[jstar] = yy;
	            jstar++;
	        }
	        #if SERIAL_DEBUG
				printf("check. jstar %d\n", jstar);
			#endif
	    }
	    #if SERIAL_DEBUG
			printf("Finished hashing.\n");
		#endif

		// row and col location of the star based on X, Y values.
		// Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX.
		// Calculate PSF and then add to model proposed
		for (k=0; k<jstar; k++){
			int idx_x = mock_ix[k]; 
			int idx_y = mock_iy[k];
			#if SERIAL_DEBUG
				printf("Proposed %d obj's ix, iy: %d, %d\n", k, idx_x, idx_y);
			#endif
			#pragma omp simd collapse(2)
			for (l=0; l<NPIX2; l++){
				for (m=0; m<INNER; m++){
					// AVX_CACHE_VERSION
					DATA[(idx_x+(l/NPIX)-NPIX_div2)*PADDED_DATA_WIDTH + (idx_y+(l%NPIX)-NPIX_div2)] += mock_dX[k*AVX_CACHE2+m] * A[m*NPIX2+l];
				} 
			}// End of PSF calculation for K-th star
		}
		#if SERIAL_DEBUG
			printf("Finished updating the local copy of the MODEL.\n");
		#endif

		// Poisson generation of the model


		// Saving the data
		FILE *fp_DATA = NULL;
		fp_DATA = fopen("MOCK_DATA.bin", "wb"); // Note that the data matrix is already padded.
		fwrite(&DATA, sizeof(float), IMAGE_SIZE, fp_DATA);
		fclose(fp_DATA);		
	#else // If GENERATE_NEW_MOCK is 0
		FILE *fp_DATA = NULL;
		fp_DATA = fopen("MOCK_DATA.bin", "rb"); // Note that the data matrix is already padded.
		fread(&DATA, sizeof(float), IMAGE_SIZE, fp_DATA);
		fclose(fp_DATA);
	#endif 
	// ------ End of GENERATE NEW MOCK ------- //







	#if COMPUTE_LOGLIKE
		double dt_loglike0 = -omp_get_wtime();

		// ---- Calculate the likelihood based on the curret model ---- //
		double lnL0 = 0; // Loglike 
		// double model_sum0 = 0; // Sum of all model values w/o padding
		// double data_sum0 = 0; // Sum of all data values w/o padding
		#pragma omp parallel for simd collapse(2) private(i,j) reduction(+:lnL0)//, model_sum0, data_sum0)
		for (i=BLOCK/2; i<((BLOCK/2)+DATA_WIDTH); i++){
			for (j=BLOCK/2; j<((BLOCK/2)+DATA_WIDTH); j++){
				int idx = i*PADDED_DATA_WIDTH+j;
				// Poisson likelihood
				float tmp = MODEL[idx];
				float f = log(tmp);
				float g = f * DATA[idx];
				lnL0 += g - tmp;
				// model_sum0 += tmp;
				// data_sum0 += DATA[idx];
			}// end of column loop
		} // End of row loop

		lnL0 *= GAIN;// Multiple by the gain factor
		dt_loglike0 += omp_get_wtime();
		printf("\n");
		printf("Time for computing initial loglike (us): %.3f\n", dt_loglike0 * 1e06);
		printf("Initial lnL: %.3f\n", lnL0);
		// printf("Initial MODEL sum: %.3f\n", model_sum0);
		// printf("Initial DATA sum: %.3f\n", data_sum0);
		printf("\n");		
	#endif


	// Files for saving (NSAMPLE, MAX_STARS) of x, y, f each or (NSAMPLE) of loglike. 
    FILE *fpx = NULL;
    FILE *fpy = NULL;
    FILE *fpf = NULL;
    FILE *fplnL = NULL;

    fpx = fopen("chain_x.bin", "wb");
    fpy = fopen("chain_y.bin", "wb");
    fpf = fopen("chain_f.bin", "wb");
    fplnL = fopen("chain_lnL.bin", "wb");

    // Save the initial draws for the model
	#if SAVE_CHAIN
		double dt_savechain0 = -omp_get_wtime();
		for (i=0; i<MAX_STARS; i++){
			int idx = i * AVX_CACHE;
			float x = OBJS[idx + BIT_X];
			float y = OBJS[idx + BIT_Y];
			float f = OBJS[idx + BIT_FLUX];
			fwrite(&x, sizeof(float), 1, fpx);
			fwrite(&y, sizeof(float), 1, fpy);
			fwrite(&f, sizeof(float), 1, fpf);				
		}
		#if COMPUTE_LOGLIKE
			fwrite(&lnL0, sizeof(double), 1, fplnL);
		#endif
		dt_savechain0 += omp_get_wtime();
	#endif

	// Save the initial model
	#if SAVE_MODEL
		// Saving the data
		FILE *fp_MODEL = NULL;
		fp_MODEL = fopen("MOCK_MODELS.bin", "wb"); // Note that the data matrix is already padded.
		fwrite(&MODEL, sizeof(float), IMAGE_SIZE, fp_MODEL);
	#endif


	double start, end, dt, dt_per_iter; // For timing purpose.
	double dt_total, start_total, end_total; // Measure time taken for the whole run.	
	dt = 0; // Time accumulator

	dt_total = -omp_get_wtime();
	printf("\nSampling starts here.\n");
	for (s=0; s<NSAMPLE; s++){
		start = omp_get_wtime(); // Timing starts here 		
		for (j=0; j<NLOOP; j++){
			#if SERIAL_DEBUG 
				printf("\n------ Start of iteration %d -------\n", j);
			#endif

			// ------- Generating offsets ------ //
			// Note that image is padded with BLOCK/2 on every side.
			// The mesh size is the same as the image size. It's shifted in each iteration.
			// Positive offset corresponds to adding offset_X, offset_Y for getting the 
			// relevant DATA and MODEL elements but subtracting when computing the block id.
			#if OFFSET
				int offset_X = generate_offset(-BLOCK/4, BLOCK/4) * 2;
				int offset_Y = generate_offset(-BLOCK/4, BLOCK/4) * 2;
			#else
				int offset_X = 0; 
				int offset_Y = 0; 
			#endif 
			#if DEBUG
				printf("Offset X, Y: %d, %d\n\n", offset_X, offset_Y);
			#endif
			#if SERIAL_DEBUG
				printf("Generated offsets.\n");
			#endif

			// ------ Set the counter to zero ------ //
			#pragma omp parallel for simd default(none) shared(BLOCK_COUNT_THREAD)
			for (i=0; i < NUM_THREADS * NUM_BLOCKS_TOTAL; i++){
				BLOCK_COUNT_THREAD[i] = 0;
			}
			#if SERIAL_DEBUG
				printf("Set the counters to zero.\n");
			#endif
			// ------ Hash objects into blocks ----- //
			// For each block, allocate an array of length MAXCOUNT * numthreads (OBJS_HASH)
			// Within each MAXCOUNT chunk, save the indices found by a particular thread.
			// All the threads take one obj at a time and determine its block id 
			// given the offset. If the objects are within the proposal region,
			// then update the corresponding OBJS_HASH array element. 
			// Otherwise, do nothing.
			#pragma omp parallel default(none) shared(BLOCK_COUNT_THREAD, OBJS, OBJS_HASH, offset_X, offset_Y)
			{
				int i;
				#pragma omp for
				for (i=0; i<MAX_STARS; i++){
					int t_id = omp_get_thread_num(); // Get thread number			

					// Get x, y of the object.
					// Offset is for the mesh offset.
					int idx = i*AVX_CACHE;
					float x_float = OBJS[idx+BIT_X];
					float y_float = OBJS[idx+BIT_Y];
					int x = floor(x_float - offset_X - (BLOCK/2));
					int y = floor(y_float - offset_Y - (BLOCK/2));

					int b_idx = x / BLOCK;
					int b_idy = y / BLOCK;
					int x_in_block = x - b_idx * BLOCK;
					int y_in_block = y - b_idy * BLOCK;
					// Check if the object falls in the right region.
					// If yes, update.
					if ( (b_idx < NUM_BLOCKS_PER_DIM) & (b_idy < NUM_BLOCKS_PER_DIM) &
						(b_idx > -1) & (b_idy > -1) &
						(x_in_block >= (MARGIN1+MARGIN2)) &  (y_in_block >= (MARGIN1+MARGIN2)) &
						(x_in_block < (MARGIN1+MARGIN2+REGION)) & (y_in_block < (MARGIN1+MARGIN2+REGION)))
					{
						int b_id = (b_idx * NUM_BLOCKS_PER_DIM) + b_idy; // Compute block id of the object.
						OBJS_HASH[MAXCOUNT * (NUM_THREADS * b_id + t_id) + BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]] = i; // Deposit the object number.
						BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]+=1; // Update the counts
						#if DEBUG
							if (b_id == BLOCK_ID_DEBUG)
							{
								printf("OBJS number: %d\n", i);							
								printf("OBJS_BID: %d\n", b_id);									
								printf("Block id x,y: %d, %d\n", b_idx, b_idy);													
								// printf("tid: %d\n", t_id);
								printf("Block count: %d\n", BLOCK_COUNT_THREAD[b_id + NUM_BLOCKS_TOTAL * t_id]);
								printf("x,y before adjustment: %.1f, %.1f\n", x_float, y_float);
								printf("x,y after adjustment: %d, %d\n", x, y);
								printf("x,y in block: %d, %d\n", x_in_block, y_in_block);
								printf("\n");							
							}
						#endif
					}//	
				}
			}// End of parallel region
			#if SERIAL_DEBUG
				printf("Hashed objects into the current proposal regions.\n");

			printf("Start computing step.\n");
			#endif 
			// ----- Model evaluation, followed by acceptance or rejection. ----- //
			// Iterating through all the blocks.
			// IMPORTANT: X is the row direction and Y is the column direction.
			time_seed = (int) (time(NULL)) * rand();	
			int ibx, iby; // Block idx	
			#pragma omp parallel for collapse(2) default(none) shared(MODEL, DATA, OBJS_HASH, OBJS, time_seed, offset_X, offset_Y, A) \
				private(ibx, iby, k, l, m, jstar, istar, xx, yy)
			for (ibx=0; ibx < NUM_BLOCKS_PER_DIM; ibx+=INCREMENT){ // Row direction				
				for (iby=0; iby < NUM_BLOCKS_PER_DIM; iby+=INCREMENT){ // Column direction
					int k, l, m; // private loop variables
					int block_ID = (ibx * NUM_BLOCKS_PER_DIM) + iby; // (0, 0) corresponds to block 0, (0, 1) block 1, etc.
					int t_id = omp_get_thread_num();

					#if SERIAL_DEBUG
						printf("\nStart of Block %d computation.\n", block_ID);
					#endif

					// ----- Pick objs that lie in the proposal region ----- //
					int p_nobjs=0; // Number of objects within the proposal region of the block
					__attribute__((aligned(64))) int p_objs_idx[MAXCOUNT_BLOCK]; // The index of objects within the proposal region of the block
												// Necessary to keep in order to update after the iteration 
												// We anticipate maximum of MAXCOUNT_BLOCK number of objects in the region.
					__attribute__((aligned(64))) float p_objs[AVX_CACHE * MAXCOUNT_BLOCK]; //Array for the object information.

					// Sift through the relevant regions of OBJS_HASH to find objects that belong to the
					// proposal region of the block.
					int start_idx = block_ID * MAXCOUNT * NUM_THREADS;
					for (k=0; k < (MAXCOUNT * NUM_THREADS); k++){
						int tmp = OBJS_HASH[start_idx+k]; // See if an object is deposited.
						if (tmp>-1){ // if yes, then collect it.
							p_objs_idx[p_nobjs] = tmp;
							p_nobjs++;
							OBJS_HASH[start_idx+k] = -1; //This way, the block needs not be reset.
						}
					}

					#if SERIAL_DEBUG
						printf("Number of objects in this block: %d\n", p_nobjs);
					#endif

					if (p_nobjs > 0) // Proceed with the rest only if there are objects in the region.
					{
						// ----- Transfer objects (x, y, f) to cache ------ //
						#pragma omp simd collapse(2)						
						for (k=0; k<p_nobjs; k++){
							for (l=0; l<AVX_CACHE; l++){
								p_objs[AVX_CACHE*k+l] = OBJS[p_objs_idx[k]*AVX_CACHE+l];
							}
						}
						#if SERIAL_DEBUG
							printf("Finished reading the current values.\n");
						#endif

						// Debug: Looking at objects selected for change. Must mach objects
						#if DEBUG
							if (block_ID==BLOCK_ID_DEBUG){
								printf("\n*** After collection in the block ***\n");
								printf("BID: %d\n", block_ID);								
								printf("Number of objects in the block: %d\n", p_nobjs);
								for (k=0; k<p_nobjs; k++){
									float x_float = p_objs[AVX_CACHE*k+BIT_X];
									float y_float = p_objs[AVX_CACHE*k+BIT_Y];
									float x = x_float - (BLOCK/2) - offset_X;
									float y = y_float - (BLOCK/2) - offset_Y;			
									int x_in_block = x - ibx * BLOCK;
									int y_in_block = y - iby * BLOCK;								
									printf("OBJS number: %d\n", p_objs_idx[k]);							
									printf("Block id x,y: %d, %d\n", ibx, iby);
									printf("x,y before adjustment: %.1f, %.1f\n", x_float, y_float);
									printf("x,y after adjustment: %.1f, %.1f\n", x, y);
									printf("x,y in block: %d, %d\n", x_in_block, y_in_block);							
									printf("\n");
								}																	
							}
						#endif	
						// ----- Gather operation for the current values ----- //
						// For simd computation later.
						__attribute__((aligned(64))) float current_flux[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) float current_x[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) float current_y[MAXCOUNT_BLOCK];					
						for (k=0; k<p_nobjs; k++){
							current_x[k] = p_objs[k*AVX_CACHE+BIT_X];
							current_y[k] = p_objs[k*AVX_CACHE+BIT_Y];
							current_flux[k] = p_objs[k*AVX_CACHE+BIT_FLUX];
						}
						#if SERIAL_DEBUG
							printf("Finished gathering x, y, f values in linear arrays.\n");
						#endif

						// ------ Draw unit normal random numbers to be used. ------- //
						// 3 * p_nobjs random normal number for f, x, y.
						unsigned int p_seed = time_seed * (1+t_id); // Note that this seeding is necessary					
						__attribute__((aligned(64))) float randn[4 * MAXCOUNT_BLOCK]; // 4 since the alogrithm below generates two random numbers at a time
														// I may be generating way more than necessary.
						#pragma omp simd
						for (k=0; k < 2 * MAXCOUNT_BLOCK; k++){
							// Using 
							float u = (rand_r(&p_seed) / (float) RAND_MAX);
							float v = (rand_r(&p_seed) / (float) RAND_MAX);
							float R = sqrt(-2 * log(u));
							float cosv = cos(2 * M_PI * v);
							float sinv = sin(2 * M_PI * v);
							randn[k] = R * cosv;
							randn[k+2*MAXCOUNT_BLOCK] = R * sinv;
							// printf("%.3f, ", randn[k]); // For debugging. 
						}

						#if SERIAL_DEBUG
							printf("Finished generating normal random number numbers.\n");
						#endif

						// ----- Generate proposed values ------ //
						// Note: Proposed fluxes must be above the minimum flux.
						__attribute__((aligned(64))) float proposed_flux[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) float proposed_x[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) float proposed_y[MAXCOUNT_BLOCK];
						#pragma omp simd
						for (k=0; k<p_nobjs; k++){
							// Flux
							#if FREEZE_F
								float df = 0;
							#else
								float df = randn[(BIT_FLUX * MAXCOUNT_BLOCK) + k] * FLUX_DIFF_RATE; // (60./np.sqrt(25.))
							#endif
							float f0 = current_flux[k];
							float pf1 = f0+df;
							float pf2 = -pf1 + 2*TRUE_MIN_FLUX; // If the proposed flux is below minimum, bounce off. Why this particular form?
							proposed_flux[k] = max(pf1, pf2);

							#if SET_UPPER_FLUX_LIMIT
								proposed_flux[k] = min(proposed_flux[k], FLUX_UPPER_LIMIT); // If the proposed flux becomes too large, then set to the max val.
							#endif

							// Position
							float dpos_rms = FLUX_DIFF_RATE / max(proposed_flux[k], f0); // dpos_rms = np.float32(60./np.sqrt(25.))/(np.maximum(f0, pf))
							#if FREEZE_XY
								float dx = 1e-3;
								float dy = 0;
							#else
								float dx = randn[BIT_X * MAXCOUNT_BLOCK + k] * dpos_rms; // dpos_rms ~ 2 x 12 / 250. Essentially sub-pixel movement.
								float dy = randn[BIT_Y * MAXCOUNT_BLOCK + k] * dpos_rms;
								// printf("%.5f, ", dx);
							#endif

							proposed_x[k] = current_x[k] + dx;
							proposed_y[k] = current_y[k] + dy;
						}
						#if SERIAL_DEBUG
							printf("Finished computing proposed x, y, f values.\n");
						#endif

						// If the position is outside the image, bounce it back inside
						for (k=0; k<p_nobjs; k++){
							float px = proposed_x[k];
							float py = proposed_y[k];
							if (px < BLOCK/2){
								float tmp = (BLOCK/2)-px;
								proposed_x[k] += 2 * tmp;
							}
							else{
								if (px > (BLOCK/2 + DATA_WIDTH - 1)){
									float tmp = px - (BLOCK/2 + DATA_WIDTH - 1);									
									proposed_x[k] -= 2 * tmp;
								}
							}

							if (py < BLOCK/2){
								float tmp = (BLOCK/2)-py;
								proposed_y[k] += 2 * tmp;
							}
							else{
								if (py > (BLOCK/2 + DATA_WIDTH - 1)){
									float tmp = py - (BLOCK/2 + DATA_WIDTH - 1);									
									proposed_y[k] -= 2 * tmp;
								}
							}
							// printf("%.3f\n", (proposed_x[k]-px));
							// printf("%.3f\n", (proposed_y[k]-py));							
						}// End of x,y bouncing
						#if SERIAL_DEBUG
							printf("Finished fixing x, y at boundaries.\n");
						#endif

						// ------ compute flux distribution prior factor ------ //
						float factor = 0; // Prior factor 
						for (k=0; k< p_nobjs; k++){
							factor -= TRUE_ALPHA * log(proposed_flux[k]/current_flux[k]); // Accumulating factor											
						}
						#if SERIAL_DEBUG
							printf("Finished evaluating flux distribution prior factor.\n");
						#endif

						// ----- Compute dX matrix for current and proposed, incorporating flux ----- //
						__attribute__((aligned(64))) int current_ix[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) int proposed_ix[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) int current_iy[MAXCOUNT_BLOCK];
						__attribute__((aligned(64))) int proposed_iy[MAXCOUNT_BLOCK];					
						#pragma omp simd
						for (k=0; k< p_nobjs; k++){
							current_ix[k] = ceil(current_x[k]);
							current_iy[k] = ceil(current_y[k]);
							proposed_ix[k] = ceil(proposed_x[k]);
							proposed_iy[k] = ceil(proposed_y[k]);
						} // end of ix, iy computation
						#if SERIAL_DEBUG
							printf("Finished computing ceil of proposed and current x, y.\n");
						#endif
						
						// For vectorization, compute dX^T [AVX_CACHE2, MAXCOUNT_BLOCK] and transpose to dX [MAXCOUNT, AVX_CACHE2]
						__attribute__((aligned(64))) float current_dX_T[AVX_CACHE2 * MAXCOUNT_BLOCK]; 
						__attribute__((aligned(64))) float proposed_dX_T[AVX_CACHE2 * MAXCOUNT_BLOCK];

						#pragma omp simd
						for (k=0; k < p_nobjs; k++){
							// Calculate dx, dy						
							float px = proposed_x[k];
							float py = proposed_y[k];
							float cx = current_x[k];
							float cy = current_y[k];
							float dpx = proposed_ix[k]-px;
							float dpy = proposed_iy[k]-py;
							float dcx = current_ix[k]-cx;
							float dcy = current_iy[k]-cy;

							// flux values
							float pf = proposed_flux[k];
							float cf = -current_flux[k];

							// Compute dX * f
							current_dX_T[k] = cf; // 1
							proposed_dX_T[k] = pf; //
							// dx
							current_dX_T[MAXCOUNT_BLOCK + k] = dcx * cf; 
							proposed_dX_T[MAXCOUNT_BLOCK + k] = dpx * pf; 
							// dy
							current_dX_T[MAXCOUNT_BLOCK * 2 + k] = dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 2+ k] = dpy * pf; 
							// dx*dx
							current_dX_T[MAXCOUNT_BLOCK * 3 + k] = dcx * dcx * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 3+ k] = dpx * dpx * pf; 
							// dx*dy
							current_dX_T[MAXCOUNT_BLOCK * 4 + k] = dcx * dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 4+ k] = dpx * dpy * pf; 
							// dy*dy
							current_dX_T[MAXCOUNT_BLOCK * 5 + k] = dcy * dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 5+ k] = dpy * dpy * pf; 
							// dx*dx*dx
							current_dX_T[MAXCOUNT_BLOCK * 6 + k] = dcx * dcx * dcx * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 6+ k] = dpx * dpx * dpx * pf; 
							// dx*dx*dy
							current_dX_T[MAXCOUNT_BLOCK * 7 + k] = dcx * dcx * dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 7+ k] = dpx * dpx * dpy * pf; 
							// dx*dy*dy
							current_dX_T[MAXCOUNT_BLOCK * 8 + k] = dcx * dcy * dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 8+ k] = dpx * dpy * dpy * pf; 
							// dy*dy*dy
							current_dX_T[MAXCOUNT_BLOCK * 9 + k] = dcy * dcy * dcy * cf;
							proposed_dX_T[MAXCOUNT_BLOCK * 9+ k] = dpy * dpy * dpy * pf; 
						} // end of dX computation 
						#if SERIAL_DEBUG
							printf("Computed dX.\n");
						#endif
						
						// Transposing the matrices: dX^T [AVX_CACHE2, MAXCOUNT_BLOCK] to dX [MAXCOUNT, AVX_CACHE2]
						// Combine current and proposed arrays. 
						__attribute__((aligned(64))) float dX[AVX_CACHE2 * MAXCOUNT_BLOCK * 2];					
						for (k=0; k<p_nobjs; k++){
							for (l=0; l<INNER; l++){
								dX[k*AVX_CACHE2+l] = current_dX_T[MAXCOUNT_BLOCK*l+k];
								dX[(p_nobjs+k)*AVX_CACHE2+l] = proposed_dX_T[MAXCOUNT_BLOCK*l+k];							
							}
						}// end of transpose
						#if SERIAL_DEBUG
							printf("Finished transposing dX.\n");
						#endif

						// Combine current and proposed arrays. 
						// Note that the integer x, y positions are in block position.
						__attribute__((aligned(64))) int ix[MAXCOUNT_BLOCK * 2];
						__attribute__((aligned(64))) int iy[MAXCOUNT_BLOCK * 2];
						int idx_row = ibx * BLOCK + offset_X + (BLOCK/2); // BLOCK/2 is for the padding.
						int idx_col = iby * BLOCK + offset_Y + (BLOCK/2);
						#pragma omp simd
						for (k=0; k<p_nobjs; k++){
							ix[k] = current_ix[k] - idx_row;
							ix[p_nobjs+k] = proposed_ix[k] - idx_row;
							iy[k] = current_iy[k] - idx_col;
							iy[p_nobjs+k] = proposed_iy[k] - idx_col;
						}
						// Since the arrays were coalesced
						p_nobjs *= 2;
						#if SERIAL_DEBUG
							printf("Finished computing ix, iy.\n");
						#endif

						// #if DEBUG 
						// 	if (block_ID == BLOCK_ID_DEBUG){
						// 		for (k=0; k<p_nobjs; k++){
						// 			printf("%d, %d\n", ix[k], iy[k]);
						// 		}
						// 		printf("Printed all objs.\n\n");
						// 	}
						// #endif 



						// Step strategy: Read in the current model, calculate the loglike, 
						// directly insert PSF, calculate loglike again and comapre

						// ------ Transfer data and model for the block ----- //
						__attribute__((aligned(64))) float model_proposed[BLOCK * BLOCK];
						__attribute__((aligned(64))) float data[BLOCK * BLOCK];

						#pragma omp simd collapse(2)
						for (l=0; l<BLOCK; l++){
							for (k=0; k<BLOCK; k++){
								model_proposed[l*BLOCK + k] = MODEL[(idx_row+l)*PADDED_DATA_WIDTH + (idx_col+k)];
								data[l*BLOCK + k] = DATA[(idx_row+l)*PADDED_DATA_WIDTH + (idx_col+k)];
							}
						}
						#if SERIAL_DEBUG
							printf("Finished transferring MODEL and DATA for the block.\n");
						#endif
				
						// // ----- Compute the original likelihood based on the current model. ----- //
						double b_loglike = 0;// Original block likelihood
						double p_loglike = 0; // Proposed move's loglikehood

						// Setting up the boundary properly. Don't evaluate the likelihood where there is no data. 
						int l_min = 0;
						int l_max = BLOCK;
						int m_min = 0;
						int m_max = BLOCK;
						if (idx_row < BLOCK/2) { l_min = BLOCK/2 - idx_row;}
						if (idx_col < BLOCK/2) { m_min = BLOCK/2 - idx_col;}
						// if ( (idx_row+BLOCK) > (DATA_WIDTH+BLOCK/2-1)) { l_max = BLOCK - (idx_row+BLOCK-DATA_WIDTH-BLOCK/2+1); }
						if ( idx_row > (DATA_WIDTH-BLOCK/2-1)) { l_max = -idx_row+DATA_WIDTH+(BLOCK/2); }
						if ( idx_col > (DATA_WIDTH-BLOCK/2-1)) { m_max = -idx_col+DATA_WIDTH+(BLOCK/2); }

						// #if DEBUG
						// 	if (block_ID == BLOCK_ID_DEBUG) { 
						// 		printf("%4d\n", block_ID);
						// 		printf("%4d, %4d\n", idx_row, idx_col);
						// 		printf("%4d, %4d, %4d, %4d\n\n", l_min, l_max, m_min, m_max); 
						// 	}
						// #endif	
						#pragma omp simd collapse(2) reduction(+:b_loglike)
						for (l = l_min; l < l_max; l++){ // Compiler automatically vectorize this.															
							for (m = m_min; m < m_max; m++){
								int idx = l*BLOCK+m;
								// Poisson likelihood
								float tmp = model_proposed[idx];
								float f = log(tmp);
								float g = f * data[idx];
								b_loglike += g - tmp;
							}
						}					

						#if SERIAL_DEBUG
							printf("Finished computing current loglike.\n");
						#endif			

						// ----- Hashing ----- //
						// This steps reduces number of PSFs that need to be evaluated.					
						__attribute__((aligned(64))) int hash[BLOCK*BLOCK];
						// Note: Objs may fall out of the inner proposal region. However
						// it shouldn't go too much out of it. So as long as MARGIN1 is 
						// 1 or 2, there should be no problem. 
						#pragma omp simd // Explicit vectorization
					    for (k=0; k<BLOCK*BLOCK; k++) { hash[k] = -1; }
				    	#if SERIAL_DEBUG
							printf("Initialized hashing variable.\n");
						#endif

					    int jstar = 0; // Number of stars after coalescing.
						int istar;
						int xx, yy;
					    for (istar = 0; istar < p_nobjs; istar++) // This must be a serial operation.
					    {
					        xx = ix[istar];
					        yy = iy[istar];
					        int idx = xx*BLOCK+yy;
					        if (hash[idx] != -1) {
					        	#pragma omp simd // Compiler knows how to unroll. But it doesn't seem to effective vectorization.
					            for (l=0; l<INNER; l++) { dX[hash[idx]*AVX_CACHE2+l] += dX[istar*AVX_CACHE2+l]; }
					        }
					        else {
					            hash[idx] = jstar;
					            #pragma omp simd // Compiler knows how to unroll.
					            for (l=0; l<INNER; l++) { dX[hash[idx]*AVX_CACHE2+l] = dX[istar*AVX_CACHE2+l]; }
					            ix[jstar] = xx;
					            iy[jstar] = yy;
					            jstar++;
					        }
					    }
					    #if SERIAL_DEBUG
							printf("Finished hashing.\n");
						#endif

						// row and col location of the star based on X, Y values.
						// Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX.
						// Calculate PSF and then add to model proposed
						for (k=0; k<jstar; k++){
							int idx_x = ix[k]; // Note that ix and iy are already within block position.
							int idx_y = iy[k];
							#if SERIAL_DEBUG
								printf("Proposed %d obj's ix, iy: %d, %d\n", k, idx_row, idx_col);
							#endif
							#pragma omp simd collapse(2)
							for (l=0; l<NPIX2; l++){
								for (m=0; m<INNER; m++){
									// AVX_CACHE_VERSION
									model_proposed[(idx_x+(l/NPIX)-NPIX_div2)*BLOCK + (idx_y+(l%NPIX)-NPIX_div2)] += dX[k*AVX_CACHE2+m] * A[m*NPIX2+l];
								} 
							}// End of PSF calculation for K-th star
						}
						#if SERIAL_DEBUG
							printf("Finished updating the local copy of the MODEL.\n");
						#endif

						// // ----- Compute the new likelihood ----- //
						#pragma omp simd collapse(2) reduction(+:p_loglike)
						for (l = l_min; l < l_max; l++){ // Compiler automatically vectorize this.															
							for (m = m_min; m < m_max; m++){
								int idx = l*BLOCK+m;
								// Poisson likelihood
								float tmp = model_proposed[idx];
								float f = log(tmp);
								float g = f * data[idx];
								p_loglike += g - tmp;
							}
						}					
						#if SERIAL_DEBUG
							printf("Computed the new loglike.\n");
						#endif
						

					
						// ----- Compare to the old likelihood and if the new value is smaller then update the loglike and continue.
						// If bigger then undo the addition by subtracting what was added to the model image.
						#if RANDOM_WALK
							if (0) // Short circuit so that the proposed changes are always accpeted.
						#else
							double dlnL = (p_loglike - b_loglike) * GAIN;
							float u = (rand_r(&p_seed) / (float) RAND_MAX); // A random uniform number.
							// printf("%.3f\n", u);
							if (log(u) > ( dlnL + factor))
						#endif
						{
							// If the proposed model is rejected. Do nothing.
						}
						else{
							// Accept the proposal
						 	// Note that since padded region is never considered for loglike calculation,
							// there is no need worry about them as we update the image.
							#pragma omp simd collapse(2)
							for (l=0; l<BLOCK; l++){
								for (k=0; k<BLOCK; k++){
									 MODEL[(idx_row+l)*PADDED_DATA_WIDTH + (idx_col+k)] = model_proposed[l*BLOCK + k];
								}
							}

							// Update each obj according to the perturbation
							p_nobjs = p_nobjs/2; // Only proposal objects gets 
							#if DEBUG
								if (block_ID == BLOCK_ID_DEBUG){
									printf("\n\n**** Accepted changes ****\n");
									printf("Number of stars to be updated: %d\n", p_nobjs);
								}
							#endif
							for (k=0; k < p_nobjs; k++){
								// printf("Begun accessing obj_num\n");	Debug
								int obj_num = p_objs_idx[k];
								// printf("Accessed obj_num\n"); Debug
								int idx =  obj_num * AVX_CACHE;
								float px = proposed_x[k];
								float py = proposed_y[k];
								float pf = proposed_flux[k];
								#if DEBUG
									if (block_ID == BLOCK_ID_DEBUG){
										float x = px - (BLOCK/2) - offset_X;
										float y = py - (BLOCK/2) - offset_Y;			
										float x_in_block = x - ibx * BLOCK;
										float y_in_block = y - iby * BLOCK;								
										printf("OBJS number: %d\n", obj_num);
										printf("idx: %d\n", idx);
										printf("Thread num: %d\n", omp_get_thread_num());
										printf("Block id x,y: %d, %d\n", ibx, iby);
										printf("x,y before adjustment: %.3f, %.3f\n", px, py);
										printf("x,y after adjustment: %.3f, %.3f\n", x, y);
										printf("x,y in block: %.3f, %.3f\n", x_in_block, y_in_block);							
										printf("Proposed flux: %.3f\n", pf);
										printf("Original x,y: %.3f, %.3f\n", OBJS[idx + BIT_X], OBJS[idx + BIT_Y]);
										printf("Original f %.3f\n", OBJS[idx + BIT_FLUX]);				
										printf("\n");								
									}	
								#endif				
								OBJS[idx + BIT_X] = px;
								OBJS[idx + BIT_Y] = py;
								OBJS[idx + BIT_FLUX] = pf;
								// printf("Finished depositing.\n");
							} // Finished updating						
						}// end of proposal accept/reject}

					}// End of a step, if there are objects to perturb
					else{
						#if SERIAL_DEBUG
							printf("There were no objects so skip.\n");
						#endif
					}
				#if SERIAL_DEBUG
					printf("End of Block %d computation.\n\n", block_ID);
				#endif
				} // End of y block loop
			} // End of x block loop // End of paralell region

			// Print the x, y, f of a particular particle

			// int idx_ref = (MAX_STARS-1) * AVX_CACHE;
			// printf("%d: (x, y, f) = (%.3f,  %.3f,  %.3f)\n", j, OBJS[idx_ref + BIT_X], OBJS[idx_ref + BIT_Y], OBJS[idx_ref + BIT_FLUX]);
			// printf("\n");

			#if SERIAL_DEBUG
				printf("-------- End of iteration %d --------\n\n", j);
			#endif

			end = omp_get_wtime();
		} // End of parallel iteration loop
		dt = end - start; // Compute time for NLOOP iterations
		dt_per_iter = (dt / (double) NLOOP) * (1e06);

		#if COMPUTE_LOGLIKE
			double dt_loglike = -omp_get_wtime();

			// ---- Calculate the likelihood based on the curret model ---- //
			double lnL = 0; // Loglike 
			// double model_sum = 0; // Sum of all model values w/o padding
			// double data_sum = 0; // Sum of all data values w/o padding
			#pragma omp parallel for simd collapse(2) private(i,j) reduction(+:lnL)//, model_sum, data_sum)
			for (i=BLOCK/2; i<((BLOCK/2)+DATA_WIDTH); i++){
				for (j=BLOCK/2; j<((BLOCK/2)+DATA_WIDTH); j++){
					int idx = i*PADDED_DATA_WIDTH+j;
					// Poisson likelihood
					float tmp = MODEL[idx];
					float f = log(tmp);
					float g = f * DATA[idx];
					lnL += g - tmp;
					// model_sum += tmp;
					// data_sum += DATA[idx];
				}// end of column loop
			} // End of row loop
			lnL *= GAIN; // Multiply by the gain factor
			dt_loglike += omp_get_wtime();
		#endif

		#if SAVE_CHAIN
			double dt_savechain = -omp_get_wtime();
			for (i=0; i<MAX_STARS; i++){
				int idx = i * AVX_CACHE;
				float x = OBJS[idx + BIT_X];
				float y = OBJS[idx + BIT_Y];
				float f = OBJS[idx + BIT_FLUX];
				fwrite(&x, sizeof(float), 1, fpx);
				fwrite(&y, sizeof(float), 1, fpy);
				fwrite(&f, sizeof(float), 1, fpf);				
			}
			#if COMPUTE_LOGLIKE
				fwrite(&lnL, sizeof(double), 1, fplnL);
			#endif
			dt_savechain += omp_get_wtime();
		#endif

		// After a certain number of iterations, periodically recompute the MODEL
		// Re-cycle model initialization variables
		#if PERIODIC_MODEL_RECOMPUTE
			if (((s*NLOOP+j) % MODEL_RECOMPUTE_PERIOD) == 0){
				for (i=0; i<MAX_STARS; i++){
					int idx = i * AVX_CACHE;
					float x = OBJS[idx + BIT_X];
					float y = OBJS[idx + BIT_Y];
					float f = OBJS[idx + BIT_FLUX];
					init_x[i] = x;
					init_y[i] = y;
					init_f[i] = f;
				}	

				// Calculating dX for each star.
				#pragma omp parallel for simd
				for (k=0; k< MAX_STARS; k++){
					init_ix[k] = ceil(init_x[k]); // Padding width is already accounted for.
					init_iy[k] = ceil(init_y[k]);
				} // end of ix, iy computation
				
				// For vectorization, compute dX^T [AVX_CACHE2, MAX_STARS] and transpose to dX [MAXCOUNT, AVX_CACHE2]
				#pragma omp parallel for simd
				for (k=0; k < MAX_STARS; k++){
					// Calculate dx, dy						
					float px = init_x[k];
					float py = init_y[k];
					float dpx = init_ix[k]-px;
					float dpy = init_iy[k]-py;

					// flux values
					float pf = init_f[k];

					// Compute dX * f
					init_dX_T[k] = pf; //
					// dx
					init_dX_T[MAX_STARS + k] = dpx * pf; 
					// dy
					init_dX_T[MAX_STARS * 2+ k] = dpy * pf; 
					// dx*dx
					init_dX_T[MAX_STARS * 3+ k] = dpx * dpx * pf; 
					// dx*dy
					init_dX_T[MAX_STARS * 4+ k] = dpx * dpy * pf; 
					// dy*dy
					init_dX_T[MAX_STARS * 5+ k] = dpy * dpy * pf; 
					// dx*dx*dx
					init_dX_T[MAX_STARS * 6+ k] = dpx * dpx * dpx * pf; 
					// dx*dx*dy
					init_dX_T[MAX_STARS * 7+ k] = dpx * dpx * dpy * pf; 
					// dx*dy*dy
					init_dX_T[MAX_STARS * 8+ k] = dpx * dpy * dpy * pf; 
					// dy*dy*dy
					init_dX_T[MAX_STARS * 9+ k] = dpy * dpy * dpy * pf; 
				} // end of dX computation 
				#if SERIAL_DEBUG
					printf("Computed dX.\n");
				#endif
				
				// Transposing the matrices: dX^T [AVX_CACHE2, MAX_STARS] to dX [MAXCOUNT, AVX_CACHE2]
				// Combine current and proposed arrays. 
				#pragma omp parallel for collapse(2)
				for (k=0; k<MAX_STARS; k++){
					for (l=0; l<INNER; l++){
						init_dX[k*AVX_CACHE2+l] = init_dX_T[MAX_STARS*l+k];
					}
				}// end of transpose
				#if SERIAL_DEBUG
					printf("Finished transposing dX.\n");
				#endif


				// ----- Hashing ----- //
				// This steps reduces number of PSFs that need to be evaluated.					
				// Note: Objs may fall out of the inner proposal region. However
				// it shouldn't go too much out of it. So as long as MARGIN1 is 
				// 1 or 2, there should be no problem. 
				#pragma omp parallel for simd // Explicit vectorization
			    for (k=0; k<IMAGE_SIZE; k++) { init_hash[k] = -1; }
				#if SERIAL_DEBUG
					printf("Initialized hashing variable.\n");
				#endif

			    jstar = 0; // Number of stars after coalescing.
			    for (istar = 0; istar < MAX_STARS; istar++) // This must be a serial operation.
			    {
			        xx = init_ix[istar];
			        yy = init_iy[istar];
			        int idx = xx*PADDED_DATA_WIDTH+yy;
			        if (init_hash[idx] != -1) {
			        	#pragma omp simd // Compiler knows how to unroll. But it doesn't seem to effective vectorization.
			            for (l=0; l<INNER; l++) { init_dX[init_hash[idx]*AVX_CACHE2+l] += init_dX[istar*AVX_CACHE2+l]; }
			        }
			        else {
			            init_hash[idx] = jstar;
			            #pragma omp simd // Compiler knows how to unroll.
			            for (l=0; l<INNER; l++) { init_dX[init_hash[idx]*AVX_CACHE2+l] = init_dX[istar*AVX_CACHE2+l]; }
			            init_ix[jstar] = xx;
			            init_iy[jstar] = yy;
			            jstar++;
			        }
			    }
			    #if SERIAL_DEBUG
					printf("Finished hashing.\n");
				#endif

				// Re-set the MODEL
				init_mat_float(MODEL, IMAGE_SIZE, TRUE_BACK, 0); // Fill data with a flat value including the padded region

				// row and col location of the star based on X, Y values.
				// Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX.
				// Calculate PSF and then add to model proposed
				for (k=0; k<jstar; k++){
					int idx_x = init_ix[k]; // Note that ix and iy are already within block position.
					int idx_y = init_iy[k];
					#if SERIAL_DEBUG
						printf("Proposed %d obj's ix, iy: %d, %d\n", k, idx_x, idx_y);
					#endif
					#pragma omp simd collapse(2)
					for (l=0; l<NPIX2; l++){
						for (m=0; m<INNER; m++){
							// AVX_CACHE_VERSION
							MODEL[(idx_x+(l/NPIX)-NPIX_div2)*PADDED_DATA_WIDTH + (idx_y+(l%NPIX)-NPIX_div2)] += init_dX[k*AVX_CACHE2+m] * A[m*NPIX2+l];
						} 
					}// End of PSF calculation for K-th star				
				}//End of model update with stars
			}// End of model recompute
		#endif // End of model recompute			

		#if SAVE_MODEL // Saving the MODEL after update
			if (s == (NSAMPLE-1)) { fwrite(&MODEL, sizeof(float), IMAGE_SIZE, fp_MODEL); }
			// conditional as a safe measure to memory overflow
		#endif			

		#if PRINT_PERF
			printf("Sample %d: T_parallel (us): %.3f,  T_serial (us): %.3f\n", s, dt_per_iter, (dt_per_iter/(double) NUM_BLOCKS_TOTAL));
			printf("Time for %d iterations (s): %.3f\n", NLOOP, dt);
			#if SAVE_CHAIN
				printf("Time for saving the sample (us): %.3f\n", dt_savechain * 1e06);
			#endif			
			#if COMPUTE_LOGLIKE
				printf("Time for computing loglike (us): %.3f\n", dt_loglike * 1e06);
				printf("Current lnL: %.3f\n", lnL);
				// printf("Current Model sum: %.3f\n", model_sum);
				// printf("Current Data sum: %.3f\n", data_sum);
			#endif
			printf("\n");				
		#endif	

	} // End of sampling looop
	printf("Sampling ended.\n");

	// Total time taken
	dt_total += omp_get_wtime();
	printf("Total time taken (s): %.2f\n\n", dt_total);
	// Re-print basic parameters of the problem.
	printf("Reprint of run info.\n");
	printf("Number of sample to collect: %d\n", NSAMPLE);
	printf("Thinning rate: %d\n", NLOOP);
	printf("Total number of parallel iterations: %d (%d K)\n", (NSAMPLE * NLOOP), (NSAMPLE * NLOOP) / (1000));
	printf("Total number of serial iterations: %.2f M\n", (NSAMPLE * NLOOP * NUM_BLOCKS_TOTAL) / (1e06));
	printf("Block width: %d\n", BLOCK);
	printf("MARGIN 1/2: %d/%d\n", MARGIN1, MARGIN2);
	printf("Proposal region width: %d\n", REGION);
	printf("Data width: %d\n", DATA_WIDTH);
	printf("Number of blocks per dim: %d\n", NUM_BLOCKS_PER_DIM);
	printf("Number of blocks processed per step: %d\n", NUM_BLOCKS_TOTAL);
	printf("MAX_STARS: %d\n", MAX_STARS);	
	printf("Obj density: %.2f per pixel\n", (float) MAX_STARS/ (float) DATA_SIZE);
	printf("Stack size being used: %dMB\n", stack_size);	
	printf("Number of processors available: %d\n", omp_get_num_procs());
	printf("Number of thread used: %d\n", NUM_THREADS);	

	printf("\nExit program.\n");	
	// Close files.
	fclose(fpx);
	fclose(fpy);
	fclose(fpf);
	fclose(fplnL);
	#if SAVE_MODEL
		fclose(fp_MODEL); // Keep adding to this.
	#endif	
}

