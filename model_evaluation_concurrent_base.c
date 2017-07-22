// Model evaluation and update prototype
// General strategy: Use each thread to update a block of 32 x 32. One thread can work on 
// one region at a time. (However, we may experiment with a thread spawning addition threads
// of its own.)
// 
// Program outline:
// - Define global dimensions: 
// 		* AVX_CACHE = 16: Number of single floats that fit in the cacheline
//		* NPIX: Number of pixels in each dimension of the PSF.
//		* NPIX2 = NPIX^2 
//		* INNER = 10: Inner dimension of the matrix multplication.
//		* BLOCK = 32: Chosen to be 2 times AVX_CACHE. 
// 		* MARGIN = 8: Margin
// 		* REGION_WIDTH = 16: Redudant but used for clarity.
//		* NUM_BLOCKS_PER_DIM: Sqrt(Desired block number x 4). For example, if 256 desired, then 32. If 64 desired, 16.
// 		Note: NPIX and NPIX2 may not be used since the design matrix will be chosen such that
//		it adds zero pixel value but cache line optimized. This choice may not matter but I 
// 		put it in for now.
// - Define global, shared variables:
//		* Image DATA [NUM_BLOCKS_PER_DIM x BLOCK, NUM_BLOCKS_PER_DIM x BLOCK]: Generate positive test data. 64 bytes aligned.
//		* Image MODEL [NUM_BLOCKS_PER_DIM x BLOCK, NUM_BLOCKS_PER_DIM x BLOCK]: Allocate model image. 64 bytes aligned.
//		* Design matrix A [INNER, (2 x AVX_CACHE)^2]
//		* num_stars: Number of stars for each region.
// 		* loglike: [NUM_BLOCKS_PER_DIM^2, PAD]: Stores loglikehood for each block.
// - Loops:
//		1) For num_stars {1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 5000}
//			2) For each iteration of loop. Randomly generate.
//				* X, Y [NUM_BLOCKS_PER_DIM^2, num_stars]: The (x,y) position of stars. Defined within each block. 
// 					The upper-left corner has (0,0) coordintate.
//				* dX, dY [NUM_BLOCKS_PER_DIM^2 x num_stars, INNER]: Floats [0, 1].
//				* f: Floats [0, 256].
//				* parity_x, parity_y: 0 or 1. Determine which sub-blocks to work on.
// 				// For later: * PSFs [num_stars * NUM_BLOCKS_PER_DIM^2, (2 x AVX_CACHE)^2]: Storage for the PSFs 
// 				* start: Begin timing
//				3) For each block chosen by the xy-parities (let each thread work on)
// 					* Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX. 
// 					Insert without intermitten storage. Make sure to multiply by the new flux. 
// 					(Note that killing a star is like giving a negative flux.)
// 					* Compute the new likelihood based on the updated model. Based on 48 x 48 region, region larger than the block.
// 					* Compare to the old likelihood and if the difference is negative then update the loglike and continue.
// 					If positive then undo the addition by subtracting what was added to the model image.
//				* end: End timing.
// 				* Compute dt and add to dT
//			Report the speed in terms of wall time.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>


#define NPIX 25
#define NPIX2 (NPIX*NPIX)
#define INNER 10
#define AVX_CACHE 16 // Number of floats that can fit into AVX512
#define BLOCK 32 
#define MARGIN 8
#define REGION 16
#define NUM_BLOCKS_PER_DIM 16 // Note that if the image size is too big, then the computer may not be able to hold. 
#define NITER 1000 // Number of iterations
#define LARGE_LOGLIKE 1000 // Large loglike value filler.


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



int main(int argc, char *argv[])
{	
	// ----- Setting random seed though given the concurrency this is less meaningful. ----- //
	srand(123); 

	// ----- Declare global, shared variables ----- //
	// Loop variables
	// int i, j, k, l;

	// Number of stars
	int nstar[11] = {1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 5000};

 	// Number of iteration to perform.
 	int niter = 1000;

	// * Pre-allocate image DATA, MODEL, design matrix, num_stars, and loglike
	int size_of_DATA = (NUM_BLOCKS_PER_DIM * BLOCK) * (NUM_BLOCKS_PER_DIM * BLOCK);

	int size_of_A = (4 * AVX_CACHE * AVX_CACHE) * INNER;
	int size_of_LOGLIKE = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM * AVX_CACHE; // Padding to avoid cache coherence issue.
	__attribute__((aligned(64))) float DATA[size_of_DATA];
	__attribute__((aligned(64))) float MODEL[size_of_DATA];
	__attribute__((aligned(64))) float A[size_of_A];
	__attribute__((aligned(64))) float LOGLIKE[size_of_LOGLIKE]; 

	printf("Image size %d\n", size_of_DATA);


	// ----- Initialize global, shared variables ----- //
	init_mat_float(DATA, size_of_DATA, 0.0, 1); // Fill data with random values
	init_mat_float(MODEL, size_of_DATA, 0.0, 0); // Fill data with random values
	init_mat_float(A, size_of_A, 0.0, 1); // Fill data with random values
	init_mat_float(LOGLIKE, size_of_LOGLIKE, LARGE_LOGLIKE, 0); // Fill data with random values

// - Loops:
//		1) For num_stars {1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 5000}
//			2) For each iteration of loop. Randomly generate.
//				* X, Y [NUM_BLOCKS_PER_DIM^2, num_stars]: The (x,y) position of stars. Defined within each block. 
// 					The upper-left corner has (0,0) coordintate.
//				* dX, dY [NUM_BLOCKS_PER_DIM^2 x num_stars, INNER]: Floats [0, 1].
//				* f: Floats [0, 256].
//				* parity_x, parity_y: 0 or 1. Determine which sub-blocks to work on.
// 				// For later: * PSFs [num_stars * NUM_BLOCKS_PER_DIM^2, (2 x AVX_CACHE)^2]: Storage for the PSFs 
// 				* start: Begin timing
//				3) For each block chosen by the xy-parities (let each thread work on)
// 					* Compute the star PSFs by multiplying the design matrix with the appropriate portion of dX. 
// 					Insert without intermitten storage. Make sure to multiply by the new flux. 
// 					(Note that killing a star is like giving a negative flux.)
// 					* Compute the new likelihood based on the updated model. Based on 48 x 48 region, region larger than the block.
// 					* Compare to the old likelihood and if the new value is smaller then update the loglike and continue.
// 					If bigger then undo the addition by subtracting what was added to the model image.
//				* end: End timing.
// 				* Compute dt and add to dT
//			Report the speed in terms of wall time.

	// // Allocate memory with aligned arrays
	// // Matrix multiplication takes the form dX x A = PSF
	// // dX [nstar, INNER] 
	// // A [INNER, NPIX2] 
	// // PSF [nstar, NPIX2]
	// int size_of_dX = INNER*nstar;
	// int size_of_A = NPIX2*INNER;
	// int size_of_PSF = NPIX2*nstar;
	// __attribute__((aligned(64))) float dX[size_of_dX];
	// __attribute__((aligned(64))) float A[size_of_A];
	// __attribute__((aligned(64))) float PSF[size_of_PSF];


	// Initialize the shared variables
	// init_mat(dX, size_of_dX, 1);
	// init_mat(A, size_of_A, 1);
	// init_mat(PSF, size_of_PSF, 0);

 // 	struct timeval time_begin, time_end;
 // 	double elapsed_time;
	// printf("Evaluating matrix vector multplication %d times.\n", count); fflush(stdout);
	// // General strategy:
	// // Multiply each row of the design matrix A with the appropriate element of dX and add the result
	// // to the resulting vector, or PSF, which acts as the accumulator. 

	// // burn-in
	// // Compute in 16 chunks
	// // Won't work if NPIX2 is not divisible by 16. 
	// for (l=0; l<count; l++){
	// 	#pragma omp parallel for
	// 	for (k=0; k<nstar; k++){
	// 		#pragma omp simd
	// 		for (j=0; j<NPIX2; j++){
	// 			for (i=0; i<INNER; i++){
	// 				// #pragma omp atomic
	// 				PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
	// 			}
	// 		}
	// 	}
	// }

	// // sum_10 (compute 625)
	// // for (l=0; l<count; l++){
	// // 	for (k=0; k<nstar; k++){
	// // 		for (i=0; i<INNER; i++){
	// // 			// xel = dX[INNER*k+i];
	// // 			// #pragma omp parallel for simd
	// // 			// #pragma omp simd				
	// // 			for (j=0; j<NPIX2; j++){
	// // 				PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
	// // 			}
	// // 		}
	// // 	}
	// // }

	// // Unnecessary optimization. Old.
	// // float xel; //current dX element
	// // int psf_row;
	// // int A_row;
	// // int dx_row;
	// // for (l=0; l<count; l++){
	// // 	for (k=0; k<nstar; k++){
	// // 		psf_row = k*NPIX2;
	// // 		dx_row = INNER*k;
	// // 		for (i=0; i<INNER; i++){
	// // 			xel = dX[dx_row+i];
	// // 			A_row = i*NPIX2;
	// // 			for (j=0; j<NPIX2; j++){
	// // 				PSF[psf_row+j] += xel * A[A_row+j];
	// // 			}
	// // 		}
	// // 	}
	// // }

	// gettimeofday(&time_begin, NULL);
	// // Compute in 16 chunks
	// // Won't work if NPIX2 is not divisible by 16. 
	// for (l=0; l<count; l++){
	// 	#pragma omp parallel		
	// 	#pragma omp for
	// 	for (k=0; k<nstar; k++){
	// 			#pragma omp simd
	// 			for (j=0; j<NPIX2; j++){
	// 				for (i=0; i<INNER; i++){
	// 					// #pragma omp atomic						
	// 					PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
	// 				}
	// 			}
	// 		}
	// }
	// gettimeofday(&time_end, NULL);



	// // Calculate the elapsed time and caculate flops
	// elapsed_time = (time_end.tv_sec - time_begin.tv_sec) + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
	// double sec = (time_end.tv_sec - time_begin.tv_sec);
	// double usec = (time_end.tv_usec - time_begin.tv_usec);	
	// printf("sec: %.3f\n", sec);
	// printf("usec: %.3f\n", usec);
	// double flops = 2*NPIX2*INNER*nstar;
	// printf("flops: %.1f\n", flops);

	// double Gflops = ((2*NPIX2*INNER*nstar)/(elapsed_time/count)) * (1.0e-09);
	// // Report
	// printf("Elapsed time : %.3f (s)\n", elapsed_time);
	// printf("FLOPS        : %.3f (GFlops)\n", Gflops);

}
