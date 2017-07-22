// Model evaluation and update prototype
// General strategy: Use each thread to update each sub-region. One thread can work on 
// one region at a time. (However, we may experiment with a thread spawning addition threads
// of its own.)
// 
// Program outline:
// - Define critical dimensions: 
// 		* AVX_CACHE = 16: Number of single floats that fit in the cacheline
//		* NPIX: Number of pixels in each dimension of the PSF.
//		* NPIX2 = NPIX^2 
//		* INNER: Inner dimension of the matrix multplication.
//		* BLOCK = 48: Chosen to be 3 times AVX_CACHE.
// 		* MARGIN = 16: Margin
// 		* REGION_WIDTH = 16: Redudant but used for clarity.
// 		Note: NPIX and NPIX2 may not be used since the design matrix will be chosen such that
//		it adds zero pixel value but cache line optimized.
// - Define global, shared variables:
//		* Image DATA: 
//		* Design matrix A [INNER, AVX_CACHE^2]
// 
// - 
// 
// - Generate a blank model of size 1024 by 1024.
// - Generate a test design matrix: Random matrix with a dimension of 625 x 10. Though the latter dimension should be fixed, I will vary 625=25^2 to see if that helps. (Cached alignmnet)
// - Pick a parity for proposal: Following the convention developed by Stephen, select one of the four sets of sub-regions to be updated. For example, 

// XOXO
// OXOX
// XOXO
// OXOX

// or 

// OXOX
// XOXO
// OXOX
// XOXO

// where X represents regions selected. I will provide some margin as well.
// - Generate stars (x, y, f) to be added for each region: For now, I will keep the same the number of stars to be added in each sub-region. (Cached alignment, hashing trick)
// - Compute PSF, multiply by flux, and add to the image: Each thread will add back one post stamp back to the matrix at a time. Variations on this might be helpful.
// - Calculate delta log likelihood: Each thread will compute the likelihood corresponding to a sub-region, one at a time. Variations on this might be helpful.
// - Accept or reject: Based on the sign of the likelihood change. Of course, the resulting chain will be meaningless but we are interested in performance.
// - Repeat and average performance.

// I will run the code on both my laptop and on KNL to compare.


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
#define REAL float
#define AVX512 16 // Number of floats that can fit into AVX512

void init_mat(float* mat, int size, float fill_val)
{
	//Initialize the matrix with random numbers
	int i; 
	for (i=0; i<size; i++){
		mat[i] = fill_val;
	}
}

void dummy_call(float* A, float* B, float*C){
	// do nothing
	return;
}


int main(int argc, char *argv[])
{	

	//Loop variables
	int i, j, k, l;
	// Number of stars. Constant.
	int nstar = 1;
 	// Number of iteration to perform.
 	int count = 1000;

	// Allocate memory with aligned arrays
	// Matrix multiplication takes the form dX x A = PSF
	// dX [nstar, INNER] 
	// A [INNER, NPIX2] 
	// PSF [nstar, NPIX2]
	int size_of_dX = INNER*nstar;
	int size_of_A = NPIX2*INNER;
	int size_of_PSF = NPIX2*nstar;
	__attribute__((aligned(64))) float dX[size_of_dX];
	__attribute__((aligned(64))) float A[size_of_A];
	__attribute__((aligned(64))) float PSF[size_of_PSF];

	// Unaligned
	// float dX[size_of_dX];
	// float A[size_of_A];
	// float PSF[size_of_PSF];	

	// Initialize the matrices
	init_mat(dX, size_of_dX, 1);
	init_mat(A, size_of_A, 1);
	init_mat(PSF, size_of_PSF, 0);

	// // Print out a part
	// for (i=0; i<10; i++){
	// 	printf("%.3f\n", A[i]);
	// }


 	struct timeval time_begin, time_end;
 	double elapsed_time;
	printf("Evaluating matrix vector multplication %d times.\n", count); fflush(stdout);
	// General strategy:
	// Multiply each row of the design matrix A with the appropriate element of dX and add the result
	// to the resulting vector, or PSF, which acts as the accumulator. 

	// burn-in
	// Compute in 16 chunks
	// Won't work if NPIX2 is not divisible by 16. 
	for (l=0; l<count; l++){
		#pragma omp parallel for
		for (k=0; k<nstar; k++){
			#pragma omp simd
			for (j=0; j<NPIX2; j++){
				for (i=0; i<INNER; i++){
					// #pragma omp atomic
					PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
				}
			}
		}
	}

	// sum_10 (compute 625)
	// for (l=0; l<count; l++){
	// 	for (k=0; k<nstar; k++){
	// 		for (i=0; i<INNER; i++){
	// 			// xel = dX[INNER*k+i];
	// 			// #pragma omp parallel for simd
	// 			// #pragma omp simd				
	// 			for (j=0; j<NPIX2; j++){
	// 				PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
	// 			}
	// 		}
	// 	}
	// }

	// Unnecessary optimization. Old.
	// float xel; //current dX element
	// int psf_row;
	// int A_row;
	// int dx_row;
	// for (l=0; l<count; l++){
	// 	for (k=0; k<nstar; k++){
	// 		psf_row = k*NPIX2;
	// 		dx_row = INNER*k;
	// 		for (i=0; i<INNER; i++){
	// 			xel = dX[dx_row+i];
	// 			A_row = i*NPIX2;
	// 			for (j=0; j<NPIX2; j++){
	// 				PSF[psf_row+j] += xel * A[A_row+j];
	// 			}
	// 		}
	// 	}
	// }

	gettimeofday(&time_begin, NULL);
	// Compute in 16 chunks
	// Won't work if NPIX2 is not divisible by 16. 
	for (l=0; l<count; l++){
		#pragma omp parallel		
		#pragma omp for
		for (k=0; k<nstar; k++){
				#pragma omp simd
				for (j=0; j<NPIX2; j++){
					for (i=0; i<INNER; i++){
						// #pragma omp atomic						
						PSF[k*NPIX2+j] += dX[INNER*k+i] * A[i*NPIX2+j];
					}
				}
			}
	}
	gettimeofday(&time_end, NULL);



	// Calculate the elapsed time and caculate flops
	elapsed_time = (time_end.tv_sec - time_begin.tv_sec) + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
	double sec = (time_end.tv_sec - time_begin.tv_sec);
	double usec = (time_end.tv_usec - time_begin.tv_usec);	
	printf("sec: %.3f\n", sec);
	printf("usec: %.3f\n", usec);
	double flops = 2*NPIX2*INNER*nstar;
	printf("flops: %.1f\n", flops);

	double Gflops = ((2*NPIX2*INNER*nstar)/(elapsed_time/count)) * (1.0e-09);
	// Report
	printf("Elapsed time : %.3f (s)\n", elapsed_time);
	printf("FLOPS        : %.3f (GFlops)\n", Gflops);

}
