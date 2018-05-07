/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    
    // Calculate position of row and col
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure it is inside range
    if (row<m && col<n) {
        float val=0;
	for(int i=0;i<k;i++) {
            val += A[row*k+i]*B[i*n+col];
	}
	C[row*n+col] = val;
    }
}

__global__ void mysgemm_tiled(int m, int n, int k, const float *A, const float *B, float *C) {
    __shared__ float ds_M[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_N[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*blockDim.y+ty;
    int col = bx*blockDim.x+tx;

    float val = 0;

    for(int i=0;i<(k-1)/TILE_SIZE+1;++i) {
	if(row<m && TILE_SIZE*i+tx<k) {
            ds_M[ty][tx] = A[row*k+tx+i*TILE_SIZE];
	} else {
            ds_M[ty][tx] = 0;
	}
	if(col<n && TILE_SIZE*i+ty<k) {
            ds_N[ty][tx] = B[(ty+i*TILE_SIZE)*n+col];
	} else {
            ds_N[ty][tx] = 0;
	}

	__syncthreads();

        if(row<m && col<n)
	    for(int j=0;j<TILE_SIZE;++j) 
                val+=ds_M[ty][j]*ds_N[j][tx];

	__syncthreads();
    }

    if(row<m && col<n)
        C[row*n+col] = val;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dimGrid((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1,1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    // Invoke CUDA kernel -----------------------------------------------------
    //INSERT CODE HERE
    mysgemm_tiled<<<dimGrid, dimBlock>>>(m,n,k,A,B,C);
}


