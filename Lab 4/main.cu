/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

#define SEG_SIZE 1000 
#define BLOCK_SIZE 256 

int main (int argc, char *argv[])
{
    unsigned VecSize;
    if (argc == 1) {
        VecSize = 1000000;
    } else if (argc == 2) {
        VecSize = atoi(argv[1]);
    } else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }
	
    //set standard seed
    srand(217); //Defualt value 217, DO NOT TOUCH

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...\n"); fflush(stdout);
    startTime(&timer);
    
    cudaStream_t stream0, stream1;
    printf("Creating stream0...\n"); fflush(stdout);
    cuda_ret = cudaStreamCreate(&stream0);
    if(cuda_ret != cudaSuccess) {
        printf("Failed to create stream 0, exiting"); fflush(stdout);
	return 0;
    }
    printf("Stream 0 created!\nCreating stream1...\n"); fflush(stdout);
    cuda_ret = cudaStreamCreate(&stream1);
    if(cuda_ret != cudaSuccess) {
        printf("Failed to create stream 0, exiting"); fflush(stdout);
        return 0;
    }
    printf("Stream 1 created!\n"); fflush(stdout);

    float *A_d0, *B_d0, *C_d0; // device memory for stream 0
    float *A_d1, *B_d1, *C_d1; // device memory for stream 1

    float *A_h, *B_h, *C_h;
    size_t d0_sz, d1_sz;

    dim3 dim_grid, dim_block;

    d0_sz = VecSize/2;
    d1_sz = (VecSize-1)/2 + 1;

    A_h = (float*) malloc(sizeof(float)*VecSize);
    for (unsigned int i=0; i < VecSize; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc(sizeof(float)*VecSize);
    for (unsigned int i=0; i < VecSize; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc(sizeof(float)*VecSize); 

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize,1);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMalloc((void **) &A_d0, sizeof(float)*d0_sz);
    cudaMalloc((void **) &B_d0, sizeof(float)*d0_sz);
    cudaMalloc((void **) &C_d0, sizeof(float)*d0_sz);

    cudaMalloc((void **) &A_d1, sizeof(float)*d1_sz);
    cudaMalloc((void **) &B_d1, sizeof(float)*d1_sz);
    cudaMalloc((void **) &C_d1, sizeof(float)*d1_sz);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Run streams here ------------------------------------------

    printf("Running streaming operations..."); fflush(stdout);
    startTime(&timer);

    dim3 DimGrid((SEG_SIZE-1)/BLOCK_SIZE+1,1,1);
    dim3 DimBlock(BLOCK_SIZE,1,1); 

    for(int i=0;i<VecSize;i+=SEG_SIZE*2) {
	if(i+SEG_SIZE*2<VecSize) {
            cudaMemcpyAsync(A_d0, A_h+i, SEG_SIZE*sizeof(float),cudaMemcpyHostToDevice, stream0);
	    cudaMemcpyAsync(B_d0, B_h+i, SEG_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream0);
	    cudaMemcpyAsync(A_d1, A_h+i+SEG_SIZE, SEG_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync(B_d1, B_h+i+SEG_SIZE, SEG_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream1);
	} else {
            cudaMemcpyAsync(A_d0, A_h+i, (VecSize-i)/2*sizeof(float),cudaMemcpyHostToDevice, stream0);
            cudaMemcpyAsync(B_d0, B_h+i, (VecSize-i)/2*sizeof(float),cudaMemcpyHostToDevice,stream0);
            cudaMemcpyAsync(A_d1, A_h+i+(VecSize-i)/2, ((VecSize-i-1)/2+1)*sizeof(float),cudaMemcpyHostToDevice,stream1);
            cudaMemcpyAsync(B_d1, B_h+i+(VecSize-i)/2, ((VecSize-i-1)/2+1)*sizeof(float),cudaMemcpyHostToDevice,stream1);
	}

	VecAdd<<<DimGrid, DimBlock, 0, stream0>>>(d0_sz, A_d0, B_d0, C_d0);
	VecAdd<<<DimGrid, DimBlock, 0, stream1>>>(d1_sz, A_d1, B_d1, C_d1);

	if(i+SEG_SIZE*2<VecSize) {
	    cudaMemcpyAsync(C_h+i, C_d0, SEG_SIZE*sizeof(float),cudaMemcpyDeviceToHost,stream0);
	    cudaMemcpyAsync(C_h+i+SEG_SIZE, C_d1, SEG_SIZE*sizeof(float),cudaMemcpyDeviceToHost,stream1);
	} else {
            cudaMemcpyAsync(C_h+i, C_d0, (VecSize-i)/2*sizeof(float),cudaMemcpyDeviceToHost,stream0);
	    cudaMemcpyAsync(C_h+i+(VecSize-i)/2, C_d1, ((VecSize-i-1)/2+1)*sizeof(float),cudaMemcpyDeviceToHost,stream1);
	}
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cudaFree(A_d0);
    cudaFree(B_d0);
    cudaFree(C_d0);

    cudaFree(A_d1);
    cudaFree(B_d1);
    cudaFree(C_d1);

    return 0;

}
