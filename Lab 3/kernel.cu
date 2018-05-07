/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE
#include <stdio.h>

__global__ void histogram_kernel(unsigned int *inputs, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {
    extern __shared__ unsigned int private_histogram[];

    int bx = blockIdx.x;  int bd = blockDim.x;
    int tx = threadIdx.x; int gd = gridDim.x;

    int i=0;
    while(i*bd+tx < num_bins) {
        private_histogram[i*bd+tx] = 0;
	i++;
    }
    __syncthreads();

    int index = bx*bd+tx;
    int stride = bd*gd;
    i=0;
    while(i*stride+index < num_elements) {
        atomicAdd(private_histogram+inputs[i*stride+index],1);
	i++;
    }    
    __syncthreads();

    i=0;
    while(i*bd+tx < num_bins) {
        atomicAdd(bins+i*bd+tx,private_histogram[i*bd+tx]);
	i++;
    }

    return;
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
    dim3 gridDim(30,1,1);
    dim3 blockDim(32,1,1);

    histogram_kernel<<<gridDim, blockDim, num_bins*sizeof(unsigned int)>>>(input,bins,num_elements,num_bins);
}
