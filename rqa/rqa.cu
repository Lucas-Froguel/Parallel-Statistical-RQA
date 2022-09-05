
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
//#include <cuda_runtime.h>  // comentado por WZ (não precisamos disso!)

//#include <helper_cuda.h>
#include "helper_cuda.h"   // mudado aqui por WZ
#include <cuda_runtime.h>
#include "chrono.c"

#include <curand.h>
#include <curand_kernel.h>


#ifdef RTX3060
  #define GPU_NAME "RTX3060-------------"
  #define MP 28
  #define THREADS_PER_BLOCK 768   // can be defined in the compilation line with -D
  #define THREADS_PER_MP 1536
#endif


#define Q 4


__global__ void calculateLAM(int *xrand, int *yrand, float *d_map, int *d_histogram, int N, int lmin, float e) {
    
    // data that we want to load to the shared memory
    // each thread loads threadsPerBlock/Q elements from global to shared memory
    extern __shared__ float mapx[Q], mapy[Q];
    extern __shared__ bool microstate[Q * Q];
    int tid = threadIdx.x;
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    
    // get two random indexes 
    i = xrand[blockIdx.x];
    j = yrand[blockIdx.x];
    
    // load data [i, i+Q] and [j, j+Q] to shared memory
    // each thread loads [tid, tid+blockDim, ...] until all elements are loaded
    for (int k = tid; k < Q; k = k + blockDim.x){
        // i, j are such that i + Q, j + Q < N
        mapx[k] = d_map[i + k];
        mapy[k] = d_map[j + k];
    }
    // ensures all threads finished loading the data
    __syncthreads();
    
    // calculate CRP / microstate
    for (int k = tid; k < Q * Q; k = k + blockDim.x){
        bool m = 0;
        int iq = k % Q, jq = k / Q;
        float val = abs(mapx[iq] - mapy[jq]);
        if (val < e) m = 1;
        microstate[k] = m;
        printf("\nThread %i (global %i, block %i) on position i=%i and j=%i put on %i the value %i", tid, gtid, blockIdx.x, iq, jq, k, m);
    }
    // ensures the CRP / microstate is finished
    __syncthreads();
    
    // each thread will look for lines in one row of the microstate
    // values will be CRP[tid, k]
    for (int k = 0; k < Q; k++){
        int line_length = 0;
        if (microstate[k + tid * Q] == 1){
            line_length += 1;
        }
        else if (microstate[k + tid * Q] == 0){
            atomicAdd(&d_histogram[line_length], 1);
            line_length = 0;
        }
    }
    
}

/**
 * Host main routine
 */
int main(void){
    // Check errors
    cudaError_t cudaerr = cudaSuccess;

    // Define the vector length to be used and compute its size
    int numElements = 2 * 10; // * 1000;
    size_t size = numElements * sizeof(float);
    int threadsPerBlock = 768; // max threads per MP / 2
    //int numBlocks = 2 * 28; // 2 * num of MPs
    int numBlocks = 1 + numElements / Q;
    int numThreads = threadsPerBlock * numBlocks;
    
    // Parameters
    float r = 4;
    int lmin = 2;
    float e = 0.3;
    int q = Q;
    
    // Allocate the host vectors
    float *h_map = (float *)malloc(size);
    float *h_LAM = (float *)malloc(sizeof(float));
    int *h_histogram = (int *)malloc(q * sizeof(int)); // this will be the histogram of lines, from size 1 to Q
    
    // Calculate logistic map
    h_map[0] = 0.4;
    for (int i = 1; i < numElements; i++){
        h_map[i] = r * h_map[i-1] * (1-h_map[i-1]);
    }

    // Allocate the device vector
    float *d_map = NULL;
    cudaMalloc((void **)&d_map, size);
    float *d_LAM = NULL;
    cudaMalloc((void **)&d_LAM, sizeof(float));
    int *d_histogram = NULL;
    cudaMalloc((void **)&d_histogram, q * sizeof(int));
    
    // Copy the vector from host to device
    cudaerr = cudaMemcpy(d_map, h_map, size, cudaMemcpyHostToDevice);
    if (cudaerr != cudaSuccess){
        printf("\n Memory copy failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    
    
    // Random numbers setup
    int *h_xrand = (int *)malloc(numBlocks * sizeof(int));
    int *h_yrand= (int *)malloc(numBlocks * sizeof(int));
    srand(time(NULL));   // Initialization, should only be called once.
    for (int i = 0; i < numBlocks; i++){
        h_xrand[i] = rand() % (numElements - Q);
        h_yrand[i] = rand() % (numElements - Q);
        printf("\nx=%i, y=%i", h_xrand[i], h_yrand[i]);
    }
    int *d_xrand = NULL;
    int *d_yrand = NULL;
    cudaMalloc((void **)&d_xrand, numBlocks * sizeof(int));
    cudaMalloc((void **)&d_yrand, numBlocks * sizeof(int));
    cudaMemcpy(d_xrand, h_xrand, numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yrand, h_yrand, numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    
    // Launch the chronometer
    long long total_time = 0;
    chronometer_t chrono_exec;
    chrono_reset(&chrono_exec);
    chrono_start(&chrono_exec);
    
    // Activate kernel
    printf("\n\n-----Launching Kernel-----\n\n");
    calculateLAM<<<numBlocks, threadsPerBlock>>>(d_xrand, d_yrand, d_map, d_histogram, numElements, lmin, e);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess){
        printf("Device Synchronize failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    
    // Copy from GPU to CPU
    cudaMemcpy(h_LAM, d_LAM, sizeof(float), cudaMemcpyDefault);
    // Print LAM
    printf("\n\nThe LAM value is: %f", h_LAM);
    
    // Stop chronometer
    chrono_stop(&chrono_exec);
    
    // Report time statistics
    chrono_reportTimeDetailed(&chrono_exec);
    total_time = chrono_get_TimeInLoop(&chrono_exec, 1);
    
    // Free host and device memory
    free(h_map);

    return 0;
}

