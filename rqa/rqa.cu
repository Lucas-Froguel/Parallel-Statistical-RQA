
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


#define Q 200


__global__ void calculateLAM(int *xrand, int *yrand, float *d_map, float *d_LAM, int N, int lmin, float e) {
    
    // data that we want to load to the shared memory
    // each thread loads threadsPerBlock/Q elements from global to shared memory
    __shared__ float mapx[Q], mapy[Q];
    __shared__ bool microstate[Q * Q];
    __shared__ int histogram[Q];
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
        // clean histogram on shared memory
        histogram[k] = 0;
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
        // printf("\nThread %i (global %i, block %i) on position i=%i and j=%i put on %i the value %i", tid, gtid, blockIdx.x, iq, jq, k, m);
    }
    // ensures the CRP / microstate is finished
    __syncthreads();
    
    // each thread will look for lines in one row of the microstate
    // values will be CRP[tid, k], k\in[0, Q-1]
    // assumes Q < threadsPerBlock (which is very reasonable)
    if (tid < Q){
        int line_length = 0;
        for (int k = 0; k < Q; k++){
            if (microstate[k + tid * Q] == 1){
                // printf("\nHey, we detected a 1 by tid=%i (block=%i) at i=%i, j=%i", tid, blockIdx.x, k, tid);
                line_length += 1;
                if (k == Q - 1){
                    // printf("\nLine detected by tid=%i (block=%i) at i=%i, j=%i of length %i", tid, blockIdx.x, k, tid, line_length);
                    atomicAdd(&histogram[line_length-1], 1);
                    line_length = 0;
                }
            }
            else if (line_length > 0 && microstate[k + tid * Q] == 0){
                // printf("\nLine detected by tid=%i (block=%i) at i=%i, j=%i of length %i", tid, blockIdx.x, k, tid, line_length);
                atomicAdd(&histogram[line_length-1], 1);
                line_length = 0;
            } 
        }
    }
    // ensures the histogram is ready
    __syncthreads();
    
    // each blocks calculates the LAM in its microstate
    // should be a cheap/fast operation, but has to be done on one thread
    if (tid == 0){
        float LAM = 0, total = 0;
        for (int k = 0; k < Q; k++){
            if (k + 1 >= lmin){
                LAM += histogram[k] * (k+1);
            }
            total += histogram[k] * (k+1);
            // printf("\nHist of k=%i is %i", k, histogram[k]);
        }
        if (total != 0) d_LAM[blockIdx.x] = LAM / total;
    }
    
}

/**
 * Host main routine
 */
int main(void){
    // Check errors
    cudaError_t cudaerr = cudaSuccess;

    // Define the vector length to be used and compute its size
    int numElements = 1000 * 1000;
    size_t size = numElements * sizeof(float);
    int threadsPerBlock = Q*Q; // max threads per MP / 2
    if (threadsPerBlock > 768) threadsPerBlock = 768; // we want to maximize thw work
    //int numBlocks = 2 * 28; // 2 * num of MPs
    int numBlocks = numElements / Q; // Q < numElements always
    int numThreads = threadsPerBlock * numBlocks;
    
    printf("\n-----SETUP-----\n\nnumElements=%i\nnumBlocks=%i\nthreadsPerBlock=%i\nQ=%i \n\n", numElements, numBlocks, threadsPerBlock, Q);
    
    // Parameters
    float r = 4;
    int lmin = 2;
    float e = 0.3;
    
    // Allocate the host vectors
    float *h_map = (float *)malloc(size);
    float *h_LAM = (float *)malloc(numBlocks * sizeof(float)); // this will be the histogram of lines, from size 1 to Q
    
    // Calculate logistic map
    h_map[0] = 0.4;
    for (int i = 1; i < numElements; i++){
        h_map[i] = r * h_map[i-1] * (1-h_map[i-1]);
    }

    // Allocate the device vector
    float *d_map = NULL;
    cudaMalloc((void **)&d_map, size);
    float *d_LAM = NULL;
    cudaMalloc((void **)&d_LAM, numBlocks * sizeof(float));
    
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
        // printf("\nx=%i, y=%i", h_xrand[i], h_yrand[i]);
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
    calculateLAM<<<numBlocks, threadsPerBlock>>>(d_xrand, d_yrand, d_map, d_LAM, numElements, lmin, e);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess){
        printf("Device Synchronize failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    
    // Copy from GPU to CPU 
    cudaMemcpy(h_LAM, d_LAM, numBlocks * sizeof(int), cudaMemcpyDefault);
    
    // Calculate LAM using the LAMs of each block
    float LAM = 0;
    for (int k = 0; k < numBlocks; k++){
        LAM += h_LAM[k];
        // printf("\nLAM(k=%i)=%f    LAM_t=%f", k, h_LAM[k], LAM);
    }
    LAM = LAM / numBlocks;
    
    // Print LAM
    printf("\n\nThe laminarity is LAM=%f", LAM);
    
    // Stop chronometer
    chrono_stop(&chrono_exec);
    
    // Report time statistics
    chrono_reportTimeDetailed(&chrono_exec);
    total_time = chrono_get_TimeInLoop(&chrono_exec, 1);
    
    // Free host and device memory
    free(h_map);

    return 0;
}

