#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include<math.h>
// the kernal
__global__ void RankSortParallel(int*input, int*output ,int N){
    // the index which the thread will work on
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < N) {
        // variable to save the rank of af the current thread cell
        int rank = 0;
        int i;
        for (i = 0; i < N; i++) {
            // if the cell is before the index cell and the value in that cell <= the value in the thread index cell then rank++
            if (i < index) {
                if (input[i] <= input[index]) {
                    rank++;
                }
            }
            // if the cell is after the index cell and the value in that cell < the value in the thread index cell then rank++
            else if (i > index) {
                if (input[i] < input[index]) {
                    rank++;
                }
            }
        }
    // put the value in its proper location(the rank)
    output[rank] = input[index];
    }
}

int main(){

    // initlize the data and some variables
    int data[] = {8,16,2,4,-5,1,6,-16,20,-16,10};
    int size = sizeof(data);
    int numOfelementsInInput = size / sizeof(int);


    int i;
    printf("the input data:\n");
    for (i = 0; i < numOfelementsInInput; i++) {
        printf("%d,", data[i]);
    }
    
    // create an array for input data in the gpu memory
    int* input_gpu= NULL;
    cudaMalloc((void**)&input_gpu,size);

    // create an array to save the data in cpu(host) after executing the kernal
    int* output = (int*)malloc(size);

    // create an array for output data in the gpu memory
    int* output_gpu = NULL;
    cudaMalloc((void**)&output_gpu, size);
    
    
    // cupy data from the data array(in host) to the input array in the gpu
    cudaMemcpy(input_gpu, data, size, cudaMemcpyHostToDevice);
    
    //run the kernal with an array of blocks, each block has an array of threads    
    float threadsPerBlock = 512.0;
    RankSortParallel<<<ceil(numOfelementsInInput/threadsPerBlock),threadsPerBlock>>>(input_gpu, output_gpu, numOfelementsInInput);
    
    // wait the kernal to finish executing
    cudaDeviceSynchronize();

    // copy the results back to cpu
    cudaMemcpy(output, output_gpu, size, cudaMemcpyDeviceToHost);
    
    // print the output
    printf("\nThe sorted array:\n");
    for (i = 0; i < numOfelementsInInput; i++) {
        printf("%d,", output[i]);
    }
}
