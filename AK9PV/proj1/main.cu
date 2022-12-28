#include <cuda_runtime.h>
#include <iostream>

const int VECTOR_SIZE = 512;

__global__ void sumVectors(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < VECTOR_SIZE){
        c[i] = a[i] + b[i];
    }
}

__global__ void initVectors(int *a, int *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        a[i] = i + 1;
        b[i] = i + 1;
    }
}

int main()
{
    int sum[VECTOR_SIZE];
    int *a, *b, *c;

    // Allocate memory on the GPU
    cudaMalloc((void**)&a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&c, VECTOR_SIZE * sizeof(int));

    // Initialize vectors on the GPU
    initVectors<<<1, VECTOR_SIZE>>>(a, b, VECTOR_SIZE);

    // Invoke the kernel
    sumVectors<<<1, VECTOR_SIZE>>>(a, b, c);

    // Copy the result from the GPU to the CPU
    cudaMemcpy(sum, c, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < VECTOR_SIZE; i++){
        std::cout <<sum[i] << std::endl;
    }

    // Free memory on the GPU and CPU
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
