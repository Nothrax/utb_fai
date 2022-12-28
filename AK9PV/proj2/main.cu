#include <cuda_runtime.h>
#include <iostream>

__global__ void cudaMatrixMatrixMul(float* c, float* a, float* b, int N){

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < N) && (col < N)) {
        int sum = 0;

        for (int i = 0; i < N; i++){
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void initMatrix(float *matrix, float rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols){
        matrix[row * cols + col] = row*rows + col;
    }
}

void printMatrix(float* matrix, int rows, int colums){
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < colums; col++){
            std::cout << matrix[row*rows + col] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv){
    const int N = 5;

    float* deviceA;
    float* deviceB;
    float* deviceC;
    float* hostC = (float *) malloc(N * N * sizeof(float));

    cudaMalloc<float>(&deviceA, N * N * sizeof(float));
    cudaMalloc<float>(&deviceB, N * N * sizeof(float));
    cudaMalloc<float>(&deviceC, N * N * sizeof(float));

    cudaMemset(deviceC, 0, N * N * sizeof(float));



    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(float(N) / blockDim.x), ceil(float(N) / blockDim.y));


    initMatrix<<<gridDim, blockDim>>>(deviceA, N, N);
    initMatrix<<<gridDim, blockDim>>>(deviceB, N, N);

    cudaMemcpy(&hostC[0], deviceA, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix A:\n";
    printMatrix(hostC, N, N);
    cudaMemcpy(&hostC[0], deviceB, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix B:\n";
    printMatrix(hostC, N, N);

    cudaMatrixMatrixMul<<<gridDim, blockDim>>>(deviceC, deviceA, deviceB, N);

    cudaMemcpy(&hostC[0], deviceC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result:\n";
    printMatrix(hostC, N, N);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostC);

    return 0;
}