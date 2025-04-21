#include <stdio.h>
#include <cuda.h>

#define TW 2  

__global__ void matmul(int width, float* M, float* N, float* P) {
    __shared__ float Md[TW][TW];
    __shared__ float Nd[TW][TW];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float ans = 0.0f;

    for (int m = 0; m < width / TW; m++) {

        Md[ty][tx] = M[row * width + (m * TW + tx)];
        Nd[ty][tx] = N[(m * TW + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TW; k++) {
            ans += Md[ty][k] * Nd[k][tx];
        }
        __syncthreads();
    }

    P[row * width + col] = ans;
}

int main() {
    int width;
    printf("Enter width (must be divisible by %d): ", TW);
    scanf("%d", &width);

    int size = width * width * sizeof(float);

    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);

    printf("Enter M:\n");
    for (int i = 0; i < width * width; i++) {
        scanf("%f", &h_M[i]);
    }

    printf("Enter N:\n");
    for (int i = 0; i < width * width; i++) {
        scanf("%f", &h_N[i]);
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TW, TW);                                  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dim3 dimGrid(width / TW, width / TW);
    matmul<<<dimGrid, dimBlock>>>(width, d_M, d_N, d_P);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix P:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", h_P[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
