#include <stdio.h>
#include <cuda.h>

__constant__ float d_M[1024];

__global__ void conv1D(float *N, float *P, int width, int MASK_WIDTH) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    float PValue = 0;
    int start_point = tid - (MASK_WIDTH / 2);  

    for (int i = 0; i < MASK_WIDTH; i++) {
        if (start_point + i >= 0 && start_point + i < width) {
            PValue += N[start_point + i] * d_M[i];
        }
    }

    if (tid >= 0 && tid < width) {
        P[tid] = PValue;
    }
}

int main() {
    int width, MASK_WIDTH;
    printf("Enter the size of the input array (N): ");
    scanf("%d", &width);
    printf("Enter the size of the convolution kernel (M): ");
    scanf("%d", &MASK_WIDTH);


    int size_N = width * sizeof(float);
    int size_M = MASK_WIDTH * sizeof(float);
    int size_P = width * sizeof(float);
    
    float *h_N = (float*)malloc(size_N);
    float *h_M = (float*)malloc(size_M);
    float *h_P = (float*)malloc(size_P);

    printf("Enter the elements of the input array N (size %d):\n", width);
    for (int i = 0; i < width; i++) {
        printf("N[%d] = ", i);
        scanf("%f", &h_N[i]);
    }

    printf("Enter the elements of the convolution kernel M (size %d):\n", MASK_WIDTH);
    for (int i = 0; i < MASK_WIDTH; i++) {
        printf("M[%d] = ", i);
        scanf("%f", &h_M[i]);
    }

    cudaMemcpyToSymbol(d_M, h_M, size_M);

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);

    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    conv1D<<<dimGrid, dimBlock>>>(d_N, d_P, width, MASK_WIDTH);

    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    printf("\nConvolution result (P):\n");
    for (int i = 0; i < width; i++) {
        printf("P[%d] = %f\n", i, h_P[i]);
    }

    free(h_N);
    free(h_M);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}

Enter the size of the input array (N): 7
Enter the size of the convolution kernel (M): 5
Enter the elements of the input array N (size 7):
N[0] = 1
N[1] = 2
N[2] = 3
N[3] = 4
N[4] = 5
N[5] = 6
N[6] = 7
Enter the elements of the convolution kernel M (size 5):
M[0] = 3
M[1] = 4
M[2] = 5
M[3] = 4
M[4] = 3

Convolution result (P):
P[0] = 22.000000
P[1] = 38.000000
P[2] = 57.000000
P[3] = 76.000000
P[4] = 95.000000
P[5] = 90.000000
P[6] = 74.000000
