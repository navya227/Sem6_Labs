#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void inclusiveScan(int *d_input, int *d_output, int n) {
    extern __shared__ int shared_mem[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        shared_mem[threadIdx.x] = d_input[idx];
    } else {
        shared_mem[threadIdx.x] = 0;  
    }

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x >= offset) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x - offset];
        }
        __syncthreads();
    }

    if (idx < n) {
        d_output[idx] = shared_mem[threadIdx.x];
    }
}

int main() {
    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *h_input = (int *)malloc(n * sizeof(int));
    int *h_output = (int *)malloc(n * sizeof(int));
    int *d_input, *d_output;

    printf("Enter elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_input[i]);
    }

    cudaMalloc((void **)&d_input, n * sizeof(int));
    cudaMalloc((void **)&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;  // Maximum threads per block
    int gridSize = (n + blockSize - 1) / blockSize;

    inclusiveScan<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Inclusive scan result:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

Enter the size of the array: 5
Enter elements of the array:
1 1 1 1 1
Inclusive scan result:
1 2 3 4 5 
