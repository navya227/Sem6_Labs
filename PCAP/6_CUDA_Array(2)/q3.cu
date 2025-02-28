#include <stdio.h>
#include <cuda.h>

__global__ void odd_even(int *A, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    if (tid % 2 != 0 && tid + 1 < n) {
        if (A[tid] > A[tid + 1]) {
            int temp = A[tid];
            A[tid] = A[tid + 1];
            A[tid + 1] = temp;
        }
    }
}

__global__ void even_odd(int *A, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    if (tid % 2 == 0 && tid + 1 < n) {
        if (A[tid] > A[tid + 1]) {
            int temp = A[tid];
            A[tid] = A[tid + 1];
            A[tid + 1] = temp;
        }
    }
}

int main() {
    int n;
    printf("Enter the number of elements in the array: ");
    scanf("%d", &n);

    int *h_A = (int*)malloc(n * sizeof(int));  

    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A;
    cudaMalloc((void**)&d_A, n * sizeof(int));

    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    for (int i = 0; i < n; i++) {
        odd_even<<<dimGrid, dimBlock>>>(d_A, n);
        even_odd<<<dimGrid, dimBlock>>>(d_A, n);
    }

    cudaMemcpy(h_A, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_A[i]);
    }
    printf("\n");

    free(h_A);
    cudaFree(d_A);

    return 0;
}
