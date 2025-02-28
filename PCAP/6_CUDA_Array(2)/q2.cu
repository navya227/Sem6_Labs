#include <stdio.h>
#include <cuda.h>
#include <cmath>

__global__ void parSort(int *A, int *O, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int data = A[tid];
        int pos = 0;
        
        for (int i = 0; i < n; i++) {
            if ((A[i] < data) || (A[i] == data && i < tid)) {
                pos++;
            }
        }

        O[pos] = data;
    }
}

int main() {
    int n;
    printf("Enter the number of elements in the array: ");
    scanf("%d", &n);

    int *h_A = (int*)malloc(n * sizeof(int)); 
    int *h_O = (int*)malloc(n * sizeof(int)); 

    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_O;
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_O, n * sizeof(int));

    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    parSort<<<dimGrid, dimBlock>>>(d_A, d_O, n);

    cudaMemcpy(h_O, d_O, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_O[i]);
    }
    printf("\n");

    free(h_A);
    free(h_O);
    cudaFree(d_A);
    cudaFree(d_O);

    return 0;
}
