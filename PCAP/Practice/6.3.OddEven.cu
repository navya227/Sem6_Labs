#include <stdio.h>
#include <cuda.h>

__global__ void odd(int* arr ,int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid%2 != 0 && tid+1<n){
        if(arr[tid] > arr[tid+1]){
            int temp = arr[tid];
            arr[tid] = arr[tid+1];
            arr[tid+1] = temp;
        }
    }
}

__global__ void even(int *arr, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid%2 == 0 && tid+1<n){
        if(arr[tid]>arr[tid+1]){
            int temp = arr[tid];
            arr[tid] = arr[tid+1];
            arr[tid+1] = temp;
        }
    }
}

int main() {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    int *h_A = (int*)malloc(n * sizeof(int));  

    printf("Enter elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A;
    cudaMalloc((void**)&d_A, n * sizeof(int));

    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    for (int i = 0; i < n/2; i++) {
        odd<<<dimGrid, dimBlock>>>(d_A, n);
        even<<<dimGrid, dimBlock>>>(d_A, n);
    }

    cudaMemcpy(h_A, d_A, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", h_A[i]);
    }
    printf("\n");

    free(h_A);
    cudaFree(d_A);

    return 0;
}

