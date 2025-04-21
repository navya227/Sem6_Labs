#include <stdio.h>
#include <cuda.h>

__global__ void inclusiveScan(int* input, int* output) {
    __shared__ int temp[10];
    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    for (int offset = 1; offset <= tid; offset++) {
        temp[tid] += input[tid - offset];
    }
    __syncthreads();

    output[tid] = temp[tid];
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int* h_input = (int*)malloc(n * sizeof(int));
    int* h_output = (int*)malloc(n * sizeof(int));

    printf("Enter elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_input[i]);
    }

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    inclusiveScan<<<1, n>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Inclusive Scan Output: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
