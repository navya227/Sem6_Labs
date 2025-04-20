#include <stdio.h>
#include <cuda.h>

__global__ void sum(int* arr, int* res, int height, int width){
    int row = threadIdx.x;
    int col = threadIdx.y;

    int ele = arr[row*width+col];

    if(ele % 2 == 0){
        int rsum = 0;
        for(int i=0;i<width;i++){
            rsum += arr[row*width+i];
        }
        res[row*width+col] = rsum;
    }
    else{
        int colsum = 0;
        for(int i=0;i<height;i++){
            colsum += arr[i*width+col];
        }
        res[row*width+col] = colsum;
    }
}

int main() {
    int h, w;
    printf("Enter matrix dimensions (h w): ");
    scanf("%d %d", &h, &w);

    int *A = (int *)malloc(h * w * sizeof(int));
    int *B = (int *)malloc(h * w * sizeof(int));

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < h * w; i++) {
        scanf("%d", &A[i]);
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, h * w * sizeof(int));
    cudaMalloc(&d_B, h * w * sizeof(int));

    cudaMemcpy(d_A, A, h * w * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock (h,w,1);
    sum <<<1, dimBlock>>>(d_A, d_B, h, w);

    cudaMemcpy(B, d_B, h * w * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output matrix B:\n");
    // for (int i = 0; i < h; i++) {
    //     for (int j = 0; j < w; j++) {
    //         printf("%d ", B[i * w + j]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < h * w; i++) {
        printf("%d ",B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);

    return 0;
}