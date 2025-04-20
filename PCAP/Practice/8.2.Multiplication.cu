#include <cuda.h>
#include <stdio.h>

__global__ void rowWiseMul(int *A, int *B, int *C, int wa, int wb) {
    int row = threadIdx.x;
    for(int col = 0; col<wb ; col++){
        int ans = 0;
        for(int k=0;k<wa;k++){
            ans += A[row*wa+k]*B[k*wb+col];
        }
        C[row*wb+col]=ans;
    }
}

__global__ void columnWiseMul(int *A, int *B, int *C, int wa, int wb, int ha) {
    int col = threadIdx.x;
    for(int row=0 ; row<ha; row++){
        int ans=0;
        for(int k=0 ; k<wa ; k++){
            ans += A[row*wa+k]*B[k*wb+col];
        }
        C[row*wb+col]=ans;
    }
}

__global__ void elementWiseMul(int *A, int *B, int *C, int wa, int wb) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int ans=0;
    for(int k=0 ; k<wa ; k++){
        ans += A[row*wa+k]*B[k*wb+col];
    }
    C[row*wb+col]=ans;
}

int main() {
    int ha, wa, hb, wb;
    printf("Enter dimensions of A : ");
    scanf("%d %d", &ha, &wa);
    printf("Enter dimensions of B : ");
    scanf("%d %d", &hb, &wb);

    int *A = (int *)malloc(ha * wa * sizeof(int));
    int *B = (int *)malloc(hb * wb * sizeof(int));
    int *C = (int *)malloc(ha * wb * sizeof(int));

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < ha * wa; i++) {
        scanf("%d", &A[i]);
    }

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < hb * wb; i++) {
        scanf("%d", &B[i]);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, ha * wa * sizeof(int));
    cudaMalloc(&d_B, hb * wb * sizeof(int));
    cudaMalloc(&d_C, ha * wb * sizeof(int));

    cudaMemcpy(d_A, A, ha * wa * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, hb * wb * sizeof(int), cudaMemcpyHostToDevice);

    rowWiseMul<<<1, ha>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Row-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    columnWiseMul<<<1, wb>>>(d_A, d_B, d_C, wa, wb, ha);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    dim3 dimGrid (1,1,1);
    dim3 dimBlock (ha,wb,1);
    elementWiseMul<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nElement-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}