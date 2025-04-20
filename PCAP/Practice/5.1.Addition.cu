#include <stdio.h>
#include <cuda.h>

__global__ void add(int *a, int *b , int *c, int n){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId < n){
        c[threadId] = a[threadId]+b[threadId];
    }
}

int main(){
    int n = 5;
    int *hA = (int*)malloc(n*sizeof(int));
    int *hB = (int*)malloc(n*sizeof(int));
    int *hC = (int*)malloc(n*sizeof(int));

    for(int i=0;i<n;i++){
        scanf("%d",&hA[i]);
    }
    for(int i=0;i<n;i++){
        scanf("%d",&hB[i]);
    }

    int *dA, *dB, *dC;

    cudaMalloc((void**)&dA,n*sizeof(int));
    cudaMalloc((void**)&dB,n*sizeof(int));
    cudaMalloc((void**)&dC,n*sizeof(int));

    cudaMemcpy(dA,hA,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid (ceil(n/256.0),1,1);
    dim3 dimBlock (256,1,1);

    add<<<dimGrid,dimBlock>>>(dA,dB,dC,n);

    cudaMemcpy(hC,dC,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        printf("%d ",hC[i]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}