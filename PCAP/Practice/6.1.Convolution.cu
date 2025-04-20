#include <stdio.h>
#include <cuda.h>

__global__ void convolution(float *dN, float *dM, float *dP, int n, int m){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int start = threadId - (m/2);

    for(int i=0;i<m;i++){
        if(start+i >= 0 && start+i < n){
            Pvalue += dN[start+i]*dM[i];
        }
    }
    dP[threadId] = Pvalue;
}

int main(){
    int n = 7, m = 5;

    float *hN = (float*)malloc(n*sizeof(float));
    float *hM = (float*)malloc(m*sizeof(float));
    float *hP = (float*)malloc(n*sizeof(float));

    float *dN, *dM, *dP;
    cudaMalloc((void**)&dN,n*sizeof(float));
    cudaMalloc((void**)&dM,m*sizeof(float));
    cudaMalloc((void**)&dP,n*sizeof(float));

    for(int i=0;i<n;i++){
        scanf("%f",&hN[i]);
    }
    for(int i=0;i<m;i++){
        scanf("%f",&hM[i]);
    }
    
    cudaMemcpy(dN,hN,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dM,hM,m*sizeof(float),cudaMemcpyHostToDevice);
    
    dim3 dimGrid (ceil(n/256.0),1,1);
    dim3 dimBlock (256,1,1);
    
    convolution <<<dimGrid,dimBlock>>>(dN,dM,dP,n,m);
    
    cudaMemcpy(hP,dP,n*sizeof(float),cudaMemcpyDeviceToHost);
    
    for(int i=0;i<n;i++){
        printf("%f ",hP[i]);
    }

    cudaFree(dN);
    cudaFree(dM);
    cudaFree(dP);

    free(hN);
    free(hM);
    free(hP);

    return 0;
}