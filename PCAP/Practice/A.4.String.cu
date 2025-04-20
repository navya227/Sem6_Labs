#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__ void pattern(char* A, int* B, char* res, int* sp, int w){
    int row = threadIdx.x;
    int col = threadIdx.y;
    char letter = A[row * w + col];
    int start = sp[row * w + col];
    int limit = B[row * w + col];

    for(int i = 0; i < limit ; i++){
        res[start + i] = letter;
    }
}

int main(){
    int h, w ;
    printf("Enter h and w : \n");
    scanf("%d %d",&h,&w);
    char *hA = (char*)malloc(h*w*sizeof(char));
    int *hB = (int*)malloc(h*w*sizeof(int));
    int *hsp = (int*)malloc(h*w*sizeof(int));
    
    printf("Enter A : \n");
    for(int i = 0 ; i < h * w ; i++){
        scanf(" %c",&hA[i]);
    }
    
    printf("Enter B : \n");
    for(int i = 0 ; i < h * w ; i++){
        scanf("%d",&hB[i]);
    }
    
    int len = 0;
    for(int i = 0 ; i < h * w ; i++){
        hsp[i] = len;
        len += hB[i];
    }
    
    char *hres = (char*)malloc(len*sizeof(char));
    
    int *dB, *dsp;
    char *dres, *dA;
    cudaMalloc((void**)&dA,h*w*sizeof(char));
    cudaMalloc((void**)&dB,h*w*sizeof(int));
    cudaMalloc((void**)&dsp,h*w*sizeof(int));
    cudaMalloc((void**)&dres,len*sizeof(char));
    
    cudaMemcpy(dB,hB,h*w*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dsp,hsp,h*w*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dA,hA,h*w*sizeof(char),cudaMemcpyHostToDevice);
    
    dim3 dimGrid (1,1,1);
    dim3 dimBlock (h,w,1);
    pattern<<<dimGrid,dimBlock>>>(dA,dB,dres,dsp,w);

    cudaMemcpy(hres,dres,len*sizeof(char),cudaMemcpyDeviceToHost);
    
    printf("Answer : \n");
    for(int i = 0 ; i<len ; i++){
        printf("%c",hres[i]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dres);
    cudaFree(dsp);
 
    free(hA);
    free(hB);
    free(hres);
    free(hsp);

}