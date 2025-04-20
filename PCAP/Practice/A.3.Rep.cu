#include <stdio.h>
#include <cuda.h>
#include <string.h>

__global__ void copy(char* word, int len, char* res){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int start = (tid * (tid+1))/2;
    for(int i = 0 ; i<tid+1 ; i++){
        res[start + i] = word[tid];
    }
}

int main(){
    int len;
    printf("Enter length: ");
    scanf("%d",&len);
    
    char* h_word = (char*)malloc(len*sizeof(char));
    printf("Enter word: ");
    scanf("%s",h_word);

    int totalLen = (len * (len+1))/2;

    char *h_res = (char*)malloc(totalLen * sizeof(char));

    char *d_word, *d_res;
    cudaMalloc((void**)&d_word, len * sizeof(char));
    cudaMalloc((void**)&d_res, totalLen * sizeof(char));

    cudaMemcpy(d_word, h_word, len * sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockDim(len);
    dim3 gridDim(1);

    copy<<<gridDim, blockDim>>>(d_word, len, d_res);

    cudaMemcpy(h_res, d_res, totalLen * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < totalLen; i++) {
        printf("%c", h_res[i]);
    }
    printf("\n");

    cudaFree(d_word);
    cudaFree(d_res);
    free(h_res);

    return 0;
}