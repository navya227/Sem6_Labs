#include <stdio.h>
#include <cuda.h>
#include <string.h>

__global__ void copy(char* word, int len, char* res){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(int i = 0 ; i<len ; i++){
        res[tid*len + i] = word[i];
    }
}

int main(){
    int len;
    printf("Enter length: ");
    scanf("%d",&len);

    int n;
    printf("Enter N: ");
    scanf("%d",&n);
    
    char* h_word = (char*)malloc(len*sizeof(char));
    printf("Enter word: ");
    scanf("%s",h_word);

    int totalLen = len * n;

    char *h_res = (char*)malloc(totalLen * sizeof(char));

    char *d_word, *d_res;
    cudaMalloc((void**)&d_word, len * sizeof(char));
    cudaMalloc((void**)&d_res, totalLen * sizeof(char));

    cudaMemcpy(d_word, h_word, len * sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockDim(n);
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