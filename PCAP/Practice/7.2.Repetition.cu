#include <stdio.h>
#include <cuda.h>
#include <string.h>

__global__ void repeat(char* word, int len, char* res){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len) {
        int start = tid*len - ((tid*(tid-1))/2); // not tid+1 because tid starts from 0 => we are technically doing tid-1 * tid-1+1 / 2
        for(int i = 0; i < len - tid; i++) {
            res[start + i] = word[i];
        }
    }
}

int main(){
    int len;
    printf("Enter length: ");
    scanf("%d",&len);
    
    char* h_word = (char*)malloc(len*sizeof(char));
    printf("Enter word: ");
    scanf("%s",h_word);

    int totalLen = (len * (len + 1)) / 2;

    char *h_res = (char*)malloc(totalLen * sizeof(char));

    char *d_word, *d_res;
    cudaMalloc((void**)&d_word, len * sizeof(char));
    cudaMalloc((void**)&d_res, totalLen * sizeof(char));

    cudaMemcpy(d_word, h_word, len * sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockDim(len);
    dim3 gridDim(1);

    repeat<<<gridDim, blockDim>>>(d_word, len, d_res);

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
