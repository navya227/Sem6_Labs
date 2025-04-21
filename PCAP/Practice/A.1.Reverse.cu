#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

__global__ void reverse(char *sent, int* len, char* res, int wordCount){
    int tid = threadIdx.x;
    if (tid >= wordCount) return;

    int sp = 0;
    for(int i = 0 ; i < tid; i++){
        sp += len[i] + 1;  
    }

    for(int i = 0; i < len[tid]; i++){
        res[sp + i] = sent[sp + len[tid] - 1 - i];
    }

    if (tid < wordCount - 1) {
        res[sp + len[tid]] = '_';
    }
}

int main() {
    char input[1000];
    printf("Enter the sentence (use _ between words): ");
    scanf("%s", input);

    int totalLen = strlen(input);

    int* wordLengths = (int*)malloc(sizeof(int) * 100);  
    int wordCount = 0, currentLen = 0;

    for (int i = 0; i <= totalLen; i++) {
        if (input[i] == '_' || input[i] == '\0') {
            wordLengths[wordCount++] = currentLen;
            currentLen = 0;
        } else {
            currentLen++;
        }
    }

    char *d_sent, *d_res;
    int *d_len;
    cudaMalloc((void**)&d_sent, totalLen * sizeof(char));
    cudaMalloc((void**)&d_res, totalLen * sizeof(char));
    cudaMalloc((void**)&d_len, wordCount * sizeof(int));

    cudaMemcpy(d_sent, input, totalLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, wordLengths, wordCount * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(wordCount);
    reverse<<<1, block>>>(d_sent, d_len, d_res, wordCount);

    char* result = (char*)malloc((totalLen + 1) * sizeof(char));
    cudaMemcpy(result, d_res, totalLen * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Reversed words: %s\n", result);

    free(wordLengths);
    free(result);
    cudaFree(d_sent);
    cudaFree(d_res);
    cudaFree(d_len);

    return 0;
}
