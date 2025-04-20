#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__ void countWordOccurrences(char* sentence, char* word, int sentLen, int wordLen, int* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx + wordLen > sentLen) return;

    bool match = true;
    for (int i = 0; i < wordLen; i++) {
        if (sentence[idx + i] != word[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomicAdd(count, 1);
    }
}

int main() {
    int sentLen, wordLen;

    printf("Enter length of the sentence: ");
    scanf("%d", &sentLen);

    printf("Enter length of the word: ");
    scanf("%d", &wordLen);

    char* h_sentence = (char*)malloc((sentLen) * sizeof(char));
    char* h_word = (char*)malloc((wordLen) * sizeof(char));

    printf("Enter the sentence (no spaces): ");
    scanf("%s", h_sentence);

    printf("Enter the word to count: ");
    scanf("%s", h_word);

    int *h_count = (int*)malloc(sizeof(int));

    char *d_sentence, *d_word;
    int *d_count;

    cudaMalloc((void**)&d_sentence, sentLen * sizeof(char));
    cudaMalloc((void**)&d_word, wordLen * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, sentLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, wordLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, h_count, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(128);
    dim3 gridDim((sentLen + blockDim.x - 1) / blockDim.x);

    countWordOccurrences<<<gridDim, blockDim>>>(d_sentence, d_word, sentLen, wordLen, d_count);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word \"%s\" appears %d times in the sentence.\n", h_word, h_count);

    free(h_sentence);
    free(h_word);
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
