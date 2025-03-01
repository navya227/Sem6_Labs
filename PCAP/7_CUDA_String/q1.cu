#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void countWordOccurrences(const char **d_words, const int *d_wLen, int n, const char *d_key, int key_len, int *d_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int word_len = d_wLen[idx];
    bool match = true;
    if (word_len == key_len) {
        for (int i = 0; i < word_len; i++) {
            if (d_words[idx][i] != d_key[i]) {
                match = false;
                break;
            }
        }
    } else {
        match = false;
    }

    if (match) {
        atomicAdd(d_count, 1);
    }
}

int main() {
    char h_sent[1024];
    char h_key[100];

    printf("Enter the sent: ");
    fgets(h_sent, sizeof(h_sent), stdin);
    
    if (h_sent[strlen(h_sent) - 1] == '\n') {
        h_sent[strlen(h_sent) - 1] = '\0';
    }

    printf("Enter the word to search for: ");
    scanf("%s", h_key);

    int n = 0;
    int h_wLen[100];
    char h_words[100][100]; 

    int i = 0, j = 0;
    while (h_sent[i] != '\0') {
        if (h_sent[i] != ' ') {
            h_words[n][j++] = h_sent[i];
        } else {
            if (j > 0) {
                h_words[n][j] = '\0';  
                h_wLen[n] = j; 
                n++;
                j = 0;  
            }
        }
        i++;
    }

    int *d_wLen;
    const char **d_words;
    int *d_count;
    char *d_key;
    int key_len = strlen(h_key);

    cudaMalloc((void**)&d_words, n * sizeof(char*));
    cudaMalloc((void**)&d_wLen, n * sizeof(int));
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&d_key, key_len * sizeof(char));

    cudaMemcpy(d_wLen, h_wLen, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key, key_len * sizeof(char), cudaMemcpyHostToDevice);

    const char **d_words_ptr;
    cudaMalloc((void**)&d_words_ptr, n * sizeof(const char*));

    char *d_words_device[n];
    for (int i = 0; i < n; i++) {
        cudaMalloc((void**)&d_words_device[i], (h_wLen[i] + 1) * sizeof(char)); // +1 for null terminator
        cudaMemcpy(d_words_device[i], h_words[i], h_wLen[i] + 1, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_words_ptr, d_words_device, n * sizeof(const char*), cudaMemcpyHostToDevice);

    int h_count = 0;
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    countWordOccurrences<<<numBlocks, blockSize>>>(d_words_ptr, d_wLen, n, d_key, key_len, d_count);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' occurs %d times in the sent.\n", h_key, h_count);

    for (int i = 0; i < n; i++) {
        cudaFree(d_words_device[i]);
    }
    cudaFree(d_words);
    cudaFree(d_wLen);
    cudaFree(d_count);
    cudaFree(d_key);
    cudaFree(d_words_ptr);

    return 0;
}
