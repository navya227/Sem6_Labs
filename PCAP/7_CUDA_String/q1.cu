#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024
#define MAX_WORD_LENGTH 100

__global__ void CUDACountWord(char *text, int textLength, char *word, int wordLength, unsigned int *d_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process valid starting positions
    if (i <= textLength - wordLength) {
        bool match = true;
        
        // Compare each character of the word
        for (int j = 0; j < wordLength; j++) {
            if (text[i + j] != word[j]) {
                match = false;
                break;
            }
        }
        
        // If we found a match, increment the counter
        if (match) {
            atomicAdd(d_count, 1);
        }
    }
}

int main() {
    char text[N];
    char word[MAX_WORD_LENGTH];
    char *d_text, *d_word;
    unsigned int count = 0, result;
    unsigned int *d_count;
    
    // Get the input text
    printf("Enter a string: ");
    fgets(text, N, stdin);
    int textLength = strlen(text);
    if (text[textLength - 1] == '\n') text[textLength - 1] = '\0'; // Remove newline
    textLength = strlen(text); // Recalculate length after newline removal
    
    // Get the word to search for
    printf("Enter word to search: ");
    fgets(word, MAX_WORD_LENGTH, stdin);
    int wordLength = strlen(word);
    if (word[wordLength - 1] == '\n') word[wordLength - 1] = '\0'; // Remove newline
    wordLength = strlen(word); // Recalculate length after newline removal
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Allocate memory
    cudaMalloc((void**)&d_text, textLength * sizeof(char));
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    
    // Copy data to device
    cudaMemcpy(d_text, text, textLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
    }
    
    // Calculate grid and block dimensions
    // For a simple approach, use 1 thread per potential starting position
    int threadsPerBlock = 256;
    int numBlocks = (textLength + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    CUDACountWord<<<numBlocks, threadsPerBlock>>>(d_text, textLength, d_word, wordLength, d_count);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(&result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("Total occurrences of '%s' = %u\n", word, result);
    printf("Time taken: %f ms\n", elapsedTime);
    
    // Free memory
    cudaFree(d_text);
    cudaFree(d_word);
    cudaFree(d_count);
    
    return 0;
}