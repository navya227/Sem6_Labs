#include <stdio.h>
#include <cuda.h>

__global__ void compute(int* arr, int* res, int h, int w){
    int row = threadIdx.x;
    int col = threadIdx.y;
    int rowsum = 0;
    int colsum = 0;
    for(int i=0 ; i<w ; i++){
        rowsum += arr[row * w + i];
    }
    for(int i=0 ; i<h ; i++){
        colsum += arr[i * w + col];
    }
    res[row * w + col] = rowsum + colsum;
    
}
int main(){
    int h, w;
    printf("Enter row and column: \n");
    scanf("%d %d",&h,&w);
    
    int size = h*w*sizeof(int);
    int *h_arr = (int*)malloc(size);
    int *h_res = (int*)malloc(size);
    
    printf("Enter elements: \n");
    for(int i=0;i<h*w;i++){
        scanf("%d",&h_arr[i]);
    }
    
    int* d_arr, *d_res;
    cudaMalloc((void**)&d_arr,size);
    cudaMalloc((void**)&d_res,size);

    cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice);
    dim3 dimGrid (1,1,1);
    dim3 dimBlock (h,w,1);
    compute<<<dimGrid,dimBlock>>>(d_arr,d_res,h,w);
    cudaMemcpy(h_res,d_res,size,cudaMemcpyDeviceToHost);
    
    printf("Result: \n");
    for(int i=0 ; i<h ; i++){
        for(int j=0 ; j<w;j++){
            printf("%d ",h_res[i*w+j]);
        }
        printf("\n");
    }
}