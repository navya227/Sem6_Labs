#include <stdio.h>
#include <cuda.h>

__global__ void compute(int* arr, int* res, int h, int w){
    int row = threadIdx.x;
    int col = threadIdx.y;

    if(row < h && col < w){
        int idx = row * w + col;
        int val = arr[idx];

        if(row == 0 || col == 0 || row == h-1 || col == w-1){
            res[idx] = val;
        } else {
            int flipped = 0, base = 1;

            while (val > 0) {
                int bit = val % 2;
                int flipped_bit = (bit == 0) ? 1 : 0;
                flipped += flipped_bit * base;
                base *= 10;  
                val /= 2;
            }

            res[idx] = flipped; 
        }
    }
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