#include <stdio.h>
#include <cuda.h>

__global__ void sorting(int *arr, int *ans, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<n){
        int pos = 0;
        for(int i=0;i<n;i++){
            if(arr[i]<arr[tid] || ((arr[i] == arr[tid]) && (i<tid))){
                pos++;
            }
        }
        ans[pos] = arr[tid];
    }
}

int main(){
    printf("Enter n: \n");
    int n;
    scanf("%d",&n);

    printf("Enter arr: \n");
    int *h_arr = (int*)malloc(n*sizeof(int));
    int *h_ans = (int*)malloc(n*sizeof(int));
    for(int i=0;i<n;i++){
        scanf("%d",&h_arr[i]);
    }

    int *d_arr, *d_ans;
    cudaMalloc((void**)&d_arr,n*sizeof(int));
    cudaMalloc((void**)&d_ans,n*sizeof(int));
    cudaMemcpy(d_arr,h_arr,n*sizeof(int),cudaMemcpyHostToDevice);

    dim3 dimGrid (ceil(n/256.0),1,1);
    dim3 dimBlock (256,1,1);

    sorting <<<dimGrid,dimBlock>>>(d_arr,d_ans,n);

    cudaMemcpy(h_ans,d_ans,n*sizeof(int),cudaMemcpyDeviceToHost);
    
    for(int i=0;i<n;i++){
        printf("%d ",h_ans[i]);
    }
}