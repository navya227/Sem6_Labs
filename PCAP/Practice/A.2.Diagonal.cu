#include <stdio.h>
#include <cuda.h>

__global__ void calculate(int* arr, int* res, int n){
    int row = threadIdx.x;
    int col = threadIdx.y;
    int ans;
    int ele = arr[row*n+col];
    if(row == col){
        ans = 0;
    }
    else if(row < col){
        ans = 1;
        for(int i=1; i<=ele; i++){
            ans *= i;
        }
    }
    else{
        ans = 0;
        while(ele > 0){
            int dig = ele%10;
            ans += dig;
            ele /= 10;
        }
    }
    res[row*n+col] = ans;
}
int main(){

    int n;

    printf("Enter n: \n");
    scanf("%d",&n);
    
    int *h_arr = (int*)malloc(sizeof(int)*n*n);
    int *h_res = (int*)malloc(sizeof(int)*n*n);
    
    printf("Enter Elements: \n");
    for(int i=0;i<n*n;i++){
        scanf("%d",&h_arr[i]);
    }
    
    int *d_arr,*d_res;
    cudaMalloc((void**)&d_arr,n*n*sizeof(int));
    cudaMalloc((void**)&d_res,n*n*sizeof(int));
    
    cudaMemcpy(d_arr,h_arr,n*n*sizeof(int),cudaMemcpyHostToDevice);

    dim3 dimGrid (1,1,1);
    dim3 dimBlock (n,n,1);
    calculate <<<dimGrid,dimBlock>>>(d_arr,d_res,n);

    cudaMemcpy(h_res,d_res,n*n*sizeof(int),cudaMemcpyDeviceToHost);
    
    // for(int i=0;i<n*n;i++){
    //     printf("%d",&h_res[i]);
    // }
    
    for(int i=0 ; i<n ; i++){
        for(int j=0 ; j<n;j++){
            printf("%d ",h_res[i*n+j]);
        }
        printf("\n");
    }

    cudaFree(d_arr);
    cudaFree(d_res);
    free(h_arr);
    free(h_res);

}