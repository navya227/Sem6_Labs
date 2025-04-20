#include<stdio.h>
#include<cuda.h>

__global__ void row(int* a,int* b,int* c,int height,int width){
    int row = threadIdx.x;
    for(int col=0 ; col<width ; col++){
        c[row*width+col]=a[row*width+col]+b[row*width+col];
    }
}
__global__ void column(int* a,int* b,int* c,int height,int width){
    int col = threadIdx.x;
    for(int row=0;row<height;row++){
        c[row*width+col]=a[row*width+col]+b[row*width+col];
    }
}
__global__ void element(int* a,int* b,int* c,int height,int width){
    int row = threadIdx.y;
    int col = threadIdx.x;
    c[row*width+col]=a[row*width+col]+b[row*width+col];
}
int main(){
    int n,m;
    printf("Enter m: \n");
    scanf("%d",&m);
    printf("Enter n: \n");
    scanf("%d",&n);

    int *hA = (int*)malloc(m*n*sizeof(int));
    int *hB = (int*)malloc(m*n*sizeof(int));
    int *hC = (int*)malloc(m*n*sizeof(int));

    printf("Enter A: \n");
    for(int i=0;i<m*n;i++){
        scanf("%d",&hA[i]);
    }
    printf("Enter B: \n");
    for(int i=0;i<m*n;i++){
        scanf("%d",&hB[i]);
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a,m*n*sizeof(int));
    cudaMalloc((void**)&d_b,m*n*sizeof(int));
    cudaMalloc((void**)&d_c,m*n*sizeof(int));

    cudaMemcpy(d_a,hA,m*n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,hB,m*n*sizeof(int),cudaMemcpyHostToDevice);

    row<<<1,m>>>(d_a,d_b,d_c,m,n);
    cudaMemcpy(hC,d_c,m*n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("Row C: \n");
    for(int i=0;i<m*n;i++){
        printf("%d ",hC[i]);
    }

    column<<<1,n>>>(d_a,d_b,d_c,m,n);
    cudaMemcpy(hC,d_c,m*n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("\nColumn C: \n");
    for(int i=0;i<m*n;i++){
        printf("%d ",hC[i]);
    }
    dim3 dimGrid (1,1,1);
    dim3 dimBlock (m,n,1);
    element<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,m,n);
    cudaMemcpy(hC,d_c,m*n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("\nElement C: \n");
    for(int i=0;i<m*n;i++){
        printf("%d ",hC[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
