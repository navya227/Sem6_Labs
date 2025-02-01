#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[3][3],n,b[3],count=0;
    if(rank == 0){
        printf("Enter elements\n");
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                scanf("%d",&arr[i][j]);
            }
        }

        printf("Enter element to search \n");
        scanf("%d",&n);
    }
    
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(arr,3,MPI_INT,b,3,MPI_INT,0,MPI_COMM_WORLD);

    for(int i=0;i<3;i++){
        if(b[i]==n){
            count++;
        }
    }
    int tot = 0;
    MPI_Reduce(&count,&tot,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    if(rank == 0){
        printf("Instances found = %d \n",tot);
    }

    MPI_Finalize();
    return 0;
}
