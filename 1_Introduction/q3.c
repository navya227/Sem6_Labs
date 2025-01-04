#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
	int rank,size;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int x = 3, y = 2;
	switch(rank){
		case 0:
	        printf("Sum = %d \n",x+y);
	        break;
		case 1:
	        printf("Difference = %d \n",x-y);
	        break;

	    case 2:
	        printf("Product = %d \n",x*y);
	        break;

	    case 3:
	        printf("Division = %d \n", x/y);
	        break;
    }	

	MPI_Finalize();
	return 0;
}