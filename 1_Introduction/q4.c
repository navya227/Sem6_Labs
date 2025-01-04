#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
	int rank,size;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char word[] = "HELLO";
	word[rank]+=32;

	printf("Rank is %d. Word is: %s \n",rank,word);	

	MPI_Finalize();
	return 0;
}