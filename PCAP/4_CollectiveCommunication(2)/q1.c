#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void handle_mpi_error(int errcode) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string;
    MPI_Error_string(errcode, error_string, &length_of_error_string);
    fprintf(stderr, "MPI Error: %s\n", error_string);
    MPI_Abort(MPI_COMM_WORLD, errcode);
}

int main(int argc, char* argv[]) {
    int rank, size;
    int errcode;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


    int n = rank + 1;
    int fact;
    errcode = MPI_Scan(&n, &fact, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    if (errcode != MPI_SUCCESS) {
        handle_mpi_error(errcode);
    }

    int tot = 0;
    MPI_Reduce(&fact, &tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Sum of factorials from 1! to %d! is: %d\n", size, tot);
    }

    MPI_Finalize();
    return 0;
}
