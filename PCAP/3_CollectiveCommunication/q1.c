#include <mpi.h>
#include <stdio.h>

int fact(int n) {
    if (n == 0) {
        return 1;
    }
    return n * fact(n - 1);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n,a[size],fin[size],sum = 0;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            scanf("%d", &a[i]);
        }
    }

    MPI_Scatter(a, 1, MPI_INT, &n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int ans = fact(n);

    MPI_Gather(&ans, 1, MPI_INT, fin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            sum += fin[i];
        }
        printf("%d\n", sum);
    }

    MPI_Finalize();
    return 0;
}
