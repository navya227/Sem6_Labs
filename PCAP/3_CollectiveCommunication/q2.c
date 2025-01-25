#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m;
    int a[100], fin[100];

    if (rank == 0) {
        printf("Enter m \n");
        scanf("%d", &m);

        printf("Enter values \n");
        for (int i = 0; i < m * size; i++) {
            scanf("%d", &a[i]);
        }
    }

    int b[m], sum = 0;
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(a, m, MPI_INT, b, m, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < m; i++) {
        sum += b[i];
    }

    int ans = sum / m;
    printf("Avg = %d, Rank = %d \n",ans,rank);

    MPI_Gather(&ans, 1, MPI_INT, fin, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_sum = 0;
        for (int i = 0; i < size; i++) {
            total_sum += fin[i];
        }
        printf("Net Avg = %d\n", total_sum / size);
    }

    MPI_Finalize();
    return 0;
}
