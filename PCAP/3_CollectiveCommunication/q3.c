#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>


int is_vowel(char c) {
    c = tolower(c);  
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int non_vow(char *str, int length) {
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (!is_vowel(str[i])) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char str[100];
    int len, chunk;
    
    if (rank == 0) {
        printf("Enter length \n");
        scanf("%d", &len);
        
        printf("Enter the string: ");
        scanf("%s", str);
        
        chunk = len / size; 
    }

    MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char sub[chunk];

    MPI_Scatter(str, chunk, MPI_CHAR, sub, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    int c = non_vow(sub, chunk);
    printf("Rank = %d non-vowels = %d \n",rank,c);

    int tot = 0;
    MPI_Reduce(&c, &tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total non-vowels in the string: %d\n", tot);
    }

    MPI_Finalize();
    return 0;
}
