#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_DIM 4
#define HIDDEN_DIM 10
#define OUTPUT_DIM 3

/* ReLU activation function */
double relu(double x) {
    return x > 0 ? x : 0;
}

/* Reads the weights from file "weights.txt" into the provided arrays. */
void read_weights(const char* filename, double W1[INPUT_DIM][HIDDEN_DIM], double b1[HIDDEN_DIM],
                  double W2[HIDDEN_DIM][OUTPUT_DIM], double b2[OUTPUT_DIM]) {
    FILE *fp;
    char layer[10];
    int r, c;
    int i, j;
    fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open weight file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Read W1 dimensions and values */
    fscanf(fp, "%s %d %d", layer, &r, &c);
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            fscanf(fp, "%lf", &W1[i][j]);
        }
    }
    /* Read b1 */
    fscanf(fp, "%s %d", layer, &r);
    for (i = 0; i < r; i++) {
        fscanf(fp, "%lf", &b1[i]);
    }
    /* Read W2 dimensions and values */
    fscanf(fp, "%s %d %d", layer, &r, &c);
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            fscanf(fp, "%lf", &W2[i][j]);
        }
    }
    /* Read b2 */
    fscanf(fp, "%s %d", layer, &r);
    for (i = 0; i < r; i++) {
        fscanf(fp, "%lf", &b2[i]);
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    int rank, size;
    int i, j, k;
    int start, end, start2, end2;
    int total_rows, total_rows2;
    int rows_per_proc, remainder;
    int rows_per_proc2, remainder2;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double W1[INPUT_DIM][HIDDEN_DIM];
    double b1[HIDDEN_DIM];
    double W2[HIDDEN_DIM][OUTPUT_DIM];
    double b2[OUTPUT_DIM];

    /* Rank 0 reads weights from file */
    if (rank == 0) {
        read_weights("weights.txt", W1, b1, W2, b2);
    }

    /* Broadcast weights to all processes */
    MPI_Bcast(W1, INPUT_DIM * HIDDEN_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b1, HIDDEN_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(W2, HIDDEN_DIM * OUTPUT_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b2, OUTPUT_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Get input sample from command line arguments.
       Expected usage: mpirun -np <nprocs> ./parallel <f1> <f2> <f3> <f4>
    */
    double x[INPUT_DIM];
    if (rank == 0) {
        if (argc != INPUT_DIM + 1) {
            fprintf(stderr, "Usage: %s <f1> <f2> <f3> <f4>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (i = 0; i < INPUT_DIM; i++) {
            x[i] = atof(argv[i+1]);
        }
        printf("Input sample from argv: %f, %f, %f, %f\n", x[0], x[1], x[2], x[3]);
    }
    MPI_Bcast(x, INPUT_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Synchronize before starting the inference timing */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    /***********************
     * First Layer Computation:
     * Compute z1 = x * W1 in parallel by dividing the rows of W1.
     ***********************/
    total_rows = INPUT_DIM;  /* Rows of W1 correspond to the input dimension */
    rows_per_proc = total_rows / size;
    remainder = total_rows % size;
    if (rank < remainder) {
        start = rank * (rows_per_proc + 1);
        end = start + (rows_per_proc + 1);
    } else {
        start = rank * rows_per_proc + remainder;
        end = start + rows_per_proc;
    }
    if (start >= total_rows) {
        start = 0;
        end = 0;
    }
    double local_z1[HIDDEN_DIM];
    for (j = 0; j < HIDDEN_DIM; j++) {
        local_z1[j] = 0.0;
    }
    for (i = start; i < end; i++) {
        for (j = 0; j < HIDDEN_DIM; j++) {
            local_z1[j] += x[i] * W1[i][j];
        }
    }
    double z1[HIDDEN_DIM];
    MPI_Reduce(local_z1, z1, HIDDEN_DIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double hidden[HIDDEN_DIM];
    if (rank == 0) {
        for (j = 0; j < HIDDEN_DIM; j++) {
            z1[j] += b1[j];
            hidden[j] = relu(z1[j]);
        }
    }
    MPI_Bcast(hidden, HIDDEN_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /***********************
     * Second Layer Computation:
     * Compute out = hidden * W2 in parallel by dividing the rows of W2.
     ***********************/
    total_rows2 = HIDDEN_DIM;  /* Rows of W2 correspond to hidden neurons */
    rows_per_proc2 = total_rows2 / size;
    remainder2 = total_rows2 % size;
    if (rank < remainder2) {
        start2 = rank * (rows_per_proc2 + 1);
        end2 = start2 + (rows_per_proc2 + 1);
    } else {
        start2 = rank * rows_per_proc2 + remainder2;
        end2 = start2 + rows_per_proc2;
    }
    if (start2 >= total_rows2) {
        start2 = 0;
        end2 = 0;
    }
    double local_out[OUTPUT_DIM];
    for (k = 0; k < OUTPUT_DIM; k++) {
        local_out[k] = 0.0;
    }
    for (j = start2; j < end2; j++) {
        for (k = 0; k < OUTPUT_DIM; k++) {
            local_out[k] += hidden[j] * W2[j][k];
        }
    }
    double output[OUTPUT_DIM];
    MPI_Reduce(local_out, output, OUTPUT_DIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Synchronize and end timing after inference computations */
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        for (k = 0; k < OUTPUT_DIM; k++) {
            output[k] += b2[k];
        }
        int pred = 0;
        double max_val = output[0];
        for (k = 1; k < OUTPUT_DIM; k++) {
            if (output[k] > max_val) {
                max_val = output[k];
                pred = k;
            }
        }
        printf("Predicted label: %d\n", pred);
        printf("Inference time: %f seconds\n", end_time - start_time);
        {
            FILE *fp;
            fp = fopen("result_single.txt", "w");
            if (fp) {
                fprintf(fp, "Predicted label: %d\n", pred);
                fprintf(fp, "Inference time: %f seconds\n", end_time - start_time);
                fclose(fp);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
