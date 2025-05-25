#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <string>

void writeMatrix(const std::string &fname, const std::vector<int> &M, int rows, int cols) {
    std::ofstream fout(fname);
    fout << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            fout << M[i * cols + j] << " ";
        fout << "\n";
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    const int iterations = 10;
    // Перебираем размеры матриц
    for (int N = 100; N <= 1000; N += 100) {
        int total_rows = N;
        int total_cols = N;

        std::vector<int> A, B;
        if (rank == 0) {
            A.resize(N * N);
            B.resize(N * N);
            std::mt19937 gen(42);
            std::uniform_int_distribution<int> dist(1, 10);
            for (int i = 0; i < N*N; ++i) {
                A[i] = dist(gen);
                B[i] = dist(gen);
            }
            writeMatrix("matrix_A_" + std::to_string(N) + ".txt", A, N, N);
            writeMatrix("matrix_B_" + std::to_string(N) + ".txt", B, N, N);
        }
        // Раздача B всем
        if (rank != 0) B.resize(N * N);
        MPI_Bcast(B.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);

        // Распределение A по строкам
        int base_rows = N / comm_size;
        int rem = N % comm_size;
        int local_rows = base_rows + (rank < rem ? 1 : 0);
        int offset = rank * base_rows + std::min(rank, rem);
        std::vector<int> localA(local_rows * N);

        std::vector<int> counts(comm_size), displs(comm_size);
        for (int i = 0; i < comm_size; ++i) {
            int rows_i = base_rows + (i < rem ? 1 : 0);
            counts[i] = rows_i * N;
            displs[i] = (i * base_rows + std::min(i, rem)) * N;
        }
        MPI_Scatterv(rank==0?A.data():nullptr, counts.data(), displs.data(), MPI_INT,
                     localA.data(), counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

        // Локальное умножение
        std::vector<int> localC(local_rows * N);
        double total_time = 0.0;
        for (int it = 0; it < iterations; ++it) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int i = 0; i < local_rows; ++i) {
                for (int j = 0; j < N; ++j) {
                    int sum = 0;
                    for (int k = 0; k < N; ++k) sum += localA[i*N + k] * B[k*N + j];
                    localC[i*N + j] = sum;
                }
            }
            double t1 = MPI_Wtime();
            total_time += (t1 - t0);
        }

        // Сбор результата
        std::vector<int> C;
        if (rank == 0) C.resize(N * N);
        MPI_Gatherv(localC.data(), counts[rank], MPI_INT,
                    C.data(), counts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            writeMatrix("matrix_C_" + std::to_string(N) + ".txt", C, N, N);
            std::cout << "Size: " << N
                      << " Average MPI multiplication time: "
                      << (total_time / iterations) << " seconds" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
