#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include "cuda_runtime.h"

// CUDA-ядро умножения матриц
__global__ void matMulKernel(const int* A, const int* B, int* C, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    for (size_t size = 100; size <= 1000; size += 100) {
        size_t N = size;
        size_t bytes = N * N * sizeof(int);

        // Хостовая память
        std::vector<int> h_A(N * N), h_B(N * N), h_C(N * N);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 10);
        for (size_t i = 0; i < N * N; ++i) {
            h_A[i] = dist(gen);
            h_B[i] = dist(gen);
        }

        // Устройство: выделение памяти
        int *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
        checkCudaError(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
        checkCudaError(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

        // Копирование данных на устройство
        checkCudaError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "Memcpy A");
        checkCudaError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "Memcpy B");

        // Настройка параметров ядра
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        // Прогрев ядра
        matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        // Измерение времени
        double total_time = 0.0;
        int iterations = 10;
        for (int iter = 0; iter < iterations; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
        }

        // Копирование результата на хост
        checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy C");

        // Функция записи матрицы в файл
        auto writeMatrix = [&](const std::string& fname, const std::vector<int>& M) {
            std::ofstream fout(fname);
            fout << N << " " << N << "\n";
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j)
                    fout << M[i * N + j] << " ";
                fout << "\n";
            }
        };

        // Запись файлов
        writeMatrix("matrix_A_" + std::to_string(N) + ".txt", h_A);
        writeMatrix("matrix_B_" + std::to_string(N) + ".txt", h_B);
        writeMatrix("matrix_C_" + std::to_string(N) + ".txt", h_C);

        std::cout << "Size: " << N
                  << " Average CUDA multiplication time: "
                  << (total_time / iterations) << " milliseconds\n";

        // Освобождение памяти устройства
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
