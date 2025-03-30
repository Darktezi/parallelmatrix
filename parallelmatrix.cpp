#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

class Matrix {
private:
    std::vector<std::vector<int>> data;
    size_t rows, cols;

public:
    Matrix(size_t rows, size_t cols, bool randomize = true) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<int>(cols, 0));
        if (randomize) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(1, 10);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    data[i][j] = dist(gen);
        }
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    int at(size_t i, size_t j) const { return data[i][j]; }
    void set(size_t i, size_t j, int value) { data[i][j] = value; }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");

        Matrix result(rows, other.cols, false);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                int temp = data[i][k];
                for (size_t j = 0; j < other.cols; ++j) {
                    result.data[i][j] += temp * other.data[k][j];
                }
            }
        }
        return result;
    }

    void print() const {
        for (const auto& row : data) {
            for (int val : row)
                std::cout << val << " ";
            std::cout << "\n";
        }
    }

    void ReadData(const std::string& filename) {
        std::ifstream fin(filename);
        if (!fin) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        if (!(fin >> rows >> cols)) {
            std::cerr << "Error reading matrix dimensions from file: " << filename << std::endl;
            return;
        }

        data.assign(rows, std::vector<int>(cols));
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                fin >> data[i][j];
    }

    void WriteData(const std::string& filename) const {
        std::ofstream fout(filename);
        if (!fout) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        fout << rows << " " << cols << std::endl;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                fout << std::fixed << std::setprecision(4) << data[i][j] << ' ';
            }
            fout << '\n';
        }
    }
};

int main() {
    for (size_t size = 100; size <= 1000; size += 100) {
        Matrix A(size, size);
        Matrix B(size, size);

        std::string filename_A = "matrix_A_" + std::to_string(size) + ".txt";
        std::string filename_B = "matrix_B_" + std::to_string(size) + ".txt";
        std::string filename_C = "matrix_C_" + std::to_string(size) + ".txt";

        A.WriteData(filename_A);
        B.WriteData(filename_B);

        double total_time = 0.0;
        int iterations = 10;
        Matrix C(size, size, false);

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            C = A * B;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        C.WriteData(filename_C);
        std::cout << "Size: " << size << " Average multiplication time: " << (total_time / iterations) << " seconds" << std::endl;
    }
    return 0;
}
