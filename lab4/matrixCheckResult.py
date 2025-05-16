import numpy as np

def load_matrix(filename):
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        data = np.loadtxt(f, dtype=np.float64)
    return data.reshape(rows, cols)

def verify_multiplication(size):
    file_A = f"lab4/results/matrix_A_{size}.txt"
    file_B = f"lab4/results/matrix_B_{size}.txt"
    file_C = f"lab4/results/matrix_C_{size}.txt"
    
    A = load_matrix(file_A)
    B = load_matrix(file_B)
    C_expected = load_matrix(file_C)
    
    C_computed = np.dot(A, B)
    
    if np.allclose(C_computed, C_expected, atol=1e-4):
        print(f"Matrix multiplication is correct for size {size}.")
    else:
        print(f"Mismatch found in multiplication for size {size}.")
        diff = np.abs(C_computed - C_expected)
        print(f"Max difference: {np.max(diff)}")

if __name__ == "__main__":
    for size in range(100, 1100, 100):
        verify_multiplication(size)
