import numpy as np
import math


class BinetCalculationEngine():
    __flops = 0

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops
    
    def __splitMatrix(self, M: np.ndarray):
        n = M.shape[0] // 2
        return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]
    
    def __adjustMatrixSize(self, M: np.ndarray):
        rows, cols = M.shape

        new_rows = 2**math.ceil(math.log2(rows))
        new_cols = 2**math.ceil(math.log2(cols))

        new_dim = max(new_cols, new_rows)
        if rows < new_rows or cols < new_cols:
            zero_rows = new_dim - rows
            zero_cols = new_dim - cols
            M = np.pad(M, ((0, zero_rows), (0, zero_cols)), mode='constant', constant_values=0)

        return M
    
    def __removeExtraZeros(self, C: np.ndarray, original_rows: int, original_cols: int):
        return C[:original_rows, :original_cols]
    
    def __recursiveMultiplyMatrices(self, A: np.ndarray, B: np.ndarray):
        if A.shape == (1, 1):
            self.__flops += 1 # flops for the multiplication
            return A * B

        A11, A12, A21, A22 = self.__splitMatrix(A)
        B11, B12, B21, B22 = self.__splitMatrix(B)

        C1 = self.multiplyMatrices(A11, B11) + self.multiplyMatrices(A12, B21)
        C2 = self.multiplyMatrices(A11, B12) + self.multiplyMatrices(A12, B22)
        C3 = self.multiplyMatrices(A21, B11) + self.multiplyMatrices(A22, B21)
        C4 = self.multiplyMatrices(A21, B12) + self.multiplyMatrices(A22, B22)

        self.__flops += 4 * math.prod(A.shape) # flops for the sums

        return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4))))
    

    def multiplyMatrices(self, A: np.ndarray, B: np.ndarray):

        original_rows_A, original_cols_A = A.shape
        original_rows_B, original_cols_B = B.shape

        if original_cols_A != original_rows_B:
            raise ValueError("Number of columns in matrix A must be equal to the number of rows in matrix B.")

        if A.shape != B.shape or not (original_rows_A & (original_rows_A - 1) == 0):     # check if the matrices are 2^n x 2^n size to avoid additional calculations
                A = self.__adjustMatrixSize(A.copy())
                B = self.__adjustMatrixSize(B.copy())

        C = self.__recursiveMultiplyMatrices(A, B)

        return self.__removeExtraZeros(C, original_rows_A, original_cols_B)
    # multiplies matrices using recursive binet method 


    
