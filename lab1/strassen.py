import numpy as np
import math

class StrassenCalculationEngine:
    __flops = 0 #  tracker for number of floating point operations

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops
    
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
    
    def __splitMatrix(self, M: np.ndarray):
        n = M.shape[0] // 2
        return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]

    def __defineAuxiliaryMatrices(self, A: np.ndarray, B: np.ndarray):

        A11, A12, A21, A22 = self.__splitMatrix(A)
        B11, B12, B21, B22 = self.__splitMatrix(B)

        M1 = self.multiplyMatrices(A11 + A22, B11 + B22)
        M2 = self.multiplyMatrices(A21 + A22, B11)
        M3 = self.multiplyMatrices(A11, B12 - B22)
        M4 = self.multiplyMatrices(A22, B21 - B11)
        M5 = self.multiplyMatrices(A11 + A12, B22)
        M6 = self.multiplyMatrices(A21 - A11, B11 + B12)
        M7 = self.multiplyMatrices(A12 - A22, B21 + B22)


        return M1, M2, M3, M4, M5, M6, M7

    def __recursiveMultiplyMatrices(self, A: np.ndarray, B: np.ndarray):

        if A.shape == (1, 1):
            #self.__flops += 1 
            return A * B
        
        M1, M2, M3, M4, M5, M6, M7 = self.__defineAuxiliaryMatrices(A, B)
        
        C1 = M1 + M4 - M5 + M7
        C2 = M3 + M5
        C3 = M2 + M4
        C4 = M1 - M2 + M3 + M6


        return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4))))
    
    def multiplyMatrices(self, A: np.ndarray, B: np.ndarray):

        original_rows_A, original_cols_A = A.shape
        original_rows_B, original_cols_B = B.shape

        if original_cols_A != original_rows_B:
            raise ValueError("Number of columns in matrix A must be equal to the number of rows in matrix B.")

        
        if A.shape != B.shape:
            rows, cols = A.shape
            if rows != cols or not (rows & (rows - 1) == 0):
                A = self.__adjustMatrixSize(A.copy())
                B = self.__adjustMatrixSize(B.copy())

        C = self.__recursiveMultiplyMatrices(A, B)

        return self.__removeExtraZeros(C, original_rows_A, original_cols_B)
    
    
    
