import numpy as np
import math

class BinetCalculationEngine:
    __flops = 0 #  tracker for number of floating point operations

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops

    # splits matrix into 4 matrices of n/2 size each
    def __splitMatrix(self, M: np.ndarray):
        n = M.shape[0] // 2
        return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]

    # multiplies matrices using recursive binet method 
    def multiplyMatrices(self, A: np.ndarray, B: np.ndarray):
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

    
