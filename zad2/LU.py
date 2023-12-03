import numpy as np
import math
from utils import split_matrix
from strassen import StrassenCalculationEngine
from inverse import InverseCalculationEngine


class LUCalculationEngine:
    __flops = 0
    strassen = StrassenCalculationEngine()
    inverse = InverseCalculationEngine()

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops

    def LU(self, A: np.ndarray):
        # resetting counter
        self.resetCounter()
        self.strassen.resetCounter()

        # calculations
        result = self.__LUInner(A)

        # updating counter
        self.__flops += self.strassen.getFlops()
        self.__flops += self.inverse.getFlops()

        return result

    def __LUInner(self, A: np.ndarray):
        if A.shape == (1, 1):
            return np.array([[1]]), A

        A11, A12, A21, A22 = split_matrix(A)

        L11, U11 = self.__LUInner(A11)
        L11_inv = self.inverse.inverse(L11)
        U11_inv = self.inverse.inverse(U11)

        L21 = self.strassen.multiplyMatrices(A21, U11_inv)
        U12 = self.strassen.multiplyMatrices(L11_inv, A12)

        S = self.strassen.multiplyMatrices(A21, U11_inv)
        S = self.strassen.multiplyMatrices(S, L11_inv)
        S = self.strassen.multiplyMatrices(S, A12)
        S = A22 - S
        
        self.__flops += S.shape[0] ** 2 # one operation for every element on square matrix

        L22, U22 = self.__LUInner(S)

        return (
            np.vstack((np.hstack((L11, np.zeros_like(A12))), np.hstack((L21, L22)))),
            np.vstack((np.hstack((U11, U12)), np.hstack((np.zeros_like(A21), U22))))
        )
