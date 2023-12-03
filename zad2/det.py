import numpy as np
from LU import LUCalculationEngine

tolerance = 0.00001

class DetCalculationEngine:
    __flops = 0
    lu = LUCalculationEngine()

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops

    def det(self, A: np.ndarray):
        self.resetCounter()
        self.lu.resetCounter()

        L, U = self.lu.LU(A)
        res = np.prod(np.diagonal(U))
        
        self.__flops += self.lu.getFlops() + A.shape[0] - 1
        
        return res

def checkDetResult(initial_matrix: np.ndarray, det):
    proper_det = np.linalg.det(initial_matrix)
    return np.abs(proper_det, det) < tolerance