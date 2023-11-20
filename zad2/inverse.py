import numpy as np
import math
from utils import split_matrix
from strassen import StrassenCalculationEngine

tolerance = 0.00001

class InverseCalculationEngine:
    __flops = 0
    strassen = StrassenCalculationEngine()

    def resetCounter(self):
        self.__flops = 0

    def getFlops(self):
        return self.__flops

    def inverse(self, A: np.ndarray):
        # reseting counter
        self.resetCounter()
        self.strassen.resetCounter()
        
        # calculations
        result = self.__inverseInner(A)

        # updating counter
        self.__flops += self.strassen.getFlops()

        return result

    def __inverseInner(self, A: np.ndarray):
        if A.shape == (1, 1):
            x = 1 / A[0, 0]
            self.__flops += 1
            return np.array([[x]])

        A11, A12, A21, A22 = split_matrix(A)

        # A11_inv = inverse(A11)
        A11_inv= self.__inverseInner(A11)

        # S22 = A22 - A21 @ A11_inv @ A12
        S22 = self.strassen.multiplyMatrices(A21, A11_inv)
        S22 = self.strassen.multiplyMatrices(S22, A12)
        S22 = A22 - S22
        self.__flops += math.prod(A.shape)

        # S22_inv = inverse(S22)
        S22_inv = self.__inverseInner(S22)

        # B11 = A11_inv @ (I + A12 @ S22_inv @ A21 @ A11_inv)
        I = np.eye(A11.shape[0])
        B11 = self.strassen.multiplyMatrices(A12, S22_inv)
        B11 = self.strassen.multiplyMatrices(B11, A21)
        B11 = self.strassen.multiplyMatrices(B11, A11_inv)
        B11 = I + B11
        self.__flops += A11.shape[0]
        B11 = self.strassen.multiplyMatrices(A11_inv, B11)

        # B12 = -1 * (A11_inv @ A12 @ S22_inv)
        B12 = self.strassen.multiplyMatrices(A11_inv, A12)
        B12 = self.strassen.multiplyMatrices(B12, S22_inv)
        B12 = -1 * B12
        self.__flops = math.prod(B12.shape)

        # B21 = -1 * (S22_inv @ A21 @ A11_inv)
        B21 = self.strassen.multiplyMatrices(S22_inv, A21)
        B21 = self.strassen.multiplyMatrices(B21, A11_inv)
        B21 = -1 * B21
        self.__flops = math.prod(B21.shape)

        return np.vstack((np.hstack((B11, B12)), np.hstack((B21, S22_inv))))


def checkInverseResult(initial_matrix, inverted_matrix):
    size = initial_matrix.shape[0]
    check = np.abs(initial_matrix @ inverted_matrix - np.eye(size))
    return np.all(check < tolerance)