# binet
from binet import BinetCalculationEngine
from strassen import StrassenCalculationEngine
import numpy as np  

A = np.array([[1,3,2], [2,4,2]])
B = np.array([[-5,8], [3,9],[3, 7]])

matrix = BinetCalculationEngine().multiplyMatrices(A, B)
print(matrix)

matrix2 = StrassenCalculationEngine().multiplyMatrices(A, B)
print(matrix2)