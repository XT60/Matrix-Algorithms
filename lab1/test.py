# binet
from binet import BinetCalculationEngine 
import numpy as np  

A = np.array([[1,3], [2,4]])
B = np.array([[-5,8], [3,9]])

matrix = BinetCalculationEngine().multiplyMatrices(A, B)
print(matrix)

