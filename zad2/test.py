from inverse import InverseCalculationEngine, checkInverseResult
from analytics import getRandomMatrix
from LU import LUCalculationEngine
from det import DetCalculationEngine
import numpy as np

engine = DetCalculationEngine()

m = np.array([
    [2, 5],
    [1, 0]
])
print(m)

print(engine.det(m), sep = "\n")
