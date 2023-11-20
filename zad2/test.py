from inverse import InverseCalculationEngine, checkInverseResult
from analytics import getRandomMatrix

engine = InverseCalculationEngine()

matrix = getRandomMatrix(4)
inverted = engine.inverse(matrix)

print(matrix)
print(inverted)
print(checkInverseResult(matrix, inverted), engine.getFlops())