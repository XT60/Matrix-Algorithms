from inverse import InverseCalculationEngine
from det import DetCalculationEngine
from LU import LUCalculationEngine
from analytics import * 

# inverse
# def inverseTimedFunction(size): 
#     A = getRandomMatrix(size)
#     engine = InverseCalculationEngine()
#     engine.inverse(A)
#     return engine.getFlops()

# analyticsArr = getAnalyticsArr(inverseTimedFunction)
# plotAnalytics(analyticsArr, "./img/inverse")

# # lu
# def inverseTimedFunction(size): 
#     A = getRandomMatrix(size)
#     engine = LUCalculationEngine()
#     engine.LU(A)
#     return engine.getFlops()

# analyticsArr = getAnalyticsArr(inverseTimedFunction)
# plotAnalytics(analyticsArr, "./img/lu")

# det
def inverseTimedFunction(size): 
    A = getRandomMatrix(size)
    engine = DetCalculationEngine()
    engine.det(A)
    return engine.getFlops()

analyticsArr = getAnalyticsArr(inverseTimedFunction)
plotAnalytics(analyticsArr, "./img/det")

