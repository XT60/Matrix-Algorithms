# binet
from binet import BinetCalculationEngine 
from strassen import StrassenCalculationEngine
from analytics import * 

analyticsArr = getAnalyticsArr(BinetCalculationEngine)
plotAnalytics(analyticsArr, "./lab1/img/binet")

analyticsArr = getAnalyticsArr(StrassenCalculationEngine)
plotAnalytics(analyticsArr, "./lab1/img/strassen")