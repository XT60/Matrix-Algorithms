import time
import numpy as np
import matplotlib.pyplot as plt
import os.path


MAX_WAIT_TIME = 50 # in seconds
TESTING_MATRIX_SIZES = [2**i for i in range(2, 16)]

class Analytics: 
    def __init__(self, size:int, time:float, flops: int) -> None:
        self.size = size
        self.time = time
        self.flops = flops

def getRandomMatrix(size: int, low: float = 0.00000001, high: float = 1.0):
    return np.random.uniform(low, high, size=(size, size))

def getAnalyticsArr(timedFunction) -> list[Analytics]:
    analyticsArr = []

    for size in TESTING_MATRIX_SIZES:
        startTime = time.time()

        flops = timedFunction(size)
        
        deltaTime = time.time() - startTime

        analytics = Analytics(size, deltaTime, flops)

        analyticsArr.append(analytics)

        print("Finished calculation:")
        print("size:", size)
        print("time:", deltaTime)
        print("------------------------------------------------------------------")

        if (deltaTime > MAX_WAIT_TIME):
            break

    return analyticsArr

def plotAnalytics(analyticsArr, outputDir):
    # Extract data for plotting
    sizes = [analytics.size for analytics in analyticsArr]
    times = [analytics.time for analytics in analyticsArr]
    flops = [analytics.flops for analytics in analyticsArr]

    # Plotting Time vs Size
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times, marker='o')
    plt.title('Time vs Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join(outputDir, 'time_vs_size.png'))  # Save the figure as an image
    plt.clf()

    # Plotting Flops vs Size
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, flops, marker='o', color='r')
    plt.title('Flops vs Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Flops')
    plt.grid(True)
    plt.savefig(os.path.join(outputDir, 'flops_vs_size.png'))  # Save the figure as an image