# Author | Eduardo Ortega
import matplotlib.pyplot as plt
import numpy as np
import math, os, time
BOUNDARY = 1/math.e

def function_EEE554(x, bound=None):
    if x < bound: return 0.0
    first = 1/x
    second = (-1.0 * (math.log(x) ** 2)) / 2.0
    second = math.e ** (second)
    return first * second

def main():
    x = np.linspace(-10, 10, 10000)
    y = [function_EEE554(t, bound=BOUNDARY) for t in x]
    plt.plot(x, y)
    plt.title("Given distribution: g(x) with respect to the boundary (1/e)")
    plt.show()
    

if __name__=="__main__":
    main()
