# Author | Eduardo Ortega
import matplotlib.pyplot as plt
import numpy as np
import math, os, time
from scipy.integrate import quad
from scipy.special import erfinv, erf
BOUNDARY = 1/math.e
c=1/(((math.pi**(1/2))*(1+erf(1/(2**(1/2))))) / (2**(1/2)))
d=c*((math.pi**(1/2))/(2**(1/2)))*erf(math.log(1/math.e)/(2**(1/2)))

def pdf(x, bound):
    if x < bound: return 0.0
    first = 1/x
    second = (-1.0 * (math.log(x) ** 2)) / 2.0
    second = math.e ** (second)
    return c * (first * second)

def cdf_num(x, bound):
    if x < bound: return 0.0
    value = (erf(math.log(x)/(2**(1/2)))*(math.pi**(1/2)))/(2**(1/2))
    value *= c
    value -= d
    return value

def cdf_int(t, bound):
    I = quad(pdf, -1*np.inf, t, args=(bound))
    return I[0]  


def invcdf_function_EEE554(u):
    if u <= 0.0: return 0.0
    inside = ((2**(1/2))*(u-d))/(c*(math.pi**(1/2)))
    x = math.e ** ((2*(1/2))*erfinv(inside))
    return x
    
def cdf_invcdf_plot():
    x = np.linspace(-10, 10, 10000)
    u = np.linspace(0.0, 1.0, 10000)
    y_cdf = [cdf_function_EEE554(t, BOUNDARY) for t in x]
    x_invcdf = [invcdf_function_EEE554(s) for s in u]
    plt.plot(x, y_cdf, c='b', label="CDF")
    plt.plot(x_invcdf, u, c='r', label="INVERSE CDF")
    plt.title("Given distribution CDF and its estimate\n(Through INVERSE CDF)", fontweight='bold')
    plt.ylabel("Probability", fontweight='bold')
    plt.xlabel("x of X", fontweight='bold')
    plt.xlim([0.0, 10.0])
    plt.legend(fancybox=True, shadow=True)
    plt.show()

def cdf_pdf_plot():
    x = np.linspace(-10, 10, 10000)
    y_cdf_num = [cdf_num(t, BOUNDARY) for t in x]
    y_cdf_int = [cdf_int(t, BOUNDARY) for t in x]
    y_pdf = [pdf(t, BOUNDARY) for t in x]
    plt.plot(x, y_cdf_num, c='b', label="CDF - NUMERICAL")
    plt.plot(x, y_cdf_int, c='g', label="CDF - INTEGRAL")
    plt.plot(x, y_pdf, c='r', label="PDF")
    plt.title("Given Distribution PDF and CDF(s)", fontweight='bold')
    plt.ylabel("Probability", fontweight='bold')
    plt.xlabel("x of X", fontweight='bold')
    plt.legend(fancybox=True, shadow=True)
    plt.show()
    
def compute_integral():
    I = quad(pdf, BOUNDARY, np.inf, args=(BOUNDARY))
    return I

def main():
    cdf_pdf_plot()

if __name__=="__main__":
    main()



