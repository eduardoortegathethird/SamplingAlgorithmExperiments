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
    if t < bound: return 0
    I = quad(pdf, bound, t, args=(bound))
    return I[0]  

def invcdf_num(u):
    if u <= 0.0: return 0.0
    inside = ((2**(1/2))*(u+d))/(c*(math.pi**(1/2)))
    x = math.e ** ((2*(1/2))*erfinv(inside))
    return x

def invcdf_bis(u, tolerance):
    if u == 0.0: return 0.0
    x_min, x_max = 0.0, 100.0
    while x_max - x_min > tolerance:
        bisect = (x_min + x_max) / 2.0
        u_est = cdf_int(bisect, BOUNDARY)
        if u_est > u:
            x_max = bisect
        else:
            x_min = bisect
    return bisect
    
def cdf_invcdf_plot():
    x = np.linspace(-10, 10, 100)
    u = np.linspace(0.0, 1.0, 100)
    y_cdf_num = [cdf_num(t, BOUNDARY) for t in x]
    y_cdf_int = [cdf_int(t, BOUNDARY) for t in x]
    x_invcdf_num = [invcdf_num(s) for s in u]
    x_invcdf_bis = [invcdf_bis(s, 0.001) for s in u]
    plt.scatter(x, y_cdf_num, c='b', 
                label="CDF - NUMERICAL")
    plt.scatter(x, y_cdf_int, c='g', 
                label="CDF - INTEGRAL")
    plt.scatter(x_invcdf_num, u, c='r', 
                label="INVERSE CDF - OPEN FORM SOLUTION")
    plt.scatter(x_invcdf_bis, u, c='m', 
                label="INVERSE CDF - BISECTION")
    plt.title("Given distribution CDF(s) and its estimate\n"
                "(Through INVERSE CDF(s))", fontweight='bold')
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

def task_6(ns=1000):
    u = np.linspace(0.0, 1.0, ns)
    x = np.linspace(0.0, 100.0, ns)
    y_pdf = [pdf(t, BOUNDARY) for t in x]
    t = time.time()
    x_invcdf_bis = [invcdf_bis(s, 0.001) for s in u]
    t_fin = time.time() - t
    print(f"time for baseline algorithm: {t_fin}s")
    weight = np.ones_like(x_invcdf_bis) / float(len(x_invcdf_bis))
    plt.hist(x_invcdf_bis, weights=weight, bins=40, label='PDF est')
    plt.plot(x, y_pdf, label='PDF')
    plt.title("Baseline Sampling", fontweight='bold')
    plt.legend(fancybox=True, shadow=True)
    plt.ylim([0.0, 1.0])
    plt.ylabel("Probability", fontweight='bold')
    plt.xlabel("x of X", fontweight='bold')
    plt.show()

def a(x_k, x_k_1, y_k, y_k_1):
    area = (x_k_1 - x_k) * (y_k - y_k_1)
    return area

def find_coor_for_suff_area(area, x, y, tolerance=0.001):
    coor_not_found = True
    x_max = 100000000000.0
    x_min = x
    print(area, x_max, x_min)
    while (x_max - x_min) > tolerance:
        x_guess = (x_max + x_min) / 2.0
        y_guess = pdf(x_guess, BOUNDARY)
        area_guess = a(x, x_guess, y, y_guess)
        AREA_BOOL = area_guess < area
        if not AREA_BOOL:
            x_max = x_guess
        else:
            x_min = x_guess
    print(area_guess)
    return x_guess, y_guess


def ziggurat(n, area_guess, tol=0.01):
    x_k = BOUNDARY
    y_k = pdf(x_k, BOUNDARY)
    area = []
    x, y = [x_k], [y_k]
    for k in range(n-1):
        print(f"Computing Zigg Level: {k}")
        x_k_1, y_k_1 = find_coor_for_suff_area(area_guess, x_k, y_k)
        a_zigg = a(x_k, x_k_1, y_k, y_k_1)
        area.append(a_zigg)
        x.append(x_k_1)
        y.append(y_k_1)
        x_k, y_k = x_k_1, y_k_1
    x_1, x_n = x[0], x[-1]
    y_n = y[-1]
    a_zigg = (x_n - x_1) * y_n
    a_quad = quad(pdf, x_n, np.inf, args=(BOUNDARY))[0]
    a_n = a_zigg + a_quad
    a_n_lower_bound = a_n > (area_guess-tol)
    a_n_upper_bound = a_n < (area_guess+tol)
    if a_n_lower_bound and a_n_upper_bound:
        area.append(a_n)
        return area, x, y
    elif a_n_lower_bound and not a_n_upper_bound:
        # choice of A is too small, guess again but make it bigger
        return ziggurat(n, area_guess+(tol/2), tol=tol)
    else:
        # choice of A is too big, guess again but make it smaller
        return ziggurat(n, area_guess-(tol/2), tol=tol)
       
def task_7():
    n_ag = [(4, 1/4), (32, 0.0049), (256, 0.000073)]
    for args in n_ag:
        n, ag = args[0], args[1]
        a, x, y = ziggurat(n, ag, tol=0.1*ag)
        print(f"Number of ZIGGS | {n}")
        print("==========================")
        for el in range(len(a)):
            print(f"ZIGG LEVEL {el} AREA{a[el]}")

def main():
    task_7()

if __name__=="__main__":
    main()



