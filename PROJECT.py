# Author | Eduardo Ortega
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
import math, os, time
from scipy.integrate import quad
from scipy.special import erfinv, erf
BOUNDARY = 1/math.e
c=1/(((math.pi**(1/2))*(1+erf(1/(2**(1/2))))) / (2**(1/2)))
d=c*((math.pi**(1/2))/(2**(1/2)))*erf(math.log(1/math.e)/(2**(1/2)))

def pdf(x, bound, const):
    if x < bound: return 0.0
    first = 1/x
    second = (-1.0 * (math.log(x) ** 2)) / 2.0
    second = math.e ** (second)
    return const * (first * second)

def cdf_num(x, bound):
    if x < bound: return 0.0
    value = (erf(math.log(x)/(2**(1/2)))*(math.pi**(1/2)))/(2**(1/2))
    value *= c
    value -= d
    return value

def cdf_int(t, bound):
    if t < bound: return 0
    I = quad(pdf, bound, t, args=(bound, c))
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
    y_pdf = [pdf(t, BOUNDARY, c) for t in x]
    plt.plot(x, y_cdf_num, c='b', label="CDF - NUMERICAL")
    plt.plot(x, y_cdf_int, c='g', label="CDF - INTEGRAL")
    plt.plot(x, y_pdf, c='r', label="PDF")
    plt.title("Given Distribution PDF and CDF(s)", fontweight='bold')
    plt.ylabel("Probability", fontweight='bold')
    plt.xlabel("x of X", fontweight='bold')
    plt.legend(fancybox=True, shadow=True)
    plt.show()
    
def compute_integral():
    I = quad(pdf, BOUNDARY, np.inf, args=(BOUNDARY, c))
    return I

def task_6(ns=1000):
    u = np.linspace(0.00000000001, 1.0, ns)
    x = np.linspace(0.0, 100.0, ns)
    y_pdf = [pdf(t, BOUNDARY, c) for t in x]
    t = time.time()
    x_invcdf_bis = [invcdf_bis(s, 0.001) for s in u]
    t_fin = time.time() - t
    print(f"time for baseline algorithm: {t_fin}s")
    weight = np.ones_like(x_invcdf_bis) / float(len(x_invcdf_bis))
    plt.hist(x_invcdf_bis, weights=weight, bins=55, label='PDF est')
    plt.plot(x, y_pdf, label='PDF')
    plt.title("Baseline Sampling", fontweight='bold')
    plt.legend(fancybox=True, shadow=True)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 10.0])
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
    while (x_max - x_min) > tolerance:
        x_guess = (x_max + x_min) / 2.0
        y_guess = pdf(x_guess, BOUNDARY, 1.0)
        area_guess = a(x, x_guess, y, y_guess)
        AREA_BOOL = area_guess < area
        if not AREA_BOOL:
            x_max = x_guess
        else:
            x_min = x_guess
    return x_guess, y_guess


def ziggurat(n, area_guess, tol):
    x_k = BOUNDARY
    y_k = pdf(x_k, BOUNDARY, 1.0)
    area = []
    x, y = [x_k], [y_k]
    for k in range(n-1):
        x_k_1, y_k_1 = find_coor_for_suff_area(area_guess, x_k, y_k)
        a_zigg = a(x_k, x_k_1, y_k, y_k_1)
        area.append(a_zigg)
        x.append(x_k_1)
        y.append(y_k_1)
        x_k, y_k = x_k_1, y_k_1
    x_1, x_n = x[0], x[-1]
    y_n = y[-1]
    a_zigg = (x_n - x_1) * y_n
    a_quad = quad(pdf, x_n, np.inf, args=(BOUNDARY, 1))[0]
    a_n = a_zigg + a_quad
    a_n_lower_bound = a_n > (area_guess-tol)
    a_n_upper_bound = a_n < (area_guess+tol)
    if a_n_lower_bound and a_n_upper_bound:
        area.append(a_n)
        return area, x, y
    elif a_n_lower_bound and not a_n_upper_bound:
        # choice of A is too small, guess again but make it bigger
        return ziggurat(n, area_guess+(tol/2), tol)
    else:
        # choice of A is too big, guess again but make it smaller
        return ziggurat(n, area_guess-(tol/2), tol)
       
def task_7():
    n_ag = [(4, 0.5, 0.01), (32, 0.012, 0.09), (256, 0.0001, 0.09)]
    task_7_dict = {}
    x_y = {}
    for args in n_ag:
        n, ag, tol = args[0], args[1], args[2]
        a, x, y = ziggurat(n, ag, tol*ag)
        ave_a = np.average(a)
        var_a = np.var(a)
        x_y[n] = (x, y)
        x_n = x[-1]
        a_n = a[-1]
        y_n = y[-1]
        x_1 = x[0]
        task_7_dict[n] = [ave_a, var_a, x_n, a_n, y_n, x_1]
    return task_7_dict, x_y

def generate_c_prime():
    zigg, _ = task_7()
    c_prime = {}
    for n in zigg.keys():
        x_n = zigg[n][2]
        x_1 = zigg[n][5]
        area = quad(pdf, x_n, np.inf, args=(BOUNDARY, 1.0))[0]
        c_val = 1/area
        a_n = zigg[n][3]
        y_n = zigg[n][4]
        a_rect = (x_n-x_1)*y_n
        p = a_rect / a_n
        c_prime[n] = (c_val, x_n, p, y_n)
    return c_prime, _

def plot_pdf_pdfT():
    data, _ = generate_c_prime()
    fig, ax = plt.subplots(3)
    for num, n in enumerate([4, 32, 256]):
        c_prime = data[n][0]
        x_n = data[n][1]
        x = np.linspace(x_n, x_n+10, 1000)
        y_pdf = [pdf(t, BOUNDARY, c) for t in x]
        y_pdfT = [pdf(t, x_n, c_prime) for t in x]
        y_gx = [pdf(t, BOUNDARY, 1.0) for t in x]
        ax[num].plot(x, y_pdf, c='r', label="PDF -> cg(x)")
        ax[num].plot(x, y_pdfT, c='b', label="PDF -> c'g(x)")
        ax[num].plot(x, y_gx, c='g', label="g(x)")
        ax[num].set_ylabel("Probability", fontweight='bold')
        ax[num].set_xlabel("x of X", fontweight='bold')
        ax[num].set_title(f"n={n}", fontweight='bold')
        ax[num].set_ylim([10**(-7), 10**(0)])
        ax[num].set_yscale('log')
        if num == 2: ax[num].legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), shadow=True, fancybox=True)
    plt.suptitle("nth Region of Ziggurat (Tail)", fontweight='bold')
    plt.show()

def generate_M_pareto(x_n):
    area = quad(pdf_pareto, x_n, np.inf, args=(x_n, 1.0, 1.0))[0]
    m = 1/area
    return m

def pdf_pareto(x, boundary, m, alpha):
    if x < boundary: return 0.0
    top = alpha * (boundary ** alpha)
    bottom = x ** (alpha+1)
    p = top / bottom
    return p

def cdf_pareto(x, bound, n, M):
    if x < bound: return 0.0
    u = M * (1-(bound/x))
    return u

def invcdf_pareto(u, x_n, M):
    if u <= 0: return 0.0
    x_min, x_max = x_n, x_n+100.0
    while x_max - x_min > 0.001:
        bisect = (x_max+x_min) / 2.0
        u_est = cdf_pareto(bisect, x_n, M, 1.0)
        if u_est > u:
            x_max = bisect
        else:
            x_min = bisect
    return bisect


def plot_pdf_pareto(cdf=False):
    data, _ = generate_c_prime()
    fig, ax = plt.subplots(3)
    A = 1.0
    for num, n in enumerate([4, 32, 256]):
        c_prime = data[n][0]
        x_n = data[n][1]
        y_n = data[n][3]
        M = generate_M_pareto(x_n)
        y_n_pareto = pdf_pareto(x_n, x_n, M, A)
        u = np.linspace(0.00000000001, y_n_pareto, 1000)
        x = np.linspace(x_n, x_n+10, 1000)
        
        y_pdf = [pdf(t, BOUNDARY, c) for t in x]
        y_pdf_par = [pdf_pareto(t, x_n, M, A) for t in x]
        y_gx = [pdf(t, BOUNDARY, 1.0) for t in x]
        y_cdf_par = [cdf_pareto(t, x_n, n, M) for t in x]
        x_invcdf_par = [invcdf_pareto(s, x_n, M) for s in u]
        if not cdf:
            w = np.ones_like(x_invcdf_par) / float(len(x_invcdf_par))
            ax[num].hist(x_invcdf_par, weights=w, bins=30, label="EST PDF Pareto")
            ax[num].plot(x, y_pdf, c='r', label="PDF -> cg(x)")
            ax[num].plot(x, y_pdf_par, c='b', label=f"PDF Pareto")
            ax[num].plot(x, y_gx, c='g', label="g(x)")
        else:
            ax[num].plot(x, y_cdf_par, c='m', label="CDF Pareto")
            ax[num].scatter(x_invcdf_par, u, c='y', s=3, label="INV CDF Pareto")
        
        ax[num].set_ylabel("Probability", fontweight='bold')
        ax[num].set_xlabel("x of X", fontweight='bold')
        ax[num].set_title(f"n={n}", fontweight='bold')
        #ax[num].set_ylim([10**(-7), 10**(0)])
        #ax[num].set_yscale('log')

        if num == 2: ax[num].legend(loc='lower center', bbox_to_anchor=(0.5, -1.0), shadow=True, fancybox=True)
    plt.suptitle("nth Region of Ziggurat (Tail)", fontweight='bold')
    plt.show()

def return_x_y_and_p():
    p_info, x_y_info = generate_c_prime()
    new_dict = {}
    for n in p_info.keys():
        p = p_info[n][2]
        x = x_y_info[n][0]
        y = [pdf(t, BOUNDARY, c) for t in x]
        new_dict[n] = (x, y, p)
    return new_dict

def plot_ziggurat_outline():
    data = return_x_y_and_p()
    x = np.linspace(0.0, 100.0, 10000)
    y = [pdf(t, BOUNDARY, c) for t in x]
    fig, ax = plt.subplots(1, 3)
    for col, n in enumerate(data.keys()):
        x_zr = data[n][0]
        y_zr = data[n][1]
        ax[col].plot(x, y, label="PDF")
        ax[col].set_title(f"n={n}", fontweight='bold')
        x_1 = x_zr[0]
        ax[col].set_xlim([0.0, 10.0])
        ax[col].set_ylim([0.0, 1.0])
        for idx in range(1, len(x_zr)):
            xk1 = x_zr[idx]
            xk = x_zr[idx-1]
            yk1 = y_zr[idx]
            yk = y_zr[idx-1]
            anchor = (x_1, yk1)
            y_h = abs(yk1 - yk)
            x_w = xk1 - x_1
            rect = pat.Rectangle(anchor, x_w, y_h, 
                                linewidth=1, 
                                edgecolor='r', 
                                facecolor='none')
            ax[col].add_patch(rect)
    plt.suptitle("PDF & Outline of Ziggurat Regions", fontweight='bold')
    plt.show()

def sample_ziggurat(ns=1000000):
    data = return_x_y_and_p()
    x_accept = {}
    for n in data.keys():
        x_accept[n] = []
        x_zigg = data[n][0]
        y_zigg = data[n][1]
        p = data[n][2]
        x_1 = x_zigg[0]
        x_n = x_zigg[-1]
        y_n = y_zigg[-1]
        i = 0
        k_list = []
        k_accept = []
        t = time.time()
        occ = {'a':0, 'b':0, 'c':0, 'd':0, 'e':0}
        for sample in range(ns):
            i += 1
            k = np.random.randint(1, n+1)
            k_list.append(k)
            if k < n:
                xk, xk1 = x_zigg[k-1], x_zigg[k]
                yk, yk1 = y_zigg[k-1], y_zigg[k]
                x = np.random.uniform(low=x_1, high=xk1)
                if x < xk:
                    occ['a'] += 1
                    k_accept.append(k)
                    x_accept[n].append(x)
                else:
                    y = np.random.uniform(low=yk1, high=yk)
                    if y < pdf(x, BOUNDARY, c):
                        occ['b'] += 1
                        k_accept.append(k)
                        x_accept[n].append(x)
                    else:
                        occ['c'] += 1
            else:
                p_x = np.random.uniform(low=0.0, high=1.0)
                x = np.random.uniform(x_1, x_n)
                if p_x <= p: 
                    occ['d'] += 1
                    k_accept.append(k)
                    x_accept[n].append(x)
                else:
                    occ['e'] += 1
                    u = np.random.uniform(low=0.0, high=y_n)
                    M = generate_M_pareto(x_n)
                    x = invcdf_pareto(u, x_n, M)
                    y = pdf_pareto(x, x_n, M, 1.0)
                    y_sample = np.random.uniform(low=0.0, high=y)
                    if y_sample < pdf(x, BOUNDARY, c):
                        k_accept.append(k)
                        x_accept[n].append(x)
        t2 = time.time()
        print(f"n: {n}\taccept ratio: {len(x_accept[n])/ns}")
        print(f"time elapsed: {t2 - t}")
        print(f"time per sample: {(t2-t) / ns}")
        print(f"RATIO FOR EACH OCCURANCE SCENARIO")
        for key in occ.keys():
            value = occ[key]
            print(f"{key}: {value/ns}")
        '''
        k_accepts = {i:k_accept.count(i) for i in k_accept}
        k_counts = {i:k_list.count(i) for i in k_list}
        for n in k_counts.keys():
            k_count = k_counts[n]
            k_acc = k_accepts[n]
            print(f"{n}: {k_acc/k_count}")
        quit()
        '''
    return x_accept
                    

def plot_x_PDF(x):
    n = list(x.keys())
    for col, key in enumerate([4]):
        if key == 32: bn = 50
        if key == 4: bn = 10
        if key == 256: bn = 150
        data = x[key]
        #w = np.ones(data) / float(len(data))
        plt.hist(data, density=True, bins=bn, label=f"EST PDF ({key})")
        x = np.linspace(0.0, 5.0, 1000000)
        y_pdf = [pdf(t, BOUNDARY, c) for t in x]
        plt.plot(x, y_pdf, label=f"PDF")
        plt.ylabel("Prob.", fontweight='bold')
        plt.xlabel("x of X", fontweight='bold')
        plt.legend(fancybox=True, shadow=True)
        plt.xlim([0.0, 5.0])
        plt.ylim([0.0, 1.0])
        plt.title("PDF and EST PDF from Ziggurat", fontweight='bold')
        plt.show()
        plt.clf()
        


def main():
    np.random.seed(seed=3)
    x_accept = sample_ziggurat()
    #plot_x_PDF(x_accept)

if __name__=="__main__":
    main()



