import common

import numpy as np
import matplotlib.pyplot as plt

# Fix random seed
common.seed(0)

####################################################################
# Figure 1.1                                                       #
####################################################################

def f(x):
    """
    Define the function f(x) = sin(5x)^2.
    
    Args:
    x (float): The input value.
    
    Returns:
    float: The value of f(x).
    """
    return np.sin(5*x)**2

for plot_no in range(2):
    nb_cols = 10
    if plot_no == 1:       
        # colors = plt.cm.Greys(np.linspace(0, 1, nb_cols))
        colors = np.full(nb_cols, common.histogram_cols[0])
    else:
        colors = plt.cm.YlOrRd(np.linspace(0, 1, nb_cols))
        

    common.plot_figure(book_scale = 0.7)

    zs = np.arange(0, 1, 0.001)
    plt.plot(zs, f(zs), 'r-', linewidth=2, label="h(x)")
    x = np.arange(0, 1 + 0.1, 0.1)

    for j in range(10):   
        plt.fill_between(x[j:j+2], 0, f(x[j:j+2]), color=colors[j], edgecolor=common.black)
    plt.plot(zs, f(zs), 'k-', linewidth=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$h(x)$')
    plt.ylim(0, 1)
    
    if plot_no == 1:        
        common.save_figure("fig1-1-trapezoid-greyscale.pdf")
    else:
        common.save_figure("fig1-1-trapezoid.pdf")
        
    

####################################################################
# Figure 1.2                                                       #
####################################################################

# Control Variates

def f(x):
    return np.sin(x)

def f1(x):
    return np.sin(x) - x

def f2(x):
    return np.sin(x) + (x**2 - 1)/2 - np.pi*x/2

z = np.random.normal(size=50)

common.plot_figure(1, 3)
plt.subplot(1, 3, 1)
zs = np.arange(-2, 2, 0.05)
plt.plot(zs, f(zs), 'k-', linewidth=2, zorder=2)
plt.xlabel(r'$x$')
plt.scatter(z, f(z), facecolors='none', edgecolors=common.scatter_cols[0], zorder=3)
plt.xlim(-2.1, 2.1)
plt.ylim(-1.1, 3.1)

plt.subplot(1, 3, 2)
plt.plot(zs, f1(zs), 'k-', linewidth=2, zorder=2)
plt.xlabel(r'$x$')
plt.scatter(z, f1(z), facecolors='none', edgecolors=common.scatter_cols[0], marker='o', zorder=3)
plt.xlim(-2.1, 2.1)
plt.ylim(-1.1, 3.1)

plt.subplot(1, 3, 3)
plt.plot(zs, f2(zs), 'k-', linewidth=2, zorder=2)
plt.xlabel(r'$x$')
plt.scatter(z, f2(z), facecolors='none', edgecolors=common.scatter_cols[0], marker='o', zorder=3)
plt.xlim(-2.1, 2.1)
plt.ylim(-1.1, 3.1)

common.save_figure("fig1-2-cv.pdf")

####################################################################
# Figure 1.6                                                       #
####################################################################

# OU plot

def simOU(Tend, npts, sigma, b, m=0, x0=0):
    delta = Tend / (npts - 1)
    ts = np.linspace(0, Tend, npts)
    xs = np.zeros(npts)
    xs[0] = x0
    for i in range(1, npts):
        xs[i] = xs[i-1] - b**2 / (2*sigma**2) * (xs[i-1] - m) * delta + b * np.random.normal() * np.sqrt(delta)
    return ts, xs

common.seed(12345808)
a1_ts, a1_xs = simOU(10, 1000, 1, 3, m=4, x0=2)
a2_ts, a2_xs = simOU(10, 1000, 1, 1, m=0)
a3_ts, a3_xs = simOU(10, 1000, 1, 1/3, m=-4, x0=-2)

all_xs = np.concatenate((a1_xs, a2_xs, a3_xs))
lo, hi = np.min(all_xs), np.max(all_xs)

common.plot_figure(book_scale = 0.7)
plt.plot(a1_ts, a1_xs, '-', linewidth=2, label="a1", c=common.line_plot_cols[0])
plt.plot(a2_ts, a2_xs, '-', linewidth=2, label="a2", c=common.line_plot_cols[1])
plt.plot(a3_ts, a3_xs, '-', linewidth=2, label="a3", c=common.line_plot_cols[2])
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.ylim(lo*1.05, hi*1.05)
common.save_figure("fig1-6-OU.pdf")

####################################################################
# Figure 1.7                                                       #
####################################################################

## Code for the additional figure: Fig 1.7

def fn(x):
    return(1/(1+(x)**2))

def approx(xs):   
    nx = len(xs)
    K = np.zeros((nx, nx))
    
    for i in range(nx):
       K[i,:] = np.exp(-(xs[i] - xs)**2)
    
    fs = fn(xs)

    inv_K = np.linalg.inv(K)
    denom = np.sqrt(fs.T @ inv_K @ fs)
    gs = inv_K @ fs /denom
    
    return(gs)


def apprfn(gs, xpts, xs): 
    n = len(xs)
    app = np.zeros(n)
    for i in range(len(xpts)):       
        app = app + gs[i] * np.exp(-(xpts[i]-xs)**2)
    
    return(app)

linestyles = [":", "-.", "--", (0, (3, 1, 1, 1)),  (5, (10, 3)), (0, (3, 5, 1, 5, 1, 5))]
lwd = 2

def plot_approx(xpts, lab, approx_no):
    gs = approx(xpts)
    apprf = apprfn(gs, xpts, xs)
    plt.plot(xs, apprf, linewidth = lwd, label = lab, c = common.line_plot_cols[approx_no + 1], linestyle = linestyles[approx_no])
 
xs = np.linspace(-4, 4, 400)
truef = fn(xs)

common.plot_figure(book_scale = 0.7)
plt.plot(xs, truef, label = "T", linewidth=lwd + 1, c = common.line_plot_cols[0], linestyle = '-')
 
plot_approx(np.array([-3]), "A", 0)
plot_approx(np.array([-3, -2, -1]), "B", 1)
plot_approx(np.array([-3,-2,-1,0,1]), "C", 2)
plot_approx(np.array([-3,-2,-1,0,1,2,3]), "D", 3)

plt.legend(loc='upper right')
plt.xlabel(r'$x$')
plt.ylabel('function')
plt.ylim(-0.01, 1.01)
common.save_figure("fig1-7-kernelApprox.pdf")
