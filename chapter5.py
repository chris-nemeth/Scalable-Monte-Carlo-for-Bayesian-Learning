import common

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numdifftools as nd

####################################################################
# Figure 5.1                                                       #
####################################################################

## compare Gustafson to ZZ on 1-d
sig=0.1
mu=2

##log-targe
def logpi(x):
    return np.log(0.5 * norm.pdf(x, mu, 1) + 0.5 * norm.pdf(x, 0, sig))
  
#derivative of log-pi
def grad_logpi(x):
    p = norm.pdf(x, mu, 1)/(norm.pdf(x, mu, 1) + norm.pdf(x, 0, sig))
    return -p * (x - mu) - (1 - p) * x/sig**2
   
x0 = 0
p0 = 1
h = 0.1
state_G = [x0, p0]
state_C = [x0, p0]

common.seed(1)

N = 1000
x_G = np.empty(N)
for i in range(N):
    U = np.random.uniform()  ##common uniform for accept/reject
    ##Gustafson
    z = np.random.normal()
    x_new = state_G[0] + h * np.abs(z) * state_G[1]
    if U < np.exp(logpi(x_new) - logpi(state_G[0])):
        state_G[0] = x_new
    else:
        state_G[1] = -state_G[1]
        
    x_G[i] = state_G[0]

##continous-time sampler
x_C = x0
t_C = 0
t = 0
x = x0
p = p0
H = max(1, 1 / sig**2)
while t < h * N:
    ##bounding rate is a+H(t)
    a = -grad_logpi(x) * p
    U = np.random.exponential()
    if a > 0:
        t_inc = -a / H + np.sqrt(a**2 + 2 * H * U) / H
    else:
        t_inc = np.abs(a / H) + np.sqrt(2 * H * U) / H
        
    x = x + p * t_inc
    t = t + t_inc
    if np.random.uniform() < (-grad_logpi(x) * p) / (a + H * t_inc):
        p = -p
        x_C = np.append(x_C, x)
        t_C = np.append(t_C, t)
        
x_C = np.append(x_C, x)
t_C = np.append(t_C, t)

common.plot_figure(1, 2)
plt.subplot(1, 2, 1)
plt.plot(range(1, N + 1), x_G)
plt.xlabel(r"Iteration ($k$)")
plt.ylabel(r"$x_k$")

plt.subplot(1, 2, 2)
plt.plot(t_C, x_C)
plt.xlabel(r"Time ($t$)")
plt.ylabel(r"$x_t$")
common.save_figure("fig5-1-1dcomparison.pdf")

####################################################################
# Figure 5.7                                                       #
####################################################################

############# PLOT FOR 1D RJ-PDMP LIMIT

###simulate next event for rate (a+bt)^+
def rlinearevent(a, b):
    if a <= 0 and b <= 0:
        return float("inf")
    else:
        U = np.random.exponential()
        if a <= 0 and b > 0:
            return -(a / b) + np.sqrt(2 * U / b)
        else:
            if a > 0 and b == 0:
                return U / a
            else:
                if a > 0 and b <= 0:
                    if U > -a**2 / (2 * b):
                        return float("inf")
                    else:
                        return -a / b + np.sqrt(a**2 + 2 * U * b) / b
                else:
                    return -a / b + np.sqrt(a**2 + 2 * U * b) / b

###simulate PDMP form 0.5N(0,1)+0.5N(0,sigma^2) for time T
### assumes sigma< 1 for bound on rate
def onedmixturePDMP(sigma, T):
    theta = np.random.normal()
    p = 1
    t = 0
    theta_st = [t, theta]
    while t < T:
        if theta * p < 0: ##move to mode before next event possible
            t = t + np.abs(theta)
            theta = 0
        pg = np.exp(-0.5 * theta**2) / (np.exp(-0.5 * theta**2) + (1 / sigma) * np.exp(-0.5 * theta**2 / sigma**2))
        a = p * (pg * theta + (1 - pg) * (theta / sigma**2)) ## current rate at theta
        b = pg + (1 - pg) / sigma**2 ##bound on increas in rate as pg is decreasing
        dt = rlinearevent(a, b)
        t = t + dt
        theta = theta + p * dt
        pg = np.exp(-0.5 * theta**2) / (np.exp(-0.5 * theta**2) + (1 / sigma) * np.exp(-0.5 * theta**2 / sigma**2))
        rate = p * (pg * theta + (1 - pg) * (theta / sigma**2)) ## current rate at theta
        if np.random.uniform() < rate / (a + b * dt):  ## event
            p = -p
            theta_st = np.vstack((theta_st, [t, theta]))
            
    return theta_st

def plot_fig5_7_subplot(plot_no, sigma):
    plt.subplot(1, 3, plot_no)
    out = onedmixturePDMP(sigma, 50)
    plt.plot(out[:, 0], out[:, 1])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\theta_t$")
    plt.ylim(-3.5, 3.5)
  
common.plot_figure(1, 3)
plot_fig5_7_subplot(1, 0.1)
plot_fig5_7_subplot(2, 0.01)
plot_fig5_7_subplot(3, 0.001)
common.save_figure("fig5-7-rjpdmp-1d.pdf")

#########CODE FOR DIFFERENT SAMPLERS#################
#BPS algorithm

###one iteration of BPS based on Algorithm in book
def BPS_it(theta, p, Q, lref):
    p = np.array(p)
    
    a = np.matmul(np.matmul(p.T, Q), theta)
    b = np.matmul(np.matmul(p.T, Q), p)
    w1 = np.random.exponential(1)
    w2 = np.random.exponential(1)
    
    if lref != 0:
        tau1 = w1 / lref ##refresh event time
    else:
        tau1 = np.inf
    
    #time of bounce event
    if a < 0:
        tau2 = np.sqrt(2 * w2 / b) + np.abs(a) / b
    else:
        tau2 = -a / b + np.sqrt(a**2 +  2 * w2 * b) / b
    
    ##event time and update position
    t = min(tau1, tau2)
    theta = theta + p * t
    
    ##update velocity
    if tau1 <= t:
        p = np.random.normal(size=len(p))
    else:
        g = np.matmul(Q, theta)
        g = g / np.sqrt(np.sum(g**2))
        p = p - 2 * np.sum(p * g) * g
    
    return {'t': t, 'theta': theta, 'p': p}

##ZZ algorithm
def ZZ_it(theta, p, Q, lref=None):
   
    d = len(p) #number of dimensions
    tau = np.zeros(d)
    
    #pre-calculation for event rate
    A = np.matmul(Q, theta)
    B = np.matmul(Q, p)
    for i in range(d): ##simulate event time for ith flip
        a = p[i] * A[i]
        b = p[i] * B[i]
        w = np.random.exponential(1)
        if b == 0:
            if a > 0:
                tau[i] = w / a
            else:
                tau[i] = np.inf
        else:
            if a < 0 and b > 0:
                tau[i] = np.abs(a) / b + np.sqrt(2 * w / b)
            else:
                if a > 0 and a**2 + 2 * w * b > 0:
                    tau[i] = -a / b + np.sqrt(a**2 + 2 * w * b) / b
                else:
                    tau[i] = np.inf
    
    ##event time and update position
    t = np.min(tau)
    theta = theta + p * t
    
    #type of event
    i = np.argmin(tau)
    
    #flip ith component of p
    p[i] = -p[i]
    return {'t': t, 'theta': theta, 'p': p}

###Co-ordinate sampler
def CS_it(theta, p, Q, lref=None):
    
    a = np.matmul(np.matmul(p.T, Q), theta)
    b = np.matmul(np.matmul(p.T, Q), p)
    w1 = np.random.exponential(1)
    w2 = np.random.exponential(1)
    
    if lref != 0:
        tau1 = w1 / lref ##refresh event time
    else:
        tau1 = np.inf
        
    #time of non-refresh event
    if a < 0:
        tau2 = np.sqrt(2 * w2 / b) + np.abs(a) / b
    else:
        tau2 = -a / b + np.sqrt(a**2 + 2 * w2 * b) / b
    
    ##event time and update position
    t = min(tau1, tau2)
    theta = theta + p * t
    d = len(p)
    
    ##update velocity
    if tau1 <= t:
        #refresh
        j = np.random.randint(1, d)
        p = np.zeros(d)
        p[j-1] = -1 + 2 * (np.random.uniform() > 0.5)
    else:
        A = np.matmul(Q, theta)
        probs = np.abs(A)        
        j = random.choices(range(1, d+1), weights = probs, k = 1)[0]
        p = np.zeros(d)
        p[j-1] = -np.sign(A[j-1])
    
    return {'t': t, 'theta': theta, 'p': p}

#HMC
##run an exact HMC, with N iterations of M steps of length h between updates
def HMC(h, N, M, Q, theta):
    d = len(theta)
    
    ##calculate the dynamics over h
    A = np.zeros((2*d, 2*d))
    for i in range(d):
        A[i, d+i] = 1
        
    for i in range(d):
        for j in range(d):
            A[d+i, j] = -Q[i, j]
    
    B = expm(h * A)
    theta_st = np.zeros((N*M, d))
    k = 0
    for i in range(N):
        p = np.random.normal(size=d)
        z = np.append(theta, p)
        for j in range(M):
            z = np.matmul(B, z)
            theta = z[:d]
            theta_st[k, :] = theta
            k += 1
    
    return theta_st

##algorithm -- input Q, refresh rate lref, initial state theta, p and time Tmax
## no error checking on input
def PDMP(Tmax, Q, theta, p, lref, event_sim):
    ##storage for skeleton (event times and positions)
    t_ev = [0]
    theta_ev = [theta]
    s = 0 ##current time
    while s < Tmax:
        next_event = event_sim(theta, p, Q, lref) #simulate next event
        theta = next_event['theta']
        p = next_event['p']
        s += next_event['t']
        
        #update storage
        t_ev.append(s)
        theta_ev = np.vstack([theta_ev, theta])
    
    return {'t': t_ev, 'theta': theta_ev}

##algorithm -- input Q, refresh rate lref, initial state theta, p and number of events N.ev
## no error checking on input
##stores p.theta;|theta| and theta_1
def PDMP_ev(N_ev, Q, theta, p, lref, event_sim):
    ##storage for skeleton (event times and positions)
    t_ev = np.zeros(N_ev+1)
    theta_ev = np.zeros((N_ev+1, 3))
    theta_ev[0, :] = [np.sum(p * theta), np.sqrt(np.sum(theta**2)), theta[0]]
    s = 0 ##next event
    n = 1
    while n <= N_ev:
        next_event = event_sim(theta, p, Q, lref) #simulate next event
        theta = next_event['theta']
        p = next_event['p']
        s += next_event['t']
        
        #update storage
        t_ev[n] = s
        theta_ev[n, :] = [np.sum(p * theta), np.sqrt(np.sum(theta**2)), theta[0]]
        n += 1
    
    return {'t': t_ev, 'theta': theta_ev}

##calculate log pi
def logpi(theta, Q):
    theta = np.array(theta)
    return -0.5 * np.matmul(np.matmul(theta.T, Q), theta)

####################################################################
# Figure 5.2                                                       #
####################################################################

####EXAMPLE Bivariate Gaussian

##FIGURE -- BPS REFRESH

sigma1 = 1
sigma2 = 1
rho = 0.0
Q = np.array([[1/sigma1**2, -rho/(sigma1*sigma2)], [-rho/(sigma1*sigma2), 1/sigma2**2]]) / (1 - rho**2)

theta0 = np.array([2, 0])
p0 = np.array([0, 1])

out_BPS0 = PDMP(100, Q, theta0, p0, 0, BPS_it)
out_BPS0_1 = PDMP(100, Q, theta0, p0, 0.1, BPS_it)
out_BPS1 = PDMP(100, Q, theta0, p0, 1, BPS_it)
out_BPS10 = PDMP(100, Q, theta0, p0, 10, BPS_it)

t_grid = np.arange(-3, 3.05, 0.05)
z = np.zeros((len(t_grid), len(t_grid)))
for i in range(len(t_grid)):
    for j in range(len(t_grid)):
        z[i, j] = logpi([t_grid[i], t_grid[j]], Q)

##PLOT
def plot_axis(plot_no, nrow = 2, ncol = 2):
    plt.subplot(nrow, ncol, plot_no)
    plt.imshow(z, extent=[min(t_grid), max(t_grid), min(t_grid), max(t_grid)], origin="lower", cmap=common.get_heat_color_map(10))
    plt.xlabel(r"$\theta^{(1)}$")
    plt.ylabel(r"$\theta^{(2)}$")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
common.plot_figure(2, 2)

plot_axis(1)
plt.plot(out_BPS0['theta'][:, 0], out_BPS0['theta'][:, 1], color=common.heat_plot_path_col)

plot_axis(2)
plt.plot(out_BPS0_1['theta'][:, 0], out_BPS0_1['theta'][:, 1], color=common.heat_plot_path_col)

plot_axis(3)
plt.plot(out_BPS1['theta'][:, 0], out_BPS1['theta'][:, 1], color=common.heat_plot_path_col)

plot_axis(4)
plt.plot(out_BPS10['theta'][:, 0], out_BPS10['theta'][:, 1], color=common.heat_plot_path_col)

common.save_figure("fig5-2-Gaussian_BPS_refresh.pdf")

####################################################################
# Figure 5.3                                                       #
####################################################################

##FIGURE 3 COMPARE BPS ZZ CS AND HMC

lref = 0
sigma1 = 1
sigma2 = 1
rho = 0.3
Q = np.array([[1 / sigma1**2, -rho / (sigma1 * sigma2)], [-rho / (sigma1 * sigma2), 1 / sigma2**2]]) / (1 - rho**2)
theta0 = np.array([2, 0])
p0 = np.array([0, 1])

common.seed(1)
out_BPS = PDMP(100, Q, theta0, p0, 1, BPS_it)
p0_zz = np.full(len(theta0), 1)
out_ZZ = PDMP(200, Q, theta0, p0_zz, lref, ZZ_it)
p0_cs = np.array([1, 0])
out_CS = PDMP(400, Q, theta0, p0_cs, lref, CS_it)

t_grid = np.arange(-3, 3.05, 0.05)
z = np.zeros((len(t_grid), len(t_grid)))
for i in range(len(t_grid)):
    for j in range(len(t_grid)):
        z[i, j] = logpi(np.array([t_grid[i], t_grid[j]]), Q)

K = 50
L = 10
M = 30
out_HMC = HMC(0.1, L, M, Q, theta0)

##PLOT
common.plot_figure(2, 2)
plot_axis(1)
plt.plot(out_CS['theta'][:K, 0], out_CS['theta'][:K, 1], color=common.heat_plot_path_col)

plot_axis(2)
plt.plot(out_ZZ['theta'][:K, 0], out_ZZ['theta'][:K, 1], color=common.heat_plot_path_col)

plot_axis(3)
plt.plot(out_BPS['theta'][:K, 0], out_BPS['theta'][:K, 1], color=common.heat_plot_path_col)

plot_axis(4)
plt.plot(out_HMC[:, 0], out_HMC[:, 1], color="lightgrey")
plt.scatter(out_HMC[(np.arange(1, L+1) * M) - 1, 0], out_HMC[(np.arange(1, L+1) * M) - 1, 1], c=common.black, s=5, zorder=3)

common.save_figure("fig5-3-Gaussian_PDMPs.pdf")

####################################################################
# Figure 5.5                                                       #
####################################################################

common.plot_figure(1, 2, pad = 3.5)
plt.subplot(1, 2, 1)
t_grid = np.arange(0, 1, 0.01)
fu = lambda t: (t - 0.8)**2 + 0.5 * np.exp(t) - 2
fn = lambda t: t**3 - np.exp(2 * t) + 4.5 * t
fndash = lambda t: 3 * t**2 - 2 * np.exp(2 * t) + 4.5
plt.plot(t_grid, fu(t_grid), color=common.black, linewidth=2)
t = np.array([0, 0.5, 1])
plt.plot(t, fu(t), color=common.blue, linewidth=2)
plt.scatter(t, fu(t), color=common.blue, marker="x")
plt.xlabel(r"$t$")
plt.ylabel(r"$f_u$")

plt.subplot(1, 2, 2)
plt.plot(t_grid, fn(t_grid), color=common.black, linewidth=2)
plt.plot(t, fn(t), color=common.blue, linewidth=2)
plt.scatter(t, fn(t), color=common.blue, marker="x")
x = fn(t)
g = fndash(t)
for i in range(len(t)):
    plt.plot(t, x[i] + g[i] * (t - t[i]), color=common.red, linewidth=2)
plt.xlabel(r"$t$")
plt.ylabel(r"$f_u$")
plt.ylim(top=0.1)

common.save_figure("fig5-5-CC.pdf.pdf")

####################################################################
# Figure 5.4                                                       #
####################################################################

common.seed(1)
d = 20
Q = np.eye(d)
theta0 = np.random.normal(size=d)
p0 = np.random.normal(size=d)
lref = 1.5
N = 20
out_BPS = PDMP_ev(N*d, Q, theta0, p0, lref, BPS_it)
out_ZZ = PDMP_ev(N*d, Q, theta0, np.ones(d), 1, ZZ_it)

d = 1000
Q = np.eye(d)
theta0 = np.random.normal(size=d)
p0 = np.random.normal(size=d)
out_BPS2 = PDMP_ev(N*d, Q, theta0, p0, lref * np.sqrt(d/20), BPS_it)
out_BPS3 = PDMP_ev(N*d, Q, theta0, p0, lref , BPS_it)
out_ZZ2 = PDMP_ev(N*d, Q, theta0, np.ones(d), 1, ZZ_it)
   
           
##ranges for y-axis in plots
r1=(2.5,6.5)
r2=(-2,2.5)
r3=(29.5,33.5)
r4=(-3.5,3)

common.plot_figure(4, 3)
plt.subplot(4, 3, 1)
plt.plot(out_ZZ['t'] / max(out_ZZ['t']), out_ZZ['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r1)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 4)
plt.plot(out_ZZ['t'] / max(out_ZZ['t']), out_ZZ['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r2)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")

plt.subplot(4, 3, 7)
plt.plot(out_ZZ2['t'] / max(out_ZZ2['t']), out_ZZ2['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r3)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 10)
plt.plot(out_ZZ2['t'] / max(out_ZZ2['t']), out_ZZ2['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r4)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")

plt.subplot(4, 3, 2)
plt.plot(out_BPS['t'] / max(out_BPS['t']), out_BPS['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r1)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 5)
plt.plot(out_BPS['t'] / max(out_BPS['t']), out_BPS['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r2)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")

plt.subplot(4, 3, 8)
plt.plot(out_BPS2['t'] / max(out_BPS2['t']), out_BPS2['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r3)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 11)
plt.plot(out_BPS2['t'] / max(out_BPS2['t']), out_BPS2['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r4)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")

plt.subplot(4, 3, 3)
plt.plot(out_BPS['t'] / max(out_BPS['t']), out_BPS['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r1)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 6)
plt.plot(out_BPS['t'] / max(out_BPS['t']), out_BPS['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r2)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")

plt.subplot(4, 3, 9)
plt.plot(out_BPS3['t'] / max(out_BPS3['t']), out_BPS3['theta'][:, 1], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r3)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\|\theta\|$")

plt.subplot(4, 3, 12)
plt.plot(out_BPS3['t'] / max(out_BPS3['t']), out_BPS3['theta'][:, 2], color = common.line_plot_cols[0], zorder=3)
plt.ylim(r4)
plt.xlabel("Proportion of run")
plt.ylabel(r"$\theta_1$")


common.save_figure("fig5-4-Gaussian_scale.pdf")

####################################################################
# Figure 5.8                                                       #
####################################################################

##############################
## SUBSAMPLING EXAMPLE
##
##############################
  
###Boomerang Sampler
##Boomerang with global bound
##input initial state; function for evaluating gradU;
##Sigma, time to simulate and constant M for global bound
##refresh rate
##the gradU function assumes data is given by y and X; and that prior is mean 0 
## variance sigma.theta
def Boomerang(theta0, p0, gradU, Sigma, theta_star, rate_ref, T, M):
    nev = 0
    t = 0
    theta = theta0
    p = p0
    theta_st = np.hstack((t, theta.flatten(), p.flatten()))
    m = np.sqrt(np.sum(gradU(theta_star, Sigma, theta_star)**2))
    
    while t < T:
        nev += 1
        rate_bounce = (M/2) * (np.sum((theta-theta_star)**2) + np.sum(p**2)) + m * np.sqrt(np.sum((theta - theta_star)**2) + np.sum(p**2))
        t_bounce = np.random.exponential(1/rate_bounce)
        t_ref = np.random.exponential(1/rate_ref)
        if t_bounce < t_ref:
            t += t_bounce
            theta_new = theta_star + (theta - theta_star)*np.cos(t_bounce) + p*np.sin(t_bounce)
            p = p * np.cos(t_bounce) - (theta - theta_star)*np.sin(t_bounce)
            theta = theta_new
            g = gradU(theta, Sigma, theta_star)
            if np.abs(np.sum(p*g)) > rate_bounce:
                print("ERROR", theta, p)                
            if np.random.uniform() < np.sum(p*g)/rate_bounce:
                p = p - 2*np.sum(p*g) * np.matmul(Sigma, g) / np.matmul(np.matmul(g.T, Sigma), g)
                theta_st = np.vstack((theta_st, np.hstack((t, theta.flatten(), p.flatten()))))
        else:
            t += t_ref
            theta = theta_star + (theta-theta_star)*np.cos(t_ref) + p*np.sin(t_ref)
            p = np.random.multivariate_normal(mean=np.zeros(2), cov=Sigma).reshape((2,1))
            theta_st = np.vstack((theta_st, np.hstack((t, theta.flatten(), p.flatten()))))
    
    if common.VERBOSE:
        print("Number of Events", nev)
        
    return theta_st

def logpi(theta, y, X, Sigma_theta):
    y_2_columns = np.hstack((y, y))
    logl = -np.sum(np.log(1 + np.exp(np.matmul(X, theta))) - np.matmul(y_2_columns*X, theta))
    return logl - 0.5 * np.matmul(theta.T, np.matmul(np.linalg.inv(Sigma_theta), theta))

def gradU(theta, Sigma, theta_star):
    d = len(theta_star)
    gradminuspi = np.zeros((d,1))
    expXtheta = np.exp(np.matmul(X, theta))
    prob = expXtheta / (1 + expXtheta)
    A = np.linalg.inv(Sigma_theta)
    for i in range(d):
        x_col = np.array(X[:, i]).reshape((n, 1)) # necessary for element-wise multiplication to work on next line
        gradminuspi[i, 0] = np.sum(x_col * (prob-y))
        
    gradminuspi = gradminuspi + np.matmul(A, theta)
    return gradminuspi - np.matmul(np.linalg.inv(Sigma), (theta-theta_star))

#two dimensional data
common.seed(4)
n = 100 ##data points
theta_true = np.array([0.5, 1]) # parameter

X = np.column_stack((np.ones(n), np.random.normal(size=n))) #covariate

expXtheta_true = np.exp(np.matmul(X, theta_true))
p = expXtheta_true / (1 + expXtheta_true)
y = np.random.binomial(1, p, size=n)
p = p.reshape((n, 1))
y = y.reshape((n, 1))

XTX = np.matmul(X.T, X)

M = 0.25 * np.sqrt(np.max(np.linalg.eigvals(np.matmul(XTX.T, XTX)))) ###bound on Hessian

Sigma_theta = np.diag([2, 2]) ##prior on theta

##Sigma for Boomerang Sampler
Sigma = Sigma_theta
theta_star = np.array([[0], [0]])
##initial values for Boomerang Sampler
theta0 = np.random.multivariate_normal(mean=theta_star.flatten(), cov=Sigma)
p0 = np.random.multivariate_normal(mean=np.zeros(2), cov=Sigma)

out_boom1 = Boomerang(theta0.reshape((2,1)), p0.reshape((2,1)), gradU, Sigma, theta_star, rate_ref=1/4, T=20/4, M=M)

##find mode and Hessian
out = minimize(lambda theta, y, X, Sigma_theta: -logpi(theta, y, X, Sigma_theta), x0=np.zeros(2), args=(y, X, Sigma_theta), method='BFGS')
theta_star = out.x.reshape((2,1))

f_hess = lambda th: logpi(th, y, X, Sigma_theta)
Sigma_inv = -1 * nd.Hessian(f_hess)(theta_star.flatten())

Sigma = np.linalg.inv(Sigma_inv)

theta0 = (np.random.multivariate_normal(mean=theta_star.flatten(), cov=Sigma)).T
p0 = np.random.multivariate_normal(mean=np.zeros(2), cov=Sigma).T

common.seed(1)
out_boom2 = Boomerang(theta0.reshape((2,1)), p0.reshape((2,1)), gradU, Sigma, theta_star, rate_ref=2, T=40, M=M)

def boomerang_plot(out_boom, theta_star):
    ##PLOT
    #t1s = np.arange(0.2, 1.41, 0.01)
    #t2s = np.arange(-0.25, 1.51, 0.01)
    t1s = np.arange(-0.50, 1.01, 0.01)
    t2s = np.arange(-0.50, 1.01, 0.01)
    z = np.zeros((len(t1s), len(t2s)))
  
    for i in range(len(t1s)):
        for j in range(len(t2s)):
            z[i,j] = logpi(np.array([[t1s[i]], [t2s[j]]]), y, X, Sigma_theta)[0,0]
                
    z = z - np.max(z)
    plt.imshow(z, extent=[t1s[0], t1s[-1], t2s[0], t2s[-1]], aspect='auto', cmap=common.get_heat_color_map(10), origin='lower')
    k = out_boom.shape[0] - 1
    for i in range(k):
        dt = out_boom[i+1,0] - out_boom[i,0]
        N = int(dt/0.1) + 2
        h = dt/N
        t = h * np.arange(N+1)
        th = out_boom[i,1:3]
        ph = out_boom[i,3:5]
        t1_plot = theta_star[0,0] + (th[0]-theta_star[0,0])*np.cos(t) + ph[0]*np.sin(t)
        t2_plot = theta_star[1,0] + (th[1]-theta_star[1,0])*np.cos(t) + ph[1]*np.sin(t)
        plt.plot(t1_plot, t2_plot, '-', linewidth=2, color=common.heat_plot_path_col)
    plt.xlabel(r'$\theta^{(1)}$')
    plt.ylabel(r'$\theta^{(2)}$')
    plt.xlim(min(t1s), max(t1s))
    plt.ylim(min(t2s), max(t2s))
    
common.plot_figure(1, 2, pad = 3.5)
plt.subplot(1, 2, 1)
boomerang_plot(out_boom1, np.array([[0], [0]]))
plt.subplot(1, 2, 2)
boomerang_plot(out_boom2, theta_star)
common.save_figure("fig5-8-Boomerang.pdf")

####################################################################
# Figure 5.6                                                       #
####################################################################

##############################
## SUBSAMPLING EXAMPLE
##
##############################
  
###simulate next event for rate (a+bt)^+
def rlinearevent(a, b):
    if a <= 0 and b <= 0:
        return np.inf
    else:
        U = np.random.exponential(1)
        if a <= 0 and b > 0:
            return -(a/b) + np.sqrt(2*U/b)
        else:
            if a > 0 and b == 0:
                return U/a
            else:
                if a > 0 and b <= 0:
                    if U > -a**2/(2*b):
                        return np.inf
                    else:
                        return -a/b + np.sqrt(a**2 + 2*U*b)/b
                else:
                    return -a/b + np.sqrt(a**2 + 2*U*b)/b

##Zig-Zag sampler
## Simulates for a fixed number of proposed events
## outputs all proposed event times and positions

#theta0,p0 is initial state
## nev is number of events
## gradlog-pi is estimator of grad-log-pi(Different for different samplers)
## bound is a function of theta that outputs (a,b) so bound is (a+bt)^+
def ZZ_SS(theta, p, nev, gradlogpi, bound):
    d = len(theta)
    theta_st = np.zeros((nev+1, d+2))
    dt = np.zeros(d)
    a = np.zeros(d)
    b = np.zeros(d)
    t = 0
    theta_st[0,:] = np.hstack((t, theta, 0))  ##store time, theta and if an event or not
    for j in range(nev):
        for i in range(d):
            out_bound = bound(theta, p, i)
            a[i] = out_bound[0]
            b[i] = out_bound[1]
            dt[i] = rlinearevent(a[i], b[i])           
                
        i = np.argmin(dt)
        ##update time and state
        t += dt[i]
        theta += p * dt[i]
        rate = -p[i] * gradlogpi(theta, i)
        prob = rate / (a[i] + b[i] * dt[i])
        
        if prob > 1:
            print("Error in thinning")
            
        if np.random.uniform() < prob:
            p[i] = -p[i]
            theta_st[j+1,:] = np.hstack((t, theta, 1))
        else:
            theta_st[j+1,:] = np.hstack((t, theta, 0))
            
    return theta_st

####functions for different samplers
##log-pi for constant prior
def logpi(theta, y, X):
    y_2_columns = np.hstack((y, y))  
    logl = -np.sum(np.log(1 + np.exp(np.matmul(X, theta))) - np.matmul(y_2_columns*X, theta))
    return logl

##grad-log-pi -- theta^i derivative at theta
def gradlogpi_full(theta, i):
    expXtheta = np.exp(np.matmul(X, theta))
    prob = (expXtheta / (1 + expXtheta)).reshape((n, 1)) 
    x_col = np.array(X[:,i]).reshape((n, 1)) # necessary for element-wise multiplication to work on next line
    gradlogpi = np.sum(x_col * (y - prob))
        
    return gradlogpi

def bound_full(theta, p, i):
    a = -p[i] * gradlogpi_full(theta, i)
    m_full_col = np.array(M_FULL[:,i]).reshape((2, 1)) # necessary for element-wise multiplication to work on next line
    b = np.sqrt(len(p)) * np.sqrt(np.sum(m_full_col**2))
    return np.array([a, b])

##subsampling
def gradlogpi_ss(theta, i):
    j = random.choice(range(n))
    expXjtheta = np.exp(np.matmul(X[j,:], theta))
    prob = expXjtheta / (1 + expXjtheta)
    gradlogpi = n * (X[j,i] * (y[j] - prob))
    return gradlogpi

def bound_ss(theta, p, i):
    return np.array([M_ss[i], 0])

##control variates
##g.hat is the gradient at theta.hat; prob.hat is the prob at theta.hat
def gradlogpi_cv(theta, i):
    j = random.choice(range(n))
    expXjtheta = np.exp(np.matmul(X[j,:], theta))
    prob = expXjtheta / (1 + expXjtheta)
    gradlogpi = g_hat[i] + n * (X[j,i] * (prob_hat[j] - prob))
    return gradlogpi

def bound_cv(theta, p, i):    
    a = -p[i] * g_hat[i] + M_cv[i] * np.sqrt(np.sum((theta-theta_hat)**2))
    b = np.sqrt(len(p)) * M_cv[i]
    return np.array([a, b])

############SIMULATE DATA AND PLOTS
  
#two dimensional data

common.seed(4)
n = 100  ##data points
L = 450  ##time steps for full
theta_true = np.array([0.5, 1]) # parameter
X = np.column_stack((np.ones(n), np.random.normal(size=n)))  #covariate
expXtheta_true = np.exp(np.matmul(X, theta_true))
p = expXtheta_true/(1 + expXtheta_true)
y = np.random.binomial(1, p, size=n) #response
p = p.reshape((n, 1))
y = y.reshape((n, 1))

XTX = np.matmul(X.T, X)
  
##PLOT
common.plot_figure(3, 2, pad = 3.5)
#t1s = np.arange(0.2, 1.41, 0.01)
#t2s = np.arange(-0.25, 1.51, 0.01)
t1s = np.arange(-0.5, 1.26, 0.01)
t2s = np.arange(-0.5, 1.26, 0.01)
z = np.zeros((len(t1s), len(t2s)))
for i in range(len(t1s)):
    for j in range(len(t2s)):
        z[i,j] = logpi(np.array([t1s[i], t2s[j]]).reshape((2,1)), y, X)
z = z - np.max(z)

##NORMAL Zig-Zag
def plot_zz(out_data, plot_no):
    plt.subplot(3, 2, plot_no)
    plt.imshow(z, extent=[t1s[0], t1s[-1], t2s[0], t2s[-1]], aspect='auto', cmap=common.get_heat_color_map(10), origin='lower', zorder=1)
    plt.plot(out_data[:,1], out_data[:,2], '-', linewidth=2, color=common.black, zorder=2)
    plt.scatter(out_data[:,1], out_data[:,2], marker='.', s=3, color=common.heat_plot_path_col, zorder=3)
    plt.xlabel(r'$\theta^{(1)}$')
    plt.ylabel(r'$\theta^{(2)}$')
    plt.xlim(min(t1s), max(t1s))
    plt.ylim(min(t2s), max(t2s))

##BOUND FOR FULL METHOD
M_FULL = 0.25 * np.matmul(X.T, X)
out_full = ZZ_SS(theta_true, np.ones(2), L, gradlogpi_full, bound_full)
plot_zz(out_full, 1)

if common.VERBOSE:
    print("FULL: nev=", out_full.shape[0]-1, "time=", np.max(out_full[:,0]), "true events=", np.sum(out_full[:,3]))

##BOUND for SS
M_ss = n * np.max(np.abs(X), axis=0)
out_ss = ZZ_SS(theta_true, np.ones(2), L*n, gradlogpi_ss, bound_ss)
plot_zz(out_ss, 3)

if common.VERBOSE:
    print("SS: nev=", out_ss.shape[0]-1, "time=", np.max(out_ss[:,0]), "true events=", np.sum(out_ss[:,3]))

###CV METHOD
out_opt = minimize(lambda theta, y, X: -logpi(theta, y, X), x0=np.zeros(2), args=(y, X), method='BFGS')
theta_hat = out_opt.x.reshape((2,1))

d = len(theta_hat) ######ADDITION TO CODE
expXtheta_hat = np.exp(np.matmul(X, theta_hat))
prob_hat = expXtheta_hat/(1 + expXtheta_hat)
g_hat = np.zeros(d)

for j in range(d):
    g_hat[j] = gradlogpi_full(theta_hat, j)
    
M_cv = np.zeros(d)

for j in range(d):
    a = np.abs(X[:,j])
    b = np.sqrt(np.sum(X**2, axis=1))
    M_cv[j] = n * np.max(a*b)
    
out_cv = ZZ_SS(theta_true, np.ones(2), L * n, gradlogpi_cv, bound_cv)
plot_zz(out_cv, 5)

if common.VERBOSE:
    print("CV: nev=", out_cv.shape[0]-1, "time=", np.max(out_cv[:,0]), "true events=", np.sum(out_cv[:,3]))

###LARGER DATA
#two dimensional data

common.seed(1)
n = 900  ##data points
L = 50
theta_true = np.array([0.5, 1]) # parameter

X = np.column_stack((np.ones(n), np.random.normal(size=n))) #covariate
expXtheta_true = np.exp(np.matmul(X, theta_true))
p = expXtheta_true/(1 + expXtheta_true)
y = np.random.binomial(1, p, size=n) #response
p = p.reshape((n, 1))
y = y.reshape((n, 1))
XTX = np.matmul(X.T, X)

##PLOT
t1s = np.arange(0.25, 0.75 + 0.01/3, 0.01/3)
t2s = np.arange(0.5, 1.1 + 0.01/3, 0.01/3)
z = np.zeros((len(t1s), len(t2s)))
for i in range(len(t1s)):
    for j in range(len(t2s)):
        z[i,j] = logpi(np.array([t1s[i], t2s[j]]).reshape((2,1)), y, X)
        
z = z - np.max(z)

##NORMAL Zig-Zag
  
##BOUND FOR FULL METHOD
M_FULL = 0.25 * np.matmul(X.T, X)
out_full = ZZ_SS(theta_true, np.ones(2), L, gradlogpi_full, bound_full)
plot_zz(out_full, 2)

if common.VERBOSE:
    print("FULL: nev=", out_full.shape[0]-1, "time=", np.max(out_full[:,0]), "true events=", np.sum(out_full[:,3]))

##BOUND for SS
M_ss = n * np.max(np.abs(X), axis=0)
out_ss = ZZ_SS(theta_true, np.ones(2), L*n, gradlogpi_ss, bound_ss)
plot_zz(out_ss, 4)

if common.VERBOSE:
    print("SS: nev=", out_ss.shape[0]-1, "time=", np.max(out_ss[:,0]), "true events=", np.sum(out_ss[:,3]))

###CV METHOD
out_opt = minimize(lambda theta, y, X: -logpi(theta, y, X), x0=np.zeros(2), args=(y, X), method='BFGS')
theta_hat = out_opt.x.reshape((2,1))

expXtheta_hat = np.exp(np.matmul(X, theta_hat))
prob_hat = expXtheta_hat/(1 + expXtheta_hat)
g_hat = np.zeros(d)

for j in range(d):
    g_hat[j] = gradlogpi_full(theta_hat, j)
    
M_cv = np.zeros(d)

for j in range(d):
    a = np.abs(X[:,j])
    b = np.sqrt(np.sum(X**2, axis=1))
    M_cv[j] = n * np.max(a*b)
    
out_cv = ZZ_SS(theta_true, np.ones(2), L*n, gradlogpi_cv, bound_cv)
plot_zz(out_cv, 6)

if common.VERBOSE:
    print("CV: nev=", out_cv.shape[0]-1, "time=", np.max(out_cv[:,0]), "true events=", np.sum(out_cv[:,3]))

common.save_figure("fig5-6-ZZsubsamplinglogistic.pdf")



