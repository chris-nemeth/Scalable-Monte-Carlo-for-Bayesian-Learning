import common

import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numdifftools as nd 
import csv
from pyhmc import hmc
from numpy.linalg import inv

####################################################################
# Figure 6.1                                                       #
####################################################################

def shade_background_1(x_vals, p_vals):    
    ax = plt.gca()
    lim = ax.axis()
    for i in range(len(x_vals)-1):
        pat = plt.Polygon([[lim[0], x_vals[i]], [lim[1], x_vals[i]], [lim[1], x_vals[i+1]], [lim[0], x_vals[i+1]]], closed=True, edgecolor='none')
        col = (1 - 0.7 * (p_vals[i] / max(p_vals))) * np.array([1, 1, 1])
        pat.set_facecolor(col)
        ax.add_patch(pat)
        ax.set_axis_on()

def shade_background_2(lim_x, lim_y, p):   
    x1_vals = np.linspace(lim_x[0], lim_x[1], 100)
    x2_vals = np.linspace(lim_y[0], lim_y[1], 100)
    f_vals = np.zeros((len(x1_vals), len(x2_vals)))
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            f_vals[j, i] = - p(x1_vals[i], x2_vals[j])
    plt.imshow(f_vals, extent=[min(x1_vals), max(x1_vals), min(x2_vals), max(x2_vals)], cmap='gray', origin='lower')
    plt.gca().set_axis_on()

# Convergence diagnostics for MCMC
common.seed(1000)  # fix random seed, 8 ok

# target pdf
def p(x):
    return 0.5 * norm.pdf(x, -2, 0.7) + 0.5 * norm.pdf(x, 2, 0.7)

def logp(x):  
    return np.log(p(x))

lims = [-5, 5]  # domain for plotting

# gradient of log target pdf
def d_logp(x):
    return nd.Gradient(logp)([x])

# the Metropolis--Hastings transition kernel (MALA)
epsilon = 0.1  # proposal step size
def mu_prop(x):
    return x + epsilon * d_logp(x)  # proposal mean

def var_prop(x):
    return 2 * epsilon  # proposal variance

def logq(x, y):
    return -0.5 * np.log(2 * np.pi * var_prop(x)) - (y - mu_prop(x))**2 / (2 * var_prop(x))  # log pdf of proposal (x -> y)

def log_alpha(x, y):
    return logp(y) - logp(x) + logq(y, x) - logq(x, y)  # log acceptance ratio (x -> y)

n = 1000 # number of MCMC iterations

# run L Markov chains in parallel
L = 3  # number of chains
x = np.zeros((n+1, L, 2))
x[0, :, 0] = [-4.7, -2.1, 1.6]  # initial states - lucky initialisation
x[0, :, 1] = [-3.4, -4.5, -0.6]  # initial states - unlucky initialisation

for r in range(2):  # independent realisations
    for l in range(L):
        for i in range(n):
            # propose a new state
            x_proposed = mu_prop(x[i, l, r]) + np.random.randn() * np.sqrt(var_prop(x[i, l, r]))
            # decide whether to accept or reject
            if np.log(np.random.rand()) < log_alpha(x[i, l, r], x_proposed):
                x[i+1, l, r] = x_proposed
            else:
                x[i+1, l, r] = x[i, l, r]

# compute convergence diagnostics along the sample path(s)
Rhat = np.zeros((n+1, 2))
for r in range(2):
    for i in range(n+1):        
        m = np.mean(np.squeeze(x[:(i+1), :, r]), axis=0)
        s2 = np.var(np.squeeze(x[:(i+1), :, r]), axis=0)
        hats2 = np.mean(s2)
        hatsig2 = (i/(i+1)) * hats2 + (1/(L-1)) * np.sum((m - np.mean(m))**2)
        Rhat[i, r] = np.sqrt(hatsig2 / hats2)

Rhat[Rhat < 1] = np.nan  # to avoid bug in plotting code

common.plot_figure(2, 2, pad = 3.6)
cols = [common.line_plot_cols[0], common.line_plot_cols[1], common.line_plot_cols[2]]
x_vals = np.linspace(lims[0], lims[1], 100)
p_vals = p(x_vals)

for r in range(2):
    plt.subplot(2, 2, r+1)    
    #plt.gca().set_prop_cycle(None)
    for l in range(L):
        plt.plot(range(n+1), np.squeeze(x[:, l, r]), color=cols[l])
        plt.plot(0, x[0, l, r], "*", markerfacecolor=cols[l], markeredgecolor='k', markersize=10)
    
    plt.ylim(lims)
    shade_background_1(x_vals, p_vals)
    plt.box(on=True)
    plt.xlabel('iteration')    
    plt.gca().set_yticks([-2, 0, 2])
    plt.ylabel(r"$x$")
   
    plt.subplot(2, 2, r+3)
    plt.semilogy(range(n+1), Rhat[:, r]-1, '-', color=common.line_plot_cols[0])
    plt.semilogy([0,n], [0.1,0.1], color=common.black, linestyle='dotted')
    plt.semilogy([0,n], [0.01,0.01], color=common.black, linestyle='dashed')
    plt.xlabel('iteration')
    plt.box(on=True)
 
    plt.ylabel(r"$\widehat{R}-1$")
       
    
    plt.ylim([np.nanmin(Rhat[1:, :].flatten() - 1), np.nanmax(Rhat[1:, :].flatten() - 1)])

common.save_figure("fig6-1-convergence_diagnostics.pdf")

####################################################################
# Figure 6.2                                                       #
####################################################################

# Bias diagnostics for MCMC
common.seed(0)  # fix random seed

# target pdf
def p(x):
    return 0.5 * norm.pdf(x, -2, 0.7) + 0.5 * norm.pdf(x, 2, 0.7)

def logp(x):
    return np.log(p(x))

lims = [-5, 5]  # domain for plotting

# run L Markov chains in parallel
n = 1000 # number of MCMC iterations
L = 3  # number of chains
x = np.zeros((n+1, L, 2))
x[0, :, 0] = [-4.7, -2.1, 1.6]  # initial states - lucky initialisation
x[0, :, 1] = [-3.4, -4.5, -0.6]  # initial states - unlucky initialisation
mu_vals = [2, 0]  # mean of the biased sampler target

for r in range(2):  # different realisations
    # incorrect pdf
    def notp(x):
        return norm.pdf(x, mu_vals[r], 0.7)

    def lognotp(x):
        return np.log(notp(x))

    lims = [-5, 5]  # domain for plotting

    # gradient of log incorrect pdf
    def d_lognotp(x):        
        return nd.Gradient(lognotp)([x])

    # the incorrect Metropolis--Hastings transition kernel (MALA)
    epsilon = 0.1  # proposal step size
    def mu_prop(x):
        return x + epsilon * d_lognotp(x)  # proposal mean

    def var_prop(x):
        return 2 * epsilon  # proposal variance

    def logq(x, y):
        return -0.5 * np.log(2 * np.pi * var_prop(x)) - (y - mu_prop(x))**2 / (2 * var_prop(x))  # log pdf of proposal (x -> y)

    def log_alpha(x, y):
        return lognotp(y) - lognotp(x) + logq(y, x) - logq(x, y)  # log acceptance ratio (x -> y)

    # run Markov chains
    for l in range(L):
        for i in range(n):
            # propose a new state
            x_proposed = mu_prop(x[i, l, r]) + np.random.randn() * np.sqrt(var_prop(x[i, l, r]))
            # decide whether to accept or reject
            if np.log(np.random.rand()) < log_alpha(x[i, l, r], x_proposed):
                x[i+1, l, r] = x_proposed
            else:
                x[i+1, l, r] = x[i, l, r]

common.plot_figure(2, 2, pad = 3.6)
cols = [common.line_plot_cols[0], common.line_plot_cols[1], common.line_plot_cols[2]]
x_vals = np.linspace(lims[0], lims[1], 100)
p_vals = p(x_vals)

Bhat = np.zeros((n + 1, L, 2))
for r in range(2):
    for l in range(L):
        Bhat[:, l, r] = np.abs(np.cumsum(x[:, l, r])) / np.arange(1, n+2) # cum mean, absolute
        
for r in range(2):
    plt.subplot(2, 2, r+1)
    plt.gca().set_prop_cycle(None)
    for l in range(L):
        plt.plot(range(n+1), np.squeeze(x[:, l, r]), color=cols[l])
        plt.plot(0, x[0, l, r], "*", markerfacecolor=cols[l], markeredgecolor='k', markersize=10)
    plt.ylim(lims)
    plt.fill_between(x_vals, p_vals, alpha=0.2)
    shade_background_1(x_vals,p_vals)
    plt.box(on=True)
    
    plt.gca().set_yticks([-2, 0, 2])
    plt.ylabel(r'$x$')
  
    plt.subplot(2, 2, r+3)
    for l in range(L):        
        plt.loglog(range(n+1), Bhat[:,l,r], color=cols[l])
        
    plt.xlabel('iteration')
    plt.box(on=True)    
    plt.ylabel(r'$\widehat{B}$')
  
    plt.ylim([min(Bhat[:,:,:].flatten()), max(Bhat[:,:,:].flatten())])

common.save_figure("fig6-2-bias_diagnostics.pdf")

####################################################################
#                                                                  #
####################################################################

# Inputs:  
# X = n x d array, each row a distinct point
# G = n x d array, each row the gradient of the log target at the corresponding point
# Gam = d x d positive definite matrix, used as a preconditioner in the IMQ kernel
# w = n x 1 vector, each element a weight assigned to the associated state

# Ouputs
# ksd1, w1,   unweighted points
# ksd2, w2,   w-weighted points
# ksd3, w3,   any sign weights
# ksd4, w4,   positive weights

# Equivalent to matlab dot(a, b, 2)
def dot2(A, B):
    return np.sum((A * B), axis = 1)

def comp_ksd(X, G, Gam, w, return_all = False, solver = 'cvxopt'):
    # dimension
    n = X.shape[0]
    if len(X.shape) == 1:        
        X = X.reshape(n, 1)  
        G = G.reshape(n, 1)
   
    # vectorised computation of Stein kernel matrix
    tmp0 = np.trace(np.linalg.inv(Gam))   
    tmp1 = dot2(np.tile(G, (n, 1)), np.repeat(G, n, axis = 0))
    tmp2 = np.tile(X, (n, 1)) - np.repeat(X, n, axis=0)
    tmp3 = np.tile(G, (n, 1)) - np.repeat(G, n, axis=0)
    tmp4 = (np.linalg.solve(Gam, tmp2.T)).T   
    tmp5 = - 3 * dot2(tmp4, tmp4) / ((1 + dot2(tmp2, tmp4))**(5/2)) + (tmp0 + dot2(tmp3, tmp4)) / ((1 + dot2(tmp2, tmp4))**(3/2)) + tmp1 / ((1 + dot2(tmp2, tmp4))**(1/2))
    
    K = tmp5.reshape(n, n)
    
    # compute outputs
    ksd1 = np.zeros((n,1))
    w1 = np.full((n, 1), 1/n)
    ksd2 = np.zeros((n,1))
    w2 = w
   
    ksd1[0] = np.sqrt(K[0, 0])
    ksd2[0] = np.sqrt(K[0, 0])
    for i in range(1, n):      
        ksd1[i] = (1/(i+1)) * np.sqrt((i**2) * ksd1[i-1]**2 + 2 * np.sum(K[i, :i]) + K[i, i])
        ksd2[i] = (1/np.sum(w[:(i+1)])) * np.sqrt(np.sum(w[:i])**2 * ksd2[i-1]**2 + 2 * w[i]*np.matmul(K[i, :i], w[:i]) + (w[i]**2)*K[i, i])
        
    ksd3 = 0 # just compute final value
    ksd4 = 0 # just compute final value
    w3 = 0
    w4 = 0
    
    if return_all:         
        w3 = solve_qp(P = K, q = np.zeros((n, 1)), A = np.ones((1, n)), b = np.ones((1, 1)), solver=solver)        
        ksd3 = np.sqrt(np.matmul(w3.T, np.matmul(K, w3)))         
        w4 = solve_qp(P = K, q = np.zeros((n, 1)), A = np.ones((1, n)), b = np.ones((1, 1)), lb = np.zeros((n, 1)), solver=solver)
        ksd4 = np.sqrt(np.matmul(w4.T, np.matmul(K, w4)))
    
    return ksd1, w1, ksd2, w2, ksd3, w3, ksd4, w4

####################################################################
# Figure 6.3                                                       #
####################################################################

# Performance of kernel Stein discrepancy
common.seed(1)  # fix random seed

# target pdf
def p(x):
    return 0.5 * norm.pdf(x, -2, 0.7) + 0.5 * norm.pdf(x, 2, 0.7)

def logp(x):
    return np.log(p(x))

lims = [-5, 5]  # domain for plotting

# gradient of log target pdf
def d_logp(x):    
    return nd.Gradient(logp)([x])

# incorrect pdf
def notp(x):
    return norm.pdf(x, 2, 0.8)

def lognotp(x):
    return np.log(notp(x))

# gradient of log not target pdf
def d_lognotp(x):    
    return nd.Gradient(lognotp)([x])

# length and number of chains
L = 3  # number of chains
n = 10000  # number of MCMC iterations
x = np.zeros((n+1, L, 2))

# the Metropolis--Hastings transition kernel (MALA)
epsilon = 0.6  # proposal step size
def mu_prop(x):
    return x + epsilon * d_logp(x)  # proposal mean

def var_prop(x):
    return 2 * epsilon  # proposal variance

def logq(x, y):
    return -0.5 * np.log(2 * np.pi * var_prop(x)) - ((y - mu_prop(x))**2) / (2 * var_prop(x))  # log pdf of proposal (x -> y)

def log_alpha(x, y):
    return logp(y) - logp(x) + logq(y, x) - logq(x, y)  # log acceptance ratio (x -> y)

# run L correct Markov chains in parallel
x[0, :, 0] = [-4.7, -2.1, 1.6]  # initial states
for l in range(L):
    for i in range(n):
        # propose a new state
        x_proposed = mu_prop(x[i, l, 0]) + np.random.randn() * np.sqrt(var_prop(x[i, l, 0]))
        # decide whether to accept or reject
        if np.log(np.random.rand()) < log_alpha(x[i, l, 0], x_proposed):
            x[i+1, l, 0] = x_proposed
        else:
            x[i+1, l, 0] = x[i, l, 0]

# the incorrect Metropolis--Hastings transition kernel (MALA)
epsilon = 0.1  # proposal step size
def mu_prop(x):
    return x + epsilon * d_lognotp(x)  # proposal mean

def var_prop(x):
    return 2 * epsilon  # proposal variance

def logq(x, y):
    return -0.5 * np.log(2 * np.pi * var_prop(x)) - ((y - mu_prop(x))**2) / (2 * var_prop(x))  # log pdf of proposal (x -> y)

def log_alpha(x, y):
    return lognotp(y) - lognotp(x) + logq(y, x) - logq(x, y)  # log acceptance ratio (x -> y)

# run L incorrect Markov chains in parallel
x[0, :, 1] = [-4.7, -2.1, 1.6]  # initial states
for l in range(L):
    for i in range(n):
        
            
        # propose a new state
        x_proposed = mu_prop(x[i, l, 1]) + np.random.randn() * np.sqrt(var_prop(x[i, l, 1]))       
           
        # decide whether to accept or reject
        if np.log(np.random.rand()) < log_alpha(x[i, l, 1], x_proposed):
            x[i+1, l, 1] = x_proposed
        else:
            x[i+1, l, 1] = x[i, l, 1]

# compute convergence diagnostics along the sample path(s)
KSD = np.zeros((n+1, L, 2))
for r in range(2):
    for l in range(L):
        # calculate each element of gradient
        d_logp_vec = np.zeros((n+1, 1))
        for i in range(n+1):
            d_logp_vec[i] = d_logp(x[i, l, r])
            
        KSD[:, l, r] = (comp_ksd(np.squeeze(x[:, l, r]), d_logp_vec, np.eye(1), np.ones(n+1)/(n+1)))[0].flatten() #get ksd1 from function

common.plot_figure(2, 2, pad = 3.5)
cols = [common.line_plot_cols[0], common.line_plot_cols[1], common.line_plot_cols[2]]
x_vals = np.linspace(lims[0], lims[1], 100)
p_vals = p(x_vals)

for r in range(2):
    plt.subplot(2, 2, r+1)
    plt.gca().set_prop_cycle(None)
    for l in range(L):
        plt.plot(range(n+1), np.squeeze(x[:, l, r]), color=cols[l])
        plt.plot(0, x[0, l, r], "*", markerfacecolor=cols[l], markeredgecolor='k', markersize=10)
    plt.ylim(lims)
    shade_background_1(x_vals, p_vals)
    plt.box(on=True)
    
    plt.gca().set_yticks([-2, 0, 2])
    plt.xlabel(r'iteration')
    plt.ylabel(r'$x$')
   
    plt.subplot(2, 2, r+3)
    for l in range(L):
        plt.loglog(range(n+1), KSD[:, l, r], color=cols[l])
        
    plt.xlabel(r'iteration')
    plt.box(on=True)  
    plt.ylabel(r'$\mathcal{D}_{\mathsf{k}_\pi}(\nu_n)$')
  
    plt.ylim([min(KSD.flatten()), max(KSD.flatten())])

common.save_figure("fig6-3-performance_ksd.pdf")

####################################################################
# Figure 6.4                                                       #
####################################################################

def get_data_from_file(filename):
    with open("Data/" + filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)      
        return np.array(data).astype(float)

# mean absolute deviation
def mad(data, axis=None):    
    a = np.mean(data, axis)
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

# Stochastic gradient Stein discrepancies
common.seed(0)  # fix random seed

d = 100  # dimension of the logistic regression target
X_100 = get_data_from_file("sgldsamples100.csv")
G_100 = get_data_from_file("sgldgrads100.csv")
X_1000 = get_data_from_file("sgldsamples1000.csv")
G_1000 = get_data_from_file("sgldgrads1000.csv")

# standardisation
scl = mad(X_1000, axis = 0) # mean absolute deviation

if min(scl) == 0:
    scl = np.maximum(np.ones(d), scl) # avoid division by 0

X_100_std = X_100 / scl
X_1000_std = X_1000 / scl
G_100_std = G_100 * scl  # using the chain rule
G_1000_std = G_1000 * scl  # using the chain rule

# compute stochastic gradient Stein discrepancy
n_max = 1000  # number of iterations
KSD_100 = comp_ksd(X_100_std[:n_max, :], G_100_std[:n_max, :], np.eye(d), np.ones(n_max)/(n_max))[0].flatten() #get ksd1 from function
KSD_1000 = comp_ksd(X_1000_std[:n_max, :], G_1000_std[:n_max, :], np.eye(d), np.ones(n_max)/(n_max))[0].flatten() #get ksd1 from function

common.plot_figure(1, 1, book_scale = 0.7)
plt.loglog(range(1, n_max+1), KSD_100, '-', color = common.line_plot_cols[0], label = r'$m = 1,000$')
plt.loglog(range(1, n_max+1), KSD_1000, '--', color = common.line_plot_cols[1], label = r'$m = 100$')
plt.ylim([0.999, 100.001])
plt.xlabel(r'iteration')
plt.box(on=True)
plt.ylabel(r'$\mathcal{D}_{\widehat{\mathsf{k}}_\pi}(\nu_n)$')
plt.legend(loc='upper right')

common.save_figure("fig6-4-stochastic_ksd.pdf")

####################################################################
# Figure 6.5                                                       #
####################################################################

# Optimal weights for MCMC
common.seed(0)  # fix random seed

# (un-normalised) posterior
a = 0
b = 3
def p(x1, x2):
    return np.exp(-(a-x1)**2 - b*(x2-x1**2)**2)

# gradient of log posterior
d = 2  # dimension
#def u(x):
#    x1, x2 = x
#    return np.array([2*(a-x1) + 4*b*x1*(x2-x1**2), 2*b*(x2-x1**2)])

# log posterior
def logp(x):
    x1, x2 = x
    return np.log(p(x1, x2))

# log posterior gradient 
def d_logp(x):
    return nd.Gradient(logp)([x])


# candidate states
n = 500
X = np.random.randn(n, 2)
G = np.zeros((n, 2))
p_vals = np.zeros(n)
q_vals = np.zeros(n)

for i in range(n):
    G[i, :] = d_logp(X[i, :]) #u(X[i, :])
    p_vals[i] = p(X[i, 0], X[i, 1]) 
    q_vals[i] = multivariate_normal.pdf(X[i, :], mean = np.zeros(d), cov = np.eye(d))

# compute optimal weights and KSD
ksd_unweighted = np.zeros(n)
ksd_weighted_1 = np.zeros(n)  # sign constrained
ksd_weighted_2 = np.zeros(n)  # not sign constrained
ksd_weighted_3 = np.zeros(n)  # self-normalised importance sampling

for i in range(n):
    # standardisation
    scl = mad(X[:i+1, :], axis = 0) # mean absolute deviation

    if min(scl) == 0:
        scl = np.maximum(np.ones(d), scl) # avoid division by 0

    X_std = X[:i+1, :] / scl
    G_std = G[:i+1, :] * scl  # using the chain rule

    # optimal weights    
    ksd1, w1, ksd2, w2, ksd3, w3, ksd4, w4 = comp_ksd(X_std, G_std, np.eye(2), np.ones(i+1)/(i+1), return_all = True)
    ksd_unweighted[i] = ksd1[-1, 0]
    ksd_weighted_1[i] = ksd4
    ksd_weighted_2[i] = ksd3
    w_SNIS = (p_vals[:i+1] / q_vals[:i+1]) / np.sum(p_vals[:i+1] / q_vals[:i+1])
    _, _, tmp, _, _, _, _, _ = comp_ksd(X_std, G_std, np.eye(2), np.array(w_SNIS), return_all = True)
    ksd_weighted_3[i] = tmp[-1, 0]

fig, ax = common.plot_figure(2, 2, pad = 3.5)
ax[1, 0].axis('off')
ax[1, 1].axis('off')
lim_x = [-2, 2]
lim_y = [-1, 2.5]

plt.subplot(2, 2, 1)
shade_background_2(lim_x, lim_y, p)
scl = max(abs(w1))
plt.scatter(X[:, 0], X[:, 1], (1/scl) * w4, color=common.scatter_cols[0], marker='o', facecolors=common.scatter_cols[0], edgecolors='white', linewidths=0.5)
plt.xlim(lim_x)
plt.ylim(lim_y)
lgnd1 = plt.legend({r'$w_i^\star$'}, loc='upper right')
lgnd1.legend_handles[0]._sizes = [40]
plt.box(on=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.subplot(2, 2, 2)
shade_background_2(lim_x, lim_y, p)
scl = max(abs(w3))
ix_pos = (w3 > 0) # positive weights
ix_neg = (w3 < 0) # negative weights
plt.scatter(X[ix_neg, 0], X[ix_neg, 1], -(30/scl) * w3[ix_neg], color=common.scatter_cols[1], marker='^', facecolors=common.scatter_cols[1], edgecolors='white', linewidths=0.5, label = r'$\widetilde{w}_i^\star < 0$')
plt.scatter(X[ix_pos, 0], X[ix_pos, 1], (30/scl) * w3[ix_pos], color=common.scatter_cols[0], marker='o', facecolors=common.scatter_cols[0], edgecolors='white', linewidths=0.5, label = r'$\widetilde{w}_i^\star > 0$')
plt.xlim(lim_x)
plt.ylim(lim_y)
lgnd2 = plt.legend(loc='upper right')
lgnd2.legend_handles[0]._sizes = [40]
lgnd2.legend_handles[1]._sizes = [40]
plt.box(on=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.subplot(2, 1, 2)
plt.loglog(range(1, n+1), ksd_unweighted, '-', color = common.line_plot_cols[0], label = r'$\nu = \nu_n$')
plt.loglog(range(1, n+1), ksd_weighted_1, ':', color = common.line_plot_cols[1], label = r'$\nu = \nu_n^\star$')
plt.loglog(range(1, n+1), ksd_weighted_2, '--', color = common.line_plot_cols[2], label = r'$\nu = \widetilde{\nu}_n^\star$') 
plt.loglog(range(1, n+1), ksd_weighted_3, '-.', color = common.line_plot_cols[3], label = 'Self-normalised I.S.')
plt.xlabel('number of samples')
plt.box(on=True)
plt.ylabel(r'$\mathcal{D}_{\mathsf{k}_\pi}(\nu)$')
plt.legend(loc='lower left')

common.save_figure("fig6-5-optimal_weights.pdf")

####################################################################
# Figure 6.6                                                       #
####################################################################

def thin(X, G, m, stnd=True):
    """
    Stein Thinning algorithm.

    Parameters:
    X      - n x d array, each row a sample from MCMC.
    G      - n x d array, each row the gradient of the log target.
    m      - desired number of points.
    stnd   - optional boolean, either True (default) or False, indicating
             whether or not to standardise the columns of X.
   
    Returns:
    pi     - m x 1 vector, containing the row indices in X of the selected
             points.
    ksd    - m x 1 vector, with jth entry giving the KSD of the unweighted
             empirical distribution based on the first j points selected.
    """
    
    n, d = X.shape
    if n == 0 or d == 0:
        raise ValueError('X is empty')
    if G.shape != (n, d):
        raise ValueError('Dimensions of X and G are inconsistent')
    if np.isnan(X).any() or np.isnan(G).any():
        raise ValueError('One of X or G contains NaNs')
    if np.isinf(X).any() or np.isinf(G).any() > 0:
        raise ValueError('One of X or G contains infs')

    # Standardisation
    if stnd:   
        # standardisation
        scl = mad(X, axis = 0) # mean absolute deviation

        if min(scl) == 0:
            raise ValueError('Too few unique samples in X')
        
        X = X / scl
        G = G * scl  # using the chain rule

    # Preconditioner    
    Gam = np.eye(d)

    # Stein kernel sub-matrix K(X,pi) (i.e., just store entries that we need)
    K_X_pi = np.zeros((n, m))
    
    # also compute the diagonal elements of the Stein kernel matrix
    K_diag = d + np.sum(G * G, axis=1) # check
    
    # main loop
    pi = np.zeros((m, 1)).astype(int)
    ksd = np.zeros((m, 1))
    tmp0 = np.matmul(X, inv(Gam))
    
    for j in range(m):
        # monitor
        if common.VERBOSE:
            print(f'Selecting point {j+1} of {m}')
        
        # select next point
        if j == 0:
            pi[0] = np.argmin((1/2) * K_diag)
        else:
            pi[j] = np.argmin((1/2) * K_diag + np.sum(K_X_pi[:, :j], axis=1))
        
        # populate row and column of kernel matrix associate with new point      
        tmp1_1 = tmp0 - np.tile(np.matmul(X[pi[j],:], inv(Gam)), (n, 1))
        tmp1 = dot2(tmp1_1, tmp1_1)       
        tmp2 = dot2(G - np.tile(G[pi[j],:], (n, 1)), tmp1_1)       
        tmp3 = dot2(X - np.tile(X[pi[j],:], (n, 1)), tmp1_1)      
        tmp4 = dot2(G, np.tile(G[pi[j],:], (n, 1)) )
        
        K_pi_j = (-3 * tmp1 / ((1 + tmp3) ** (5/2)) +
                  (np.trace(inv(Gam), dtype=float) + tmp2) / ((1 + tmp3) ** (3/2)) +
                  tmp4 / (1 + tmp3)**(1/2))
        
        K_X_pi[:, j] = K_pi_j
        
        # compute kernel Stein discrepancy       
        ksd[j] = (1/(j+1)) * np.sqrt( np.ones((1, j+1)) @ (K_X_pi[pi[:(j+1)].reshape(j+1), :(j+1)] @ np.ones((j+1, 1))) )
        
    return pi, ksd

####################################################################
      

# Optimal thinning of MCMC
common.seed(1)  # fix random seed

# (un-normalised) posterior
a = 0
b = 3
d = 2  # dimension

#def p(x):
#    return np.exp(-(a - x[0])**2 - b * (x[1] - x[0]**2)**2)

def p(x1, x2):
    return np.exp(-(a - x1)**2 - b*(x2 - x1**2)**2)

# log posterior
def logp(x):
    x1, x2 = x
    return np.log(p(x1, x2))

# log posterior gradient 
def d_logp(x):
    return nd.Gradient(logp)([x])

# log posterior and gradient
def logp_d_logp(x):  
    return logp(x), d_logp(x)

# candidate states
startpoint = np.array([0, 0])
n = 1000
X = hmc(logp_d_logp, x0 = startpoint, n_samples=n)

G = np.zeros((n, 2))
for i in range(n):
    G[i, :] = d_logp(X[i, :])

# standardisation
scl = mad(X, axis = 0) # mean absolute deviation

X_std = X / scl
G_std = G * scl  # using the chain rule

# optimal thinning
m = 1000  # number of states to select
ksd_unweighted, _, ksd2, _, ksd3, _, ksd_weighted, _ = comp_ksd(X_std, G_std, np.eye(2), np.ones(n)/n, True, solver = 'osqp')

common.plot_figure(1, 2)
plt.subplot(1, 2, 1)
lim_x = [-2, 2]
lim_y = [-1, 2.5]
shade_background_2(lim_x, lim_y, p)
plt.plot(X[:, 0], X[:, 1], 'k-', label = 'Markov Chain', linewidth=0.5, zorder=2)
N = 25  # number of states to plot
idx, ksd_thinned = thin(X_std, G_std, m, False)
plt.scatter(X[idx[:N], 0], X[idx[:N], 1], 10, color=common.scatter_cols[0], marker='o', facecolors=common.scatter_cols[0], edgecolors='white', linewidths=0.5, label = 'Selected States', zorder=3)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend(loc='upper right')
plt.box(on=True)

plt.subplot(1, 2, 2)
plt.loglog(range(1, m+1), ksd_thinned, color=common.line_plot_cols[0], label = r'$\nu = \nu_{n,m}$')
plt.gca().axhline(ksd_unweighted[-1], linestyle=':', label = r'$\nu = \nu_n$', color = common.black)
plt.gca().axhline(ksd_weighted, linestyle='--', label = r'$\nu = \nu_n^\star$', color = common.black)
plt.xlabel(r'$m$')
plt.ylabel(r'$\mathcal{D}_{\mathsf{k}_\pi}(\nu)$')
lgnd3 = plt.legend(loc='upper right')
lgnd3.legend_handles[0]._sizes = [50]
plt.box(on=True)
plt.ylim([0.5 * ksd_weighted, np.max(ksd_thinned)])

common.save_figure("fig6-6-optimal_thinning.pdf")



