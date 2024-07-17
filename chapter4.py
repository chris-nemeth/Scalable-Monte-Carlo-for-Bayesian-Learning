import common

import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from itertools import cycle

####################################################################
# Figure 4.1                                                       #
####################################################################

###function to calculate pi, TVD and TVDc
##
## N is number of states; h is size of move (uniform step of 1,2,..,h in positive/negative direction plus 1/N stay still)
## eps is non-reversible bias -- prob of positive move is (1+eps)/2
##
## burnin is burnin for calculating the proportion of time in each state
## M is the number of steps to run the chain for; thin is for TVD calculations

def nonrev(N, h=1, eps=0, burnin=0, M=0, thin=1):
  # default value for M
  if M == 0:
    M = N**2
    
  # construct P (remember range not inclusive of last number)
  P = np.zeros((N,N))
  for i in range(1, N + 1):
    index1 = map(lambda x:(x % N) + 1, range(i - h - 1, i - 2 + 1))
    index2 = map(lambda x:(x % N) + 1, range(i, i + h - 1 + 1))
    
    for ind1 in index1:
        P[i-1, ind1-1] = (1-eps)*((1-1/(2*h+1))/(2*h))
    
    for ind2 in index2:
        P[i-1, ind2-1] = (1+eps)*((1-1/(2*h+1))/(2*h))
        
    P[i-1, i-1] = 1/(2*h+1)
      
  TVD = np.zeros(M)
  TVD2 = np.full(M, np.nan)
  pi = np.append([1], np.zeros(N-1))
  pic = pi
  for i in range(1, M + 1):
    pi = np.matmul(pi, P)
    TVD[i-1] = sum(abs(pi - 1/N))
    if i > burnin:
      pic = pic * ((i - burnin - 1)/(i - burnin)) + pi /(i - burnin)
      TVD2[i-1] = sum(abs(pic - 1/N)) 
  
  x = int(np.floor(N/2))
  xs = np.zeros(M)
  TVD3 = np.full(M, np.nan)
  pic = np.zeros(N)
  j = 1
  for i in range(1, M + 1):
    x = random.choices(range(1, N + 1), weights = P[x-1], k = 1)[0]    
    xs[i-1] = x
    if i >= burnin:
      if ((i - burnin) % thin == 0):
        pic = pic * ((j - 1)/j)
        pic[x-1] = pic[x-1] + 1/j
        j=j+1
      
      TVD3[i-1] = sum(abs(pic - 1/N))
    
  return xs, TVD, TVD2, TVD3


TVD1 = np.zeros(4000)
TVD2 = np.zeros(16000)
TVD3 = np.zeros(64000)
TVD2l = np.zeros(8000)
TVD3l = np.zeros(16000)
TVD1n = np.zeros(4000)
TVD2n = np.zeros(8000)
TVD3n = np.zeros(16000)

for k in range(1,11):
    common.seed(k + 1234586) 

    out1 = nonrev(100,1,0,100,16000,1)
    out2 = nonrev(200,1,0,100,64000,1)
    out3 = nonrev(400,1,0,100,256000,1)
    out2l = nonrev(200,2,0,100,16000,1)
    out3l = nonrev(400,4,0,100,32000,1)
    out1n = nonrev(100,1,0.5,100,8000,1)
    out2n = nonrev(200,1,0.5,100,16000,1)
    out3n = nonrev(400,1,0.5,100,32000,1)

    # This was much clearer in R! TVD1=TVD1+out1$TVD1/10 etc.
    # Need to use cycle as arrays are of a different length
    TVD1 = [x + y for x, y in zip(cycle(TVD1), map(lambda x:x / 10, out1[3]))]
    TVD2 = [x + y for x, y in zip(cycle(TVD2), map(lambda x:x / 10, out2[3]))]
    TVD3 = [x + y for x, y in zip(cycle(TVD3), map(lambda x:x / 10, out3[3]))]
    TVD2l = [x + y for x, y in zip(cycle(TVD2l), map(lambda x:x / 10, out2l[3]))]
    TVD3l = [x + y for x, y in zip(cycle(TVD3l), map(lambda x:x / 10, out3l[3]))]
    TVD1n = [x + y for x, y in zip(cycle(TVD1n), map(lambda x:x / 10, out1n[3]))]
    TVD2n = [x + y for x, y in zip(cycle(TVD2n), map(lambda x:x / 10, out2n[3]))]
    TVD3n = [x + y for x, y in zip(cycle(TVD3n), map(lambda x:x / 10, out3n[3]))]
    
    if common.VERBOSE:
        print(".")

# function for ploting a sub plot for figure 4.1 
def plot_fig4_1_subplot(plot_no, values, x_max, y_max, S):
    plt.subplot(2, 3, plot_no)    
    plt.plot(range(1, len(values[0]) + 1), list(map(lambda x:x - 1, values[0])), 'o', color=common.line_plot_cols[0], markersize=1) 
    if S != 0:
       plt.title(r'$S={}$'.format(S))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$X_k$')
    plt.xlim(1, x_max)
    plt.ylim(0, y_max)

common.plot_figure(2, 3)
plot_fig4_1_subplot(1, out1, 4000, 100, 100)
plot_fig4_1_subplot(2, out2, 8000, 200, 200)
plot_fig4_1_subplot(3, out3, 16000, 400, 400)
plot_fig4_1_subplot(4, out1n, 400, 100, 0)
plot_fig4_1_subplot(5, out2n, 800, 200, 0)
plot_fig4_1_subplot(6, out3n, 1600, 400, 0)
common.save_figure("fig4-1-trace_nonrev.pdf")

####################################################################
# Figure 4.2                                                       #
####################################################################

# function for ploting a sub plot for figure 4.2 
def plot_fig4_2_subplot(plot_no, values1, values2, values3, S_squared = False):
   S1 = 400
   S2 = 200
   S3 = 100
   if S_squared:
       S1 *= S1
       S2 *= S2
       S3 *= S3
      
   plt.subplot(2, 2, plot_no)   
   plt.plot(list(map(lambda x:x / S1, range(1, len(values1) + 1))), values1, '-', color = common.line_plot_cols[0], label = r'$S=400$')
   plt.plot(list(map(lambda x:x / S2, range(1, len(values2) + 1))), values2, '--', color = common.line_plot_cols[1], label = r'$S=200$')
   plt.plot(list(map(lambda x:x / S3, range(1, len(values3) + 1))), values3, ':', color = common.line_plot_cols[2], label = r'$S=100$')
   plt.ylabel("TVD")
   plt.ylim(0, 2)
   plt.legend(loc='upper right')
   if S_squared:
      plt.xlabel(r"$\text{Iterations}/S^2$")     
   else:
      plt.xlabel(r"$\text{Iterations}/S$")
      
   if plot_no == 1 or plot_no == 3 or plot_no == 4:
      plt.xlim(0, 80)
   else:
      plt.xlim(0, (len(values3)/S3))
   
common.plot_figure(2, 2)
plot_fig4_2_subplot(1, TVD3, TVD2, TVD1)
plot_fig4_2_subplot(2, TVD3, TVD2, TVD1, True)
plot_fig4_2_subplot(3, TVD3n, TVD2n, TVD1n)
plot_fig4_2_subplot(4, TVD3l, TVD2l, TVD1)
common.save_figure("fig4-2-TVD_nonrev.pdf")

####################################################################
# Figure 4.3                                                       #
####################################################################

########GUSTAFSON EXAMPLE

common.seed(9)

##1-D
## input theta,p -- current state and sigma normal variance
## and logpi the log-density we wish to sample from
def gustafson_iteration(theta, p, sigma, logpi):
    ##proposal
    z = np.abs(np.random.normal(0, 1)*sigma)
    theta_dash = theta + p * z
    
    #acceptance probability
    alpha = np.exp(logpi(theta_dash) - logpi(theta))
    if np.random.uniform() < alpha:
        return theta_dash, p
    else:
        return theta, -p

#RW-Metropolis, one iteration
def RWMH_1d_iteration(theta, sigma, logpi):
    ##proposal
    z = np.random.normal(0, sigma)
    theta_dash = theta + z
    
    #acceptance probability
    alpha = np.exp(logpi(theta_dash) - logpi(theta))
    if np.random.uniform() < alpha:
        #acceptance
        return theta_dash
    else:
        #rejection then flip direction
        return theta

def gustafson(theta0, p0, N, sigma, logpi):
    Gstate = np.zeros(N)
    ##Gustafson

    state = (theta0, p0)
    for i in range(N):
        state = gustafson_iteration(state[0], state[1], sigma, logpi)
        Gstate[i] = state[0]
    
    return Gstate

####multiple-d version of Gustafson
def gustafson_d(theta0, p0, N, sigma, logpi, refresh):
  d = len(theta0)
  Gstate = np.zeros((N, d))
  
  ##Gustafson
  theta = theta0
  p = p0
  for i in range(N):
    if (i + 1) % refresh == 0:
      p_new = np.random.normal(size=d)
      p = p_new/np.sqrt(sum(p_new**2))
    
    theta, p = gustafson_iteration(theta, p, sigma, logpi)
    Gstate[i,] = theta
  
  return Gstate

def RWMH(theta0, N, sigma, logpi):
    MHstate = np.zeros(N)
    
    theta = theta0
    for i in range(N):
        theta = RWMH_1d_iteration(theta, sigma, logpi)
        MHstate[i] = theta
    
    return MHstate

##Target is Gaussian
def logpi(x):
  return -0.5 * x**2

def RWMH_iteration(theta, sigma, logpi):
  
  ##proposal
  z = np.random.normal(size=len(theta)) * sigma
  theta_dash = theta + z
 
  #acceptance probability
  alpha = np.exp(logpi(theta_dash) - logpi(theta))
  
  if(np.random.uniform(size=1) < alpha):
    #acceptance
    return theta_dash
  else:
    #rejection then flip direction
    return theta
  
def RWMH_d(theta0, N, sigma, logpi):
  d = len(theta0)
  MHstate = np.zeros((N, d))
  
  ##MH
  theta=theta0
  for i in range(N):
    theta = RWMH_iteration(theta, sigma, logpi)
    MHstate[i,]=theta
  
  return(MHstate)

def acc_rate(x):
    xa = x[1:]
    xb = x[:-1]
    return 1 - np.mean(xa == xb)

def normalize(x):
    return x / np.sqrt(np.sum(x**2))

## Discrete bouncy particle sampler
## Takes a variance-covariance preconditioner
##  which is the identity for the book example
def DBPS(nits, x0, u0, delta, kappa, V, lp_fn, glp_fn, print_freq=0):
    
    d = len(x0)
    Xs = np.zeros((nits, d+1))
    Us = np.zeros((nits, d))
    sqV = np.linalg.cholesky(V)
    sqVinv = np.linalg.inv(sqV)
    xc = sqVinv @ x0
    uc = sqVinv @ u0
    lpc = lp_fn(sqV @ xc)
    nbounce = 0
    nrej = 0
    dpsum = 0
    uab = uc
    for i in range(nits):
        xp = xc + uc * delta
        lpp = lp_fn(sqV @ xp)
        lr1 = lpp - lpc
        if np.log(np.random.uniform()) > lr1:
            dpsum += np.sum(uc * uab)**2 / (np.sum(uc**2) * np.sum(uab**2))
            lalphafw = min(0, lr1)
            g = glp_fn(sqV @ xp)
            g = g / np.sqrt(np.sum(g**2))
            upp = uc - 2 * g * np.sum(uc * g)
            xpp = xp + upp * delta
            lppp = lp_fn(sqV @ xpp)
            lalphabw = min(0, lpp - lppp)
            lr2 = lppp - lpc + np.log(1 - np.exp(lalphabw)) - np.log(1 - np.exp(lalphafw))
            if np.log(np.random.uniform()) > lr2:
                uc = -uc
                nrej += 1
            else:
                uc = upp
                xc = xpp
                lpc = lppp
                nbounce += 1
            uab = uc
        else:
            xc = xp
            lpc = lpp
        Xs[i, :] = np.concatenate((sqV @ xc, [lpc]))
        if common.VERBOSE and print_freq > 0:
            if print_freq * (i // print_freq) == i:
                print(i)
                print(Xs[i, :])
        z = np.random.normal(0, 1, d)
        if kappa > 0.0:
            uctil = uc
            uctilnorm = normalize(uctil)
            z = np.random.normal(0, 1, d)
            z = normalize(z - np.sum(z * uctilnorm) * uctilnorm)
            uctil = (uctil + np.sqrt(kappa * delta) * z) / np.sqrt(1 + kappa * delta)
            uctil = normalize(uctil)
            uc = uctil
        Us[i, :] = sqV @ uc
       
    dpmean = dpsum / (nbounce + nrej)
    
    if common.VERBOSE:
        print(f"nbounce={nbounce}")
        print(f"nrej={nrej}")        
        print(f"dpmean={dpmean}")
        
    return {"fbounce": nbounce / nits, "frej": nrej / nits, "dpmean": dpmean, "X": Xs, "U": Us}

def circle_log_density(x):
    return -50 * (np.sum(x**2) - 1)**2

def circle_grad_log_density(x):
    return -200 * (np.sum(x**2) - 1) * x

##RUN Methods
N = 10000

theta0=-3

fig, ax = common.plot_figure(2, 2)
outMH = RWMH(theta0, N, 0.1,logpi)
outG = gustafson(theta0, -1, N, 0.1, logpi)
plt.subplot(2, 2, 1)
plt.plot(range(1000), outMH[:1000], linestyle="solid", color=common.line_plot_cols[0], linewidth=2)
plt.title("RWMH")
plt.xlabel("k")
plt.ylabel(r"$\theta_k$")
plt.ylim(-3, 3)

plt.subplot(2, 2, 2)
plt.plot(range(1000), outG[:1000], linestyle="solid", color=common.line_plot_cols[0], linewidth=2)
plt.title("Gustafson")
plt.xlabel(r"$k$")
plt.ylabel(r"$\theta_k$")
plt.ylim(-3, 3)

sm.graphics.tsa.plot_acf(outMH, ax=ax[1, 0], markersize=1.5)
sm.graphics.tsa.plot_acf(outG, ax=ax[1, 1], markersize=1.5)

for plot_no in range(2):
   ax[1, plot_no].set_xlabel('Lag')
   ax[1, plot_no].set_ylabel('ACF')
   ax[1, plot_no].set_title('')
   ax[1, plot_no].set_ylim(-0.3, 1.1)
   
common.save_figure("fig4-3-1dcomparison_sigma_small.pdf")

####################################################################
# Figure 4.4                                                       #
####################################################################

common.seed(6)

common.plot_figure(2, 2)
N=1000

fig, ax = common.plot_figure(2, 2)
outMH = RWMH(theta0, N, 2.38, logpi)
outG = gustafson(theta0, -1, N, 2.38, logpi)
plt.subplot(2, 2, 1)
plt.plot(range(100), outMH[:100], linestyle="solid", color=common.line_plot_cols[0], linewidth=2)
plt.title("RWMH")
plt.xlabel(r"$k$")
plt.ylabel(r"$\theta_k$")
plt.ylim(-3, 2)

plt.subplot(2, 2, 2)
plt.plot(range(100), outG[:100], linestyle="solid", color=common.line_plot_cols[0], linewidth=2)
plt.title("Gustafson")
plt.xlabel(r"$k$")
plt.ylabel(r"$\theta_k$")
plt.ylim(-3, 2)

sm.graphics.tsa.plot_acf(outMH, ax=ax[1, 0], markersize=1.5)
sm.graphics.tsa.plot_acf(outG, ax=ax[1, 1], markersize=1.5)

for plot_no in range(2):
   ax[1, plot_no].set_xlabel('Lag')
   ax[1, plot_no].set_ylabel('ACF')
   ax[1, plot_no].set_title('')
   ax[1, plot_no].set_ylim(-0.2, 1.1)
   
common.save_figure("fig4-4-1dcomparison_sigma_large.pdf")

####################################################################
# Figure 4.5                                                       #
####################################################################

x_grid = np.arange(-1.2, 1.21, 0.01)
y_grid = np.arange(-1.2, 1.21, 0.01)
log_pi_grid = np.zeros((len(x_grid), len(y_grid)))
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        log_pi_grid[i, j] = circle_log_density(np.array([x_grid[i], y_grid[j]]))

common.seed(4)
nits = 50000

common.plot_figure(book_scale = 0.7)
plt.imshow(np.exp(log_pi_grid), extent=[-1.2, 1.2, -1.2, 1.2], origin="lower", cmap="YlOrRd")
out = gustafson_d(np.array([0, 0]), np.array([1, 1]), nits, sigma=0.01, logpi=circle_log_density, refresh=10)
plt.plot(out[:,0], out[:,1], color=common.black, linewidth=0.5)
common.save_figure("fig4-5-circle_gustafson.pdf")

####################################################################
# Figure 4.6                                                       #
####################################################################

common.seed(4)

common.plot_figure(book_scale = 0.7)
plt.imshow(np.exp(log_pi_grid), extent=[-1.2, 1.2, -1.2, 1.2], origin="lower", cmap="YlOrRd")
out = RWMH_d(np.array([0, 0]), nits, sigma=0.01, logpi=circle_log_density)
plt.plot(out[:,0], out[:,1], color=common.black, linewidth=0.5)
common.save_figure("fig4-6-circle_RWMH.pdf")

####################################################################
# Figure 4.8                                                       #
####################################################################

common.plot_figure(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(np.exp(log_pi_grid), extent=[-1.2, 1.2, -1.2, 1.2], origin="lower", cmap="YlOrRd")
V = np.eye(2)
out = DBPS(nits // 10, np.array([0, 0]), np.array([1, 1]), delta=0.01, kappa=3, V=V, lp_fn=circle_log_density, glp_fn=circle_grad_log_density, print_freq=1000)
plt.plot(out["X"][:, 0], out["X"][:, 1], color=common.black, linewidth=0.5)

plt.subplot(1, 2, 2)
plt.imshow(np.exp(log_pi_grid), extent=[-1.2, 1.2, -1.2, 1.2], origin="lower", cmap="YlOrRd")
V = np.eye(2)
out = DBPS(nits, np.array([0, 0]), np.array([1, 1]), delta=0.01, kappa=3, V=V, lp_fn=circle_log_density, glp_fn=circle_grad_log_density, print_freq=1000)
plt.plot(out["X"][:, 0], out["X"][:, 1], color=common.black, linewidth=0.5)
common.save_figure("fig4-8-DBPScirc.pdf")

####################################################################
# Figure 4.7                                                       #
####################################################################

############
## Figure showing the bounce used in the DBPS
############

lwdpi = 2
lwdtangent = 1.5
ltytangent = "dashed"
lwdpath = 1.0
ltypath = "solid"
lwdarrow = 1
eps = 0.1
font_size = 10 #8

common.plot_figure(book_scale = 0.7)
plt.xlim(-0.7, 0.4)
plt.ylim(-0.15, 1.0)

plt.plot([-0.5-eps, 0.22+eps], [-eps, 0.72+eps], color="white")
#plt.plot([-0.5-eps, 0.22+eps], [-eps, 0.72+eps], color=common.black, marker="o", markersize=0)
xs = np.linspace(-1, 1, 1000)
ys = xs**2 / 2
plt.plot(xs, ys, linestyle="solid", linewidth=lwdpi, color=common.black)
plt.plot([-2, 0], [2, 0], linestyle=ltypath, linewidth=lwdpath, color=common.black)

plt.plot([-0.45-eps/2, -0.45], [0.45, 0.45], linestyle="solid", linewidth=lwdpath, color=common.black)
plt.plot([-0.45, -0.45], [0.45+eps/2, 0.45], linestyle="solid", linewidth=lwdpath, color=common.black)

plt.plot([-0.15-eps/2, -0.15], [0.15, 0.15], linestyle="solid", linewidth=lwdpath, color=common.black)
plt.plot([-0.15, -0.15], [0.15+eps/2, 0.15], linestyle="solid", linewidth=lwdpath, color=common.black)

plt.plot([-1, 1], [0, 0], linestyle=ltytangent, linewidth=lwdtangent, color=common.black)
plt.plot([0.3, 0], [0.3, 0], linestyle=ltypath, linewidth=lwdpath, color=common.blue)

plt.scatter([0.3], [0.3], marker="o", s=50, color=common.blue)
plt.scatter([0.3], [0.3], marker="o", s=30, color="white", zorder=3)

plt.plot([0.15-eps/2, 0.15], [0.15, 0.15], linestyle="solid", linewidth=lwdpath, color=common.blue)
plt.plot([0.15, 0.15], [0.15-eps/2, 0.15], linestyle="solid", linewidth=lwdpath, color=common.blue)

plt.scatter([-0.6, -0.3, 0], [0.6, 0.3, 0], marker="o", s=50, color=[common.black, common.black, common.red], zorder=3)
#plt.scatter([0], [0], marker="o", s=150, color=red)  ## go over it again

# Remove ticks
plt.xticks([])  
plt.yticks([]) 

## Labels
plt.text(-0.58, 0.67, r"$\theta_{k-1}$", fontsize=font_size)
plt.text(-0.3, 0.35, r"$\theta_k$", fontsize=font_size)
plt.text(0.0, 0.06, r"$\theta'$", fontsize=font_size, color=common.red)
plt.text(0.3, 0.35, r"$\theta''$", fontsize=font_size, color=common.blue)
plt.text(-0.44, 0.53, r"$\text{p}_{k-1}$", fontsize=font_size)
plt.text(-0.15, 0.23, r"$\text{p}_k$", fontsize=font_size)
plt.text(0.21, 0.15, r"$-\text{p}''$", fontsize=font_size, color=common.blue)
#plt.axis("off")
common.save_figure("fig4-7-DBPS_bounce.pdf")

