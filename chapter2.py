import common
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

## Generic functions

def rwm(nits, theta0, log_pi, lambda_val, qV=None, *args):
   
    # Get length of vector, length 1 if a single number
    if np.isscalar(theta0):
        d = 1
    else:
        d = len(theta0)
       
    if qV is None:
        A = np.eye(d)
    else:
        ## If d=1, for Z~N(0,1),     X=mu+aZ ~ N(mu,a^2).
        ## If d>1, for Z~MVN(0,I_d), X=mu+AZ ~ N(mu,AA').
        ## The Cholesky decomposition finds A such that A'A=qV.
        A = np.linalg.cholesky(qV).T
        
    store = np.zeros((nits+1, d+1))
    psis = np.zeros((nits, d))
    nacc = 0
    theta_curr = theta0
    log_pi_curr = log_pi(theta_curr, *args)        
    store[0, :] = np.append(theta_curr, log_pi_curr)
    
    for i in range(nits):
        psi = theta_curr + lambda_val * np.matmul(A, np.random.randn(d))      
        psis[i, :] = psi
        log_pi_prop = log_pi(psi, *args)
        log_alpha = log_pi_prop - log_pi_curr
        if np.log(np.random.uniform()) < log_alpha:
            theta_curr = psi
            log_pi_curr = log_pi_prop
            nacc += 1
          
        store[i+1, :] = np.append(theta_curr, log_pi_curr)
        
    return {'acc': nacc/nits, 'store': store, 'psis': psis}

def log_pi_GaussIso(x):
    return -0.5 * np.sum(x * x)

## **To get the precise figures in the book I had to manually adjust the size of the plot window, either reducing the height (Fig 2.1 and 2.2) or the width (Fig 2.3).**

####################################################################
# Figure 2.1                                                       #
####################################################################

###############################
## Short RWM runs - trace plots and histograms
###############################

## Run algorithms for first set of plots
common.seed(1234678)
rwm1 = rwm(1000, 20, log_pi_GaussIso, 0.2)
rwm2 = rwm(1000, 1, log_pi_GaussIso, 0.2)
rwm3 = rwm(1000, 1, log_pi_GaussIso, 2)
rwm4 = rwm(1000, 1, log_pi_GaussIso, 20)

## Produce first set of trace plots and histograms
common.plot_figure(1, 3)
plt.subplot(1, 3, 1)
plt.plot(range(1001), rwm1['store'][:, 0], '-', linewidth=2, c = common.line_plot_cols[0])
plt.xlabel(r'$k$')
plt.title(r'$\lambda = 0.2$')
plt.subplot(1, 3, 2)
plt.plot(range(1001), rwm3['store'][:, 0], '-', linewidth=2, c = common.line_plot_cols[0])
plt.xlabel(r'$k$')
plt.title(r'$\lambda = 2$')
plt.subplot(1, 3, 3)
plt.plot(range(1001), rwm4['store'][:, 0], '-', linewidth=2, c = common.line_plot_cols[0])
plt.xlabel(r'$k$')
plt.title(r'$\lambda = 20$')
plt.tight_layout()
common.save_figure('fig2-1-rwm.pdf')

if common.VERBOSE:
    print([rwm1['acc'], rwm3['acc'], rwm4['acc']]) ## get the acceptance rates

####################################################################
# Figure 2.2                                                       #
####################################################################

common.plot_figure(1, 3)
plt.subplot(1, 3, 1)
plt.hist(rwm1['store'][300:1001, 0], bins=30, density=True, color = common.histogram_cols[0], edgecolor='k', linewidth=0.5)
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
#plt.title(r'$\lambda = 0.2$')
plt.subplot(1, 3, 2)
plt.hist(rwm3['store'][0:1001, 0], bins=30, density=True, color = common.histogram_cols[0], edgecolor='k', linewidth=0.5)
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
#plt.title(r'$\lambda = 2$')
plt.subplot(1, 3, 3)
plt.hist(rwm4['store'][0:1001, 0], bins=30, density=True, color = common.histogram_cols[0], edgecolor='k', linewidth=0.5)
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
#plt.title(r'$\lambda = 20$')
plt.tight_layout()
common.save_figure('fig2-2-rwmhist.pdf')

####################################################################
# Figure 2.3                                                       #
####################################################################

###############################
## Longer RWM runs and corresponding trace plots and then ACF plot
###############################

common.seed(1234583)

rwm5 = rwm(3000, np.random.normal(size=50), log_pi_GaussIso, 2/np.sqrt(50), None)
rwm6 = rwm(30000, np.random.normal(size=500), log_pi_GaussIso, 2/np.sqrt(500), None)

common.seed(12345783)

rwm5b = rwm(10000, np.random.normal(size=50), log_pi_GaussIso, 2/np.sqrt(50), None)
rwm6b = rwm(100000, np.random.normal(size=500), log_pi_GaussIso, 2/np.sqrt(500), None)

common.plot_figure(2, 1, width_scale = 2) #double width
plt.subplot(2, 1, 1)
plt.plot(rwm5['store'][:, 0], '-', linewidth=1, c = common.line_plot_cols[0])
plt.xlabel(r'$k$')
plt.ylabel(r'$\theta$')
plt.title(r'$d = 50$')
plt.ylim(-3, 3)

plt.subplot(2, 1, 2)
# Plot every 10th element
plt.plot(np.arange(10, len(rwm6['store'][:, 0]), 10), rwm6['store'][10::10, 0], '-', linewidth=1, c = common.line_plot_cols[0])
plt.xlabel(r'$k$')
plt.ylabel(r'$\theta$')
plt.title(r'$d = 500$')
plt.ylim(-3, 3)
common.save_figure('fig2-3-rwm05-rwm06.pdf')

####################################################################
# Figure 2.4                                                       #
####################################################################

lag50 = 300
lag500 = 3000
acfs50 = np.zeros(lag50 + 1)
acfs500 = np.zeros(lag500 + 1)

for i in range(50):
    acfs50 += acf(rwm5b['store'][:, i], nlags=lag50)

for i in range(100):    
    if common.VERBOSE and i % 10 == 0:
        print(i)
        
    acfs500 += acf(rwm6b['store'][:, i], nlags=lag500)

acfs50 /= 50
acfs500 /= 100

common.plot_figure(book_scale = 0.7)
plt.plot(np.arange(lag50 + 1), acfs50, ':', linewidth = 2, c = common.line_plot_cols[0], zorder=3)
plt.plot(np.arange(lag500 + 1) / 10, acfs500, '--', linewidth = 2, c = common.line_plot_cols[1], zorder=1)
plt.xlabel(r'lag $(d=50)$ or lag/10 $(d=500)$')
plt.ylabel(r'$\rho$')
#plt.title('Autocorrelation')
plt.legend([r'$d=50$', r'$d=500$'])
common.save_figure('fig2-4-AutoCord50d500.pdf')

if common.VERBOSE:
    print(np.sum(acfs50), np.sum(acfs500))

####################################################################
# Figure 2.5                                                       #
####################################################################

##############################
## HMC bunch of leapfrog steps
##############################

def lp(x, H):
    y = np.array(x).T
    y = y.reshape(y.shape[0], 1)
    return -0.5 * np.matmul(np.matmul(y.T, H), y)[0, 0]

def glp(x, H):
    y = np.array(x).T    
    return -np.matmul(H, y)

def Lleaps(x0, p0, L, eps, H):
    
    x = x0
    p = p0
    d = len(x0)
    g = glp(x, H)
 
    xs = np.zeros((L+1, d))
    ps = np.zeros((L+1, d))
    isapogee = np.zeros(L+1)
    xs[0, :] = x0
    ps[0, :] = p0
   
    currdotpos = (np.dot(np.squeeze(np.asarray(g)), np.squeeze(np.asarray(p))) > 0)
   
    hadnegdot = not currdotpos
    for i in range(L):
           
        p = p + 0.5 * eps * g
        x = x + eps * p
        g = glp(x, H)
        p = p + 0.5 * eps * g
        xs[i+1, :] = x
        ps[i+1, :] = p
     
        # g and p now in matrix format, take first row
        currdotpos = (np.dot(np.squeeze(np.asarray(g)), np.squeeze(np.asarray(p))) > 0)
   
        hadnegdot = hadnegdot or not currdotpos
        if currdotpos and hadnegdot:
            isapogee[i] = 1
            hadnegdot = False
    return {'xs': xs, 'ps': ps, 'isapogee': isapogee, 'npt': L+1}


## contour of log pi

H = np.diag([1, 12])
sds = 1 / np.sqrt(np.diag(H))
x0 = [-2, 1]
p0 = [1.8, 1.5]
Lleap = 25
length = 200
xs = np.linspace(-3, 3, num=length)
ys = np.linspace(-3, 3, num=length)
LP = np.zeros((length, length))

for i in range(length):
    for j in range(length):
        LP[i, j] = lp([xs[i], ys[j]], H)
        
a = Lleaps(x0, p0, Lleap, 0.1, H)

aisstart = np.append([1], np.zeros(a['npt']-1))
aisend = np.append(np.zeros(a['npt']-1), [1])
aismid = np.append([0], np.append(np.ones(a['npt']-2), [0]))

markers = ['o'] * len(a['xs'][:, 0])
markers[0] = '+'
markers[-1] = 'x'

## Simple plot of path forward from 
common.plot_figure(1, 1, book_scale = 0.7)
plt.imshow(-LP.T, extent=[-3, 3, -3, 3], cmap=common.get_heat_color_map(), origin='lower')
plt.scatter(a['xs'][0, 0], a['xs'][0, 1], c = common.black, s=100, marker = '+')
plt.scatter(a['xs'][-1, 0], a['xs'][-1, 1], c = common.black, s=80, marker = 'x')
plt.scatter(a['xs'][1:-1, 0], a['xs'][1:-1, 1], c = common.black, s=10, marker = 'o')
plt.quiver(a['xs'][0, 0], a['xs'][0, 1], a['ps'][0, 0]/3, a['ps'][0, 1]/3, angles='xy', scale_units='xy', scale=1, color=common.black, width=0.008)
plt.quiver(a['xs'][Lleap, 0], a['xs'][Lleap, 1], a['ps'][Lleap, 0]/3, a['ps'][Lleap, 1]/3, angles='xy', scale_units='xy', scale=1, color=common.black, width=0.008)
#plt.title('Lleapfrog Forward')
#plt.legend(-3,-1.95,c("current","proposed"),pch=c(3,4),cex=1.5,,bg="white")
plt.legend(['current', 'proposed'])
common.save_figure('fig2-5-LleapsForward.pdf')

####################################################################
# Figure 2.6                                                       #
####################################################################

##############################
## HMC autocorrelation plot
##############################

sigmas = np.array([1, 2, 3, 4, 5])
nsig = len(sigmas)
ts = np.linspace(0, max(sigmas) * 2 * np.pi, num=200)
linestyles = [":", "-.", "--", (0, (3, 1, 1, 1)),  (5, (10, 3)), (0, (3, 5, 1, 5, 1, 5))]
colours = [common.red, common.green, common.blue, common.cyan, common.magenta]

common.plot_figure(1, 1, width_scale = 1.5, pad = 3.5)
for i in range(nsig):
    corrs = np.cos(ts / sigmas[i])
    if i == 0:
        plt.plot(ts, corrs, linestyle = linestyles[i], linewidth=1, c = colours[i])
        mx = corrs
    else:
        plt.plot(ts, corrs, linestyle = linestyles[i], linewidth=1, c = colours[i])
        mx = np.maximum(mx, corrs)
        
plt.plot(ts, mx, 'k-', linewidth=3)
plt.xlabel(r'$T$')
plt.ylabel(r'$\mathsf{Cor}(\theta_0, \theta_T)$')
#plt.title('HMC Correlations')
plt.title(r'$\theta_i \sim \mathsf{N}(0, i^2), i=1,\ldots,5$')

common.save_figure('fig2-6-HMCcorr.pdf')



