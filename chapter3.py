import common

#Import required packages
#We're going to use the SGMCMCJax package, a stochastic gradient MCMC package based on JAX. Using JAX means that we can benefit from automatic differentiation.
#If you're interested in the SGMCMCJax pacakge, then the documentation can be found here https://sgmcmcjax.readthedocs.io/en/latest/index.html

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import norm
from sgmcmcjax.samplers import build_sgld_sampler, build_sgldCV_sampler, build_sghmc_sampler, build_sghmcCV_sampler

import optax
from sgmcmcjax.optimizer import build_optax_optimizer

import blackjax
import blackjax.sgmcmc.gradients as gradients

from itertools import cycle
from functools import partial

####################################################################
# Figure 3.1                                                       #
####################################################################

# Fig 3.1 left and right
# book_scale = 0.35
# common.plot_figure(1, 1, width_scale = 1, height_scale = 1, book_scale = 0.35, pad = 3) 
# ...
# common.save_figure("fig3-1a.pdf")
# common.plot_figure(1, 1, width_scale = 1, height_scale = 1, book_scale = 0.35, pad = 3) 
# ...
# common.save_figure("fig3-1b.pdf")

####################################################################
# Figure 3.2                                                       #
####################################################################

# define our likelihood and prior
def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)


# generate a dataset with N data points of dimension D. For JAX, we also need to generate a random key.
N, D = 10_000, 1
key = random.PRNGKey(0)
X_data = random.normal(key, shape=(N, D))

#Finding the mode - for SGLD with control variates we need to find the mode of the posterior. This requires a pre-compute step where we use an optimiser to find the mode of the posterior distribtion
#For this example, we're going to use the Adam optimiser. Other choices are available thorough JAX https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html

batch_size = int(0.1*N)  #data subsample size, i.e. 10%
dt_adam = 1e-3           #step size parameter
opt = optax.adam(learning_rate=dt_adam)
optimizer = build_optax_optimizer(opt, loglikelihood, logprior, (X_data,), batch_size)
Niters = 10_000
init_position = 1.0
opt_params, log_post_list = optimizer(key, Niters, init_position)

# build stochastic gradient MCMC samplers
dt = 1e-5 #step-size

#SGLD and SGLD-CV with a 1% subsample size
batch_size = int(0.01*N)
sgld_sampler_1 = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size, pbar = common.VERBOSE)
sgldcv_sampler_1 = build_sgldCV_sampler(dt, loglikelihood, logprior, (X_data,), batch_size, opt_params, pbar = common.VERBOSE)

#ULA with 100% of the data
batch_size = N
ula_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size, pbar = common.VERBOSE)

# Run the samplers
Nsamples = 10_000 #number of iterations

samples_sgld_1 = sgld_sampler_1(key, Nsamples, jnp.zeros(D))
samples_sgldcv_1 = sgldcv_sampler_1(key, Nsamples, jnp.zeros(D))

samples_ula = ula_sampler(key, Nsamples, jnp.zeros(D))

#True posterior - for this example we know the posterior is Gaussian so we specify its mean and variance
sigma_true = 1 #variance in the data likelihood
mu0 = 0        #prior mean
tau0 = jnp.sqrt(1) #prior standard deviation

mu_post = (jnp.sum(X_data)/sigma_true**2 + mu0/tau0**2)/(1/tau0**2 + N/sigma_true**2) #posterior mean
sigma_post = jnp.sqrt(1/(1/tau0**2 + N/sigma_true**2))                                #posterior standard deviation

################################################

grad_fn = gradients.grad_estimator(logprior, loglikelihood, N)

grad_cv = grad_fn(opt_params, X_data)

n = int(0.1*N)
thetas = jnp.arange(-0.5,0.5,0.1)
n_iter = 1000

#Calculate the variance of SGLD gradients
grads_sgld = jnp.zeros((len(thetas), n_iter))
for i in range(len(thetas)):
  for j in range(n_iter):
      key, batch_key = random.split(key, 2)
      idx = random.permutation(batch_key,jnp.arange(N),independent=True)[:n]
      grad = (N/n)*(grad_fn(thetas[i], X_data[idx]))
      grads_sgld = grads_sgld.at[i,j].set(grad)

#Calculate the variance of the gradients for SGLD-CV
grads_sgld_cv = jnp.zeros((len(thetas),n_iter))
for i in range(len(thetas)):
  for j in range(n_iter):
      key, batch_key = random.split(key, 2)
      idx = random.permutation(batch_key,jnp.arange(N),independent=True)[:n]
      grad = grad_cv + (N/n)*(grad_fn(thetas[i], X_data[idx])-grad_fn(opt_params,X_data[idx]))
      grads_sgld_cv = grads_sgld_cv.at[i,j].set(grad)

# Plot the histogram.
xmin, xmax = samples_sgld_1[:,0].min(), samples_sgld_1[:,0].max()
x = jnp.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_post, sigma_post)

fig, axs = common.plot_figure(2, 2)
axs[0,0].plot(thetas,jnp.log10(jnp.var(grads_sgld,axis=1)), '-', label='SGLD', c = common.line_plot_cols[0])
axs[0,0].plot(thetas,jnp.log10(jnp.var(grads_sgld_cv,axis=1)), '--',label='SGLD-CV', c = common.line_plot_cols[1])
axs[0,0].set_xlim([-0.5,0.5])
axs[0,0].set_title('Variance of stochastic gradients')
axs[0,0].set_xlabel(r'$\mathrm{\theta}$')
axs[0,0].legend()
axs[0,0].set_ylabel('Variance (log10)')
axs[1,0].hist(samples_sgld_1[:,0], bins=100, density=True, alpha=0.6, color=common.histogram_cols[0], edgecolor='k', linewidth=0.1)
axs[1,0].plot(x, p, linewidth=2, c = common.line_plot_cols[0])
axs[1,0].set_title('SGLD sampler 1% data')
axs[1,0].set_xlabel(r'$\mathrm{\theta}$')
axs[0,1].hist(samples_ula[:,0], bins=50, density=True, alpha=0.6, color=common.histogram_cols[0], edgecolor='k', linewidth=0.1)
axs[0,1].plot(x, p, linewidth=2, c = common.line_plot_cols[0])
axs[0,1].set_xlabel(r'$\mathrm{\theta}$')
axs[0,1].set_title('ULA sampler')
axs[1,1].hist(samples_sgldcv_1[:,0], bins=50, density=True, alpha=0.6, color=common.histogram_cols[0], edgecolor='k', linewidth=0.1)
axs[1,1].plot(x, p, linewidth=2, c = common.line_plot_cols[0])
axs[1,1].set_xlabel(r'$\mathrm{\theta}$')
axs[1,1].set_title('SGLD-CV sampler 1% data')

common.save_figure("fig3-2-ULA-SGLD.pdf")

####################################################################
# Figure 3.3                                                       #
####################################################################

#Function to calculate the Wasserstein_2 distance between two Gaussians
def w2_distance(mu1, Sigma1, mu2, Sigma2):
  """Calculates the Wasserstein-2 distance between two multivariate Gaussian distributions.

  Args:
    mu1: A numpy array representing the mean of the first distribution.
    sigma1: A numpy array representing the covariance matrix of the first distribution.
    mu2: A numpy array representing the mean of the second distribution.
    sigma2: A numpy array representing the covariance matrix of the second distribution.

  Returns:
    A float representing the Wasserstein-2 distance between the two distributions.
  """

  # Compute the difference between the means of the two distributions.
  mu_diff = mu1 - mu2

  # Compute the square root of the sum of the variances of the two distributions.
  sigma_sum = Sigma1 + Sigma2 -2*(jnp.matmul(jnp.matmul(jnp.linalg.cholesky(Sigma1),Sigma2),jnp.linalg.cholesky(Sigma1)))
  sigma_sum_sqrt = jnp.linalg.cholesky(sigma_sum)

  # Compute the Wasserstein-2 distance.
  wasserstein_distance = jnp.sqrt(jnp.sum(mu_diff**2) + jnp.trace(sigma_sum_sqrt))

  return wasserstein_distance

# Function to plot a subfigure for figure 3.3
def plot_fig3_3_subplot(a, b, c):
    #a, b, c = 1, 10, 0 #params for the covariance matrix - THIS PRODUCES THE LEFT PANEL
    #a, b, c = 1, 10, 3 #params for the covariance matrix - THIS PRODUCES THE RIGHT PANEL

    # generate a dataset with N data points of dimension D. For JAX, we also need to generate a random key.
    N, D = 1000, 2
    key = random.PRNGKey(0)
    theta = jnp.array([[1,0]])
    Sigma = jnp.array([[a,c],[c,b]])
    X_data = random.multivariate_normal(key, mean=theta, cov=Sigma, shape=(N,1))

    Sigma_post = jnp.linalg.inv(N*jnp.linalg.inv(Sigma)+1)   #posterior variance
    mu_post = jnp.dot(Sigma_post,jnp.dot(jnp.linalg.inv(Sigma),jnp.sum(X_data,axis=0)[0])) #posterior mean

    #Check the posterior correlation
    Dinv=jnp.diag(1/jnp.sqrt(jnp.diag(Sigma_post)))
    
    if common.VERBOSE:
        print(Dinv@Sigma_post@Dinv)

    #Finding the mode - for SGLD with control variates we need to find the mode of the posterior. This requires a pre-compute step where we use an optimiser to find the mode of the posterior distribtion
    #For this example, we're going to use the Adam optimiser. Other choices are available thorough JAX https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html

    batch_size = int(0.1*N)  #data subsample size, i.e. 10%
    dt_adam = 1e-3           #step size parameter
    opt = optax.adam(learning_rate=dt_adam)
    
    # define our likelihood and prior
    def loglikelihood(theta, x):
        return jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(x,mean=theta,cov=Sigma))

    def logprior(theta):
        return -0.5*jnp.dot(theta, theta)

    optimizer = build_optax_optimizer(opt, loglikelihood, logprior, (X_data,), batch_size)
    Niters = 10_000
    init_position = jnp.array([0.0,1.0])
    opt_params, log_post_list = optimizer(key, Niters, init_position)

    # build stochastic gradient MCMC samplers
    dt = 1e-5 #step-size

    sgldcv_sampler = build_sgldCV_sampler(dt, loglikelihood, logprior, (X_data,), batch_size, opt_params, pbar = common.VERBOSE)

    # Run the samplers
    Nsamples = 10_000 #number of iterations

    samples_sgldcv = sgldcv_sampler(key, Nsamples, jnp.zeros(D))

    #Using the convergence results from Dalalyan and Karagulyan, let's try to find a reasonable stepsize parameter

    eigen_values = jnp.real(jnp.linalg.eigvals(Sigma_post))
    h_opt = jnp.min(eigen_values)/(jnp.max(eigen_values)+1)

    # Run the sampler with the recommended step-size parameter
    sgldcv_sampler = build_sgldCV_sampler(h_opt, loglikelihood, logprior, (X_data,), batch_size, opt_params, pbar = common.VERBOSE)

    Nsamples = 10_000 #number of iterations
    samples_sgldcv_opt = sgldcv_sampler(key, Nsamples, jnp.zeros(D))

    #Let's compare a range of stepsize parameters and test the accuracy of the samplers


    Nsamples = 1000 #number of iterations
    stepsizes = [10e-6,10e-5,10e-4,0.5*10e-3,10e-2]
    recorded_w2_dists=[]

    for h in stepsizes:
      sgldcv_sampler = build_sgldCV_sampler(h/2, loglikelihood, logprior, (X_data,), batch_size, opt_params, pbar = common.VERBOSE)
      samples_sgldcv = sgldcv_sampler(key, Nsamples, jnp.zeros(D))
      recorded_w2_dists.append(w2_distance(mu_post,Sigma_post, jnp.mean(samples_sgldcv,axis=0), jnp.cov(samples_sgldcv,rowvar=False)))
    recorded_w2_dists = jnp.array(recorded_w2_dists)

    ################################

    #Plot the results, settings NaNs to a large value
    recorded_w2_dists = recorded_w2_dists.at[jnp.isnan(recorded_w2_dists)].set(1.0)
    plt.plot(jnp.log10(jnp.array(stepsizes)),recorded_w2_dists, '-', color = common.line_plot_cols[0])
    plt.scatter(jnp.log10(h_opt),w2_distance(mu_post,Sigma_post, jnp.mean(samples_sgldcv_opt,axis=0), jnp.cov(samples_sgldcv_opt,rowvar=False)), color=common.scatter_cols[0])
    plt.ylabel('Wasserstein-2 Distance')
    plt.xlabel('Step size (log10)')

#Do the plot for figure 3.3
common.plot_figure(1, 2)
plt.subplot(1, 2, 1)
plot_fig3_3_subplot(1, 10, 0)
plt.subplot(1, 2, 2)
plot_fig3_3_subplot(1, 10, 3)
common.save_figure("fig3-3-Wasserstein-2-distance.pdf")

####################################################################
# Figure 3.4                                                       #
####################################################################


#Functions to generate data for logistic regression model + functions to evaluate the model's loglikelihood and logprior

def genCovMat(key, d, rho):
    Sigma0 = np.diag(np.ones(d))
    for i in range(1, d):
        for j in range(0, i):
            Sigma0[i, j] = (random.uniform(key) * 2 * rho - rho) ** (i - j)
            Sigma0[j, i] = Sigma0[i, j]

    return jnp.array(Sigma0)


def logistic(theta, x):
    return 1 / (1 + jnp.exp(-jnp.dot(theta, x)))

batch_logistic = jit(vmap(logistic, in_axes=(None, 0)))
batch_benoulli = vmap(random.bernoulli, in_axes=(0, 0))

def gen_data(key, dim, N):
    """
    Generate data with dimension `dim` and `N` data points

    Parameters
    ----------
    key: uint32
        random key
    dim: int
        dimension of data
    N: int
        Size of dataset

    Returns
    -------
    theta_true: ndarray
        Theta array used to generate data
    X: ndarray
        Input data, shape=(N,dim)
    y_data: ndarray
        Output data: 0 or 1s. shape=(N,)
    """
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    rho = 0.0
    if common.VERBOSE:
        print(f"generating data, with N={N} and dim={dim}")
    theta_true = random.normal(subkey1, shape=(dim,))
    covX = genCovMat(subkey2, dim-1, rho)
    X = jnp.dot(random.normal(subkey3, shape=(N, dim-1)), jnp.linalg.cholesky(covX))
    Xdata = jnp.empty((N,dim))
    Xdata = Xdata.at[:,1:].set(X)
    Xdata = Xdata.at[:, 0].set(jnp.ones(N))

    p_array = batch_logistic(theta_true, Xdata)
    keys = random.split(key, N)
    y_data = batch_benoulli(keys, p_array).astype(jnp.int32)
    return theta_true, Xdata, y_data

@jit
def loglikelihood(theta, x_val, y_val):
    return -logsumexp(jnp.array([jnp.zeros(y_val.shape), (1.0 - 2.0 * y_val) * jnp.dot(theta, x_val)]))

@jit
def logprior(theta):
    return -0.5 * jnp.dot(theta, theta)

@jit
def logposterior(theta):
    return jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X,y_data)) + logprior(theta)

def log_loss(theta,x,y):
    return -np.mean(y*np.log(logistic(theta,x))+(1-y)*np.log(1-logistic(theta,x)+1e-6))

#Generate the data

key = random.PRNGKey(42)
dim = 10
N = 100000

theta_true, X, y_data = gen_data(key, dim, N)
data = (X, y_data)

# generating data, with N=100000 and dim=10

# Build the samplers
batch_size = int(0.1*N)
dt = 1/N

#SGLD
sgld_sampler = build_sgld_sampler(dt, loglikelihood, logprior, data, batch_size, pbar = common.VERBOSE)
sgld_sampler = partial(jit, static_argnums=(1,))(sgld_sampler) # jit compile the sampler

#SGHMC
sghmc_sampler = build_sghmc_sampler(dt, 5, loglikelihood, logprior, data, batch_size, pbar = common.VERBOSE)
sghmc_sampler = partial(jit, static_argnums=(1,))(sghmc_sampler) # jit compile the sampler

# %%time

# run sampler
Nsamples = 1000
sgld_samples = sgld_sampler(key, Nsamples, jnp.zeros(dim))
sghmc_samples = sghmc_sampler(key, Nsamples, jnp.zeros(dim))

idx = 0
plt.plot(sgld_samples[:, idx])
plt.plot(sghmc_samples[:,idx])
plt.axhline(y=theta_true[idx])

#Set-up code for running MCMC on the full data using the BlackJax package


def mcmc_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

##########################################################

#Now that the basic implementation is working, let's test these samplers over increasing dimensions

D = [100, 200, 300, 400, 500]  #dimension
N = 10000                #number of data points
Nrepeats = 5             #monte carlo repreats

mean_error_sgld = np.empty([Nrepeats,len(D)]); var_error_sgld = np.empty([Nrepeats,len(D)])
mean_error_sgldCV = np.empty([Nrepeats,len(D)]); var_error_sgldCV = np.empty([Nrepeats,len(D)])
mean_error_sghmc = np.empty([Nrepeats,len(D)]); var_error_sghmc = np.empty([Nrepeats,len(D)])
mean_error_sghmcCV = np.empty([Nrepeats,len(D)]); var_error_sghmcCV = np.empty([Nrepeats,len(D)])

#mcmc_ll = np.empty([len(D)])
#sgld_ll = np.empty([Nrepeats,len(D)])
#sgldCV_ll = np.empty([Nrepeats,len(D)])
#sghmc_ll = np.empty([Nrepeats,len(D)])
#sghmcCV_ll = np.empty([Nrepeats,len(D)])

mcmc_logpred = np.empty([len(D)]);
sgld_logpred = np.empty([Nrepeats,len(D)]);
sgldCV_logpred = np.empty([Nrepeats,len(D)]);
sghmc_logpred = np.empty([Nrepeats,len(D)]);
sghmcCV_logpred = np.empty([Nrepeats,len(D)]);

j=0
for d in D:
  #generate dataset
  key = random.PRNGKey(543210)
  theta_true, X, y_data = gen_data(key, d, N)
  X_train, y_train = X[:int(0.8*N)], y_data[:int(0.8*N)]
  data_train = (X_train,y_train)
  X_test, y_test = X[int(0.8*N):], y_data[int(0.8*N):]


  @jit
  def logposterior(theta):
    return jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_train,y_train)) + logprior(theta)

  @jit
  def logpred_fn(samples):
      logpred=0
      for theta in samples:
        logpred += jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test))
      return logpred/samples.shape[0]


  # Adam - get MAP for control variate algorithms
  batch_size_adam = int(0.1 * N)
  dt_adam = 1e-3
  opt = optax.adam(learning_rate=dt_adam)
  optimizer = build_optax_optimizer(opt, loglikelihood, logprior, data_train, batch_size_adam)
  Niters = 10_000
  opt_params, log_post_list = optimizer(key, Niters, theta_true)

  #Compile samplers
  batch_size = int(0.1*N)
  dt = 1/N

  #SGLD
  sgld_sampler = build_sgld_sampler(dt, loglikelihood, logprior, data_train, batch_size, pbar=False)
  sgld_sampler = partial(jit, static_argnums=(1,))(sgld_sampler) # jit compile the sampler

  #SGLD-CV
  sgldCV_sampler = build_sgldCV_sampler(dt, loglikelihood, logprior, data_train, batch_size, opt_params, pbar=False)
  sgldCV_sampler = partial(jit, static_argnums=(1,))(sgldCV_sampler) # jit compile the sampler

  #SGHMC
  sghmc_sampler = build_sghmc_sampler(1e-5, 15, loglikelihood, logprior, data_train, batch_size, pbar=False)
  sghmc_sampler = partial(jit, static_argnums=(1,))(sghmc_sampler) # jit compile the sampler

  #SGHMC-CV
  sghmcCV_sampler = build_sghmcCV_sampler(1e-5, 15, loglikelihood, logprior, data_train, batch_size, opt_params, pbar=False)
  sghmcCV_sampler = partial(jit, static_argnums=(1,))(sghmcCV_sampler) # jit compile the sampler


  # Run samplers
  Nsamples = 10000

  #Run MCMC on the full data to provide a baseline - starting with a warmup phase
  warmup = blackjax.window_adaptation(blackjax.nuts, logposterior)

  key, init_key, warmup_key, sample_key = jax.random.split(key, 4)
  init_params = jax.random.multivariate_normal(init_key, jnp.zeros(d), jnp.eye(d))

  (initial_states, tuned_params), _ = warmup.run(warmup_key, init_params, 1000)

  #Run full MCMC
  nuts = blackjax.nuts(logposterior, **tuned_params)
  states = mcmc_loop(sample_key, nuts.step, initial_states, Nsamples//2)
  mcmc_samples = states.position

  #mcmc_ll[j] = log_loss(mcmc_samples[:Nsamples//2],X_test.T,y_test)
  mcmc_logpred[j] = np.mean([jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test)) for theta in mcmc_samples[:Nsamples//2]])
   
  for i in range(Nrepeats):
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

    #SGMCMC samplers
    #SGLD
    sgld_samples = sgld_sampler(subkey1, Nsamples, opt_params)
    var_error_sgld[i,j] = jnp.mean(((jnp.var(sgld_samples[:Nsamples//2],axis=0)-jnp.var(mcmc_samples[:Nsamples//2],axis=0))**2))
    mean_error_sgld[i,j] = jnp.mean(((jnp.mean(sgld_samples[:Nsamples//2],axis=0)-jnp.mean(mcmc_samples[:Nsamples//2],axis=0))**2))
    #sgld_ll[i,j] = log_loss(sgld_samples[:Nsamples//2],X_test.T,y_test)
    sgld_logpred[i,j] = np.mean([jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test)) for theta in sgld_samples[:Nsamples//2]])
   

    #SGLD-CV
    sgldCV_samples = sgldCV_sampler(subkey2, Nsamples, opt_params)
    var_error_sgldCV[i,j] = jnp.mean(((jnp.var(sgldCV_samples[:Nsamples//2],axis=0)-jnp.var(mcmc_samples[:Nsamples//2],axis=0))**2))
    mean_error_sgldCV[i,j] = jnp.mean(((jnp.mean(sgldCV_samples[:Nsamples//2],axis=0)-jnp.mean(mcmc_samples[:Nsamples//2],axis=0))**2))
    #sgldCV_ll[i,j] = log_loss(sgldCV_samples[:Nsamples//2],X_test.T,y_test)
    sgldCV_logpred[i,j] = np.mean([jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test)) for theta in sgldCV_samples[:Nsamples//2]])
   
    #SGHMC
    sghmc_samples = sghmc_sampler(subkey3, Nsamples, opt_params)
    var_error_sghmc[i,j] = jnp.mean(((jnp.var(sghmc_samples[:Nsamples//2],axis=0)-jnp.var(mcmc_samples[:Nsamples//2],axis=0))**2))
    mean_error_sghmc[i,j] = jnp.mean(((jnp.mean(sghmc_samples[:Nsamples//2],axis=0)-jnp.mean(mcmc_samples[:Nsamples//2],axis=0))**2))
    #sghmc_ll[i,j] = log_loss(sghmc_samples[:Nsamples//2],X_test.T,y_test)
    sghmc_logpred[i,j] = np.mean([jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test)) for theta in sghmc_samples[:Nsamples//2]])
   
    #SGHMC-CV
    sghmcCV_samples = sghmcCV_sampler(subkey4, Nsamples, opt_params)
    var_error_sghmcCV[i,j] = jnp.mean(((jnp.var(sghmcCV_samples[:Nsamples//2],axis=0)-jnp.var(mcmc_samples[:Nsamples//2],axis=0))**2))
    mean_error_sghmcCV[i,j] =jnp.mean(((jnp.mean(sghmcCV_samples[:Nsamples//2],axis=0)-jnp.mean(mcmc_samples[:Nsamples//2],axis=0))**2))
    #sghmcCV_ll[i,j] = log_loss(sghmcCV_samples[:Nsamples//2],X_test.T,y_test)
    sghmcCV_logpred[i,j] = np.mean([jnp.sum(vmap(loglikelihood, (None, 0, 0), 0)(theta,X_test,y_test)) for theta in sghmcCV_samples[:Nsamples//2]])
   
  j=j+1
  #plt.plot(mcmc_samples[:,0],label='NUTS')
  #plt.plot(sgld_samples[:,0],label='SGLD')
  #plt.plot(sgldCV_samples[:,0],label='SGLD-CV')
  #plt.plot(sghmc_samples[:,0],label='SGHMC')
  #plt.plot(sghmcCV_samples[:,0],label='SGHMC-CV')
  #plt.axhline(opt_params[0])
  #plt.legend(loc='upper right')
  
  #plt.show()

common.plot_figure(5, 2)
plt.subplot(5, 2, 1)
plt.plot(sgld_samples[:, 0], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_1$")
plt.title("SGLD")

plt.subplot(5, 2, 2)
plt.plot(sgld_samples[:, 1], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_2$")
plt.title("SGLD")

plt.subplot(5, 2, 3)
plt.plot(sgldCV_samples[:, 0], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_1$")
plt.title("SGLD-CV")

plt.subplot(5, 2, 4)
plt.plot(sgldCV_samples[:, 1], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_2$")
plt.title("SGLD-CV")

plt.subplot(5, 2, 5)
plt.plot(sghmc_samples[:, 0], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_1$")
plt.title("SGHMC")

plt.subplot(5, 2, 6)
plt.plot(sghmc_samples[:, 1], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_2$")
plt.title("SGHMC")

plt.subplot(5, 2, 7)
plt.plot(sghmcCV_samples[:, 0], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_1$")
plt.title("SGHMC-CV")

plt.subplot(5, 2, 8)
plt.plot(sghmcCV_samples[:, 1], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_2$")
plt.title("SGHMC-CV")

plt.subplot(5, 2, 9)
plt.plot(mcmc_samples[:, 0], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_1$")
plt.title("NUTS")

plt.subplot(5, 2, 10)
plt.plot(mcmc_samples[:, 1], color = common.line_plot_cols[0], zorder=3)
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_2$")
plt.title("NUTS")

common.save_figure("fig3-5-trace-plots.pdf")

#-----------------------------------
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)
#fig, ax = plt.subplots(1,2, figsize=(10,5))
fig, ax = common.plot_figure(1, 2, width_scale = 1.4, height_scale = 1.4, pad = 6) 
ax[0].plot(D,mean_error_sgld.mean(axis=0), next(linecycler), label='SGLD', c = common.line_plot_cols[0])
ax[0].plot(D,mean_error_sgldCV.mean(axis=0), next(linecycler), label='SGLD-CV', c = common.line_plot_cols[1])
ax[0].plot(D,mean_error_sghmc.mean(axis=0), next(linecycler), label='SGHMC', c = common.line_plot_cols[2])
ax[0].plot(D,mean_error_sghmcCV.mean(axis=0), next(linecycler),label='SGHMC-CV', c = common.line_plot_cols[3])
#ax[0].set(ylabel=r'MSE of $\mathbb{E}_{\pi} [\theta]$')
ax[0].set(ylabel='MSE')
ax[0].set(xlabel='Dimension')
ax[0].legend(loc='upper left')

linecycler = cycle(lines)
ax[1].plot(D,var_error_sgld.mean(axis=0), next(linecycler),label='SGLD', c = common.line_plot_cols[0])
ax[1].plot(D,var_error_sgldCV.mean(axis=0), next(linecycler), label='SGLD-CV', c = common.line_plot_cols[1])
ax[1].plot(D,var_error_sghmc.mean(axis=0), next(linecycler), label='SGHMC', c = common.line_plot_cols[2])
ax[1].plot(D,var_error_sghmcCV.mean(axis=0), next(linecycler), label='SGHMC-CV', c = common.line_plot_cols[3])
#ax[1].set(ylabel=r'MSE of $\textsf{Var}_{\pi} [\theta]$')
ax[1].ticklabel_format(style='sci')
ax[1].set(ylabel='MSE')
ax[1].set(xlabel='Dimension')
ax[1].legend(loc='upper left')

common.save_figure("fig3-4-NUTS-posterior.pdf")

####################################################################
# Figure 3.5                                                       #
####################################################################

common.plot_figure(1, 1, width_scale = 1.6, height_scale = 1.4, book_scale = 0.7, pad = 4.1) 

linecycler = cycle(lines)
#plt.plot(D, mcmc_logpred, next(linecycler), label='NUTS', c = common.line_plot_cols[0])
plt.plot(D, sgld_logpred.mean(axis=0)-mcmc_logpred, next(linecycler),label='SGLD', c = common.line_plot_cols[0])
plt.plot(D, sgldCV_logpred.mean(axis=0)-mcmc_logpred, next(linecycler), label='SGLD-CV', c = common.line_plot_cols[1])
plt.plot(D, sghmc_logpred.mean(axis=0)-mcmc_logpred, next(linecycler), label='SGHMC', c = common.line_plot_cols[2])
plt.plot(D, sghmcCV_logpred.mean(axis=0)-mcmc_logpred, linestyle = (0, (3, 1, 1, 1, 1, 1)), label='SGHMC-CV', c = common.line_plot_cols[3])
plt.ylabel("Difference in log-predictive densities")
plt.xlabel('Dimension')
plt.legend(loc='upper left')

common.save_figure("fig3-6-log-predictive-density.pdf")


common.plot_figure(1, 1, width_scale = 1.6, height_scale = 1.4, book_scale = 0.7, pad = 4.1) 

linecycler = cycle(lines)
#plt.plot(D, mcmc_logpred, next(linecycler), label='NUTS', c = common.line_plot_cols[0])
plt.plot(D, (sgld_logpred.mean(axis=0)-mcmc_logpred)/np.abs(mcmc_logpred)*100, next(linecycler),label='SGLD', c = common.line_plot_cols[0])
plt.plot(D, (sgldCV_logpred.mean(axis=0)-mcmc_logpred)/np.abs(mcmc_logpred)*100, next(linecycler), label='SGLD-CV', c = common.line_plot_cols[1])
plt.plot(D, (sghmc_logpred.mean(axis=0)-mcmc_logpred)/np.abs(mcmc_logpred)*100, next(linecycler), label='SGHMC', c = common.line_plot_cols[2])
plt.plot(D, (sghmcCV_logpred.mean(axis=0)-mcmc_logpred)/np.abs(mcmc_logpred)*100, linestyle = (0, (3, 1, 1, 1, 1, 1)), label='SGHMC-CV', c = common.line_plot_cols[3])
plt.ylabel("Log-predictive percentage improvement")
plt.xlabel('Dimension')
plt.legend(loc='upper left')
common.save_figure("fig3-6-log-predictive-density-percentage-improvement.pdf")
