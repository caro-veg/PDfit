# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:12:59 2019

@author: Carolin
"""

import numpy as np
import emcee as em
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import corner


# read in data
df = pd.read_excel(r'C:\Users\cvegvari\Documents\GSK_info\Microbiology\TK-C-Preliminary-Linelist.xlsx', sheet_name='All Isolates',
                   usecols='B,C,D,E,G,H,I,J', skiprows=[0])

df.columns = ['Strain', 'Rep', 'Agent', 'Conc', 'hr0', 'hr2', 'hr4', 'hr8']

#df['Strain'].unique()
#df['Agent'].unique()

# extract observation
df_gep_42BU = df[(df['Strain']=='100042BU') & (df['Agent']=='Gepotidacin') & (df['Conc']==0.03)]
logobs = df_gep_42BU[['hr0', 'hr2', 'hr4']]
obs = 10 ** logobs
obs_array = np.array(obs)


# time over which growth occurs
times = np.array([0, 2, 4])
# repetitions of each experiment (Agent/Strain combination)
num_reps = 5

#plt.plot(times, logobs.T)

# define priors
r = np.random.uniform(low=-1., high=1.)
log10_K = np.random.uniform(low=0., high=6.)
K = 10**log10_K
params = np.array([r, K])


# define model
def exp_growth(params, times):
    #P0 = obs_array[:,0]
    K = params[1]
    r = params[0]
    return K * np.exp(r * times)
    

# define log likelihood
def log_like(params, times, d):
    M = exp_growth(params, times)
    M = np.log10(M)
    ll = sum(sum(np.log(stats.norm.pdf(d, M))))
    if not np.isfinite(ll):
        return -np.inf    
    return ll

# determine starting values from ML fitting
nll = lambda *args: - log_like(*args)
initial = np.array(params)
soln = minimize(nll, initial, args=(times, logobs))

# define prior probability
def log_prior(params):
    r, K = params
    if -2. < r < 1. and 0. < K < 1e6:
        return 0
    return -np.inf

# define log probability using prior probability and log likelihood
def log_prob(params, times, d):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params, times, d)


# draw initial values using ML results
nwalkers = 8
ndim = len(params)
params0 = soln.x + 1e-5 * np.random.randn(nwalkers, ndim)

                   
# define sampler
sampler = em.EnsembleSampler(nwalkers, ndim, log_prob, args=(times, logobs))

# run MCMC
state = sampler.run_mcmc(params0, 10000, progress=True)

# get MCMC chain for downstream analysis
samples = sampler.get_chain()

# plot chains for all fitted parameters
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["r", "K"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number")
fig.savefig('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\chains_100042BU-gep-0.03.jpeg', dpi=300, bbox_inches='tight')
plt.close(fig)


# check autocorrelation
tau = sampler.get_autocorr_time()
print(tau)

# process chain: discard burn-in, thin by half the autocorrelation time, flatten chain
flat_samples = sampler.get_chain(discard=100, thin=25, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples)
fig.savefig('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\corner_100042BU-gep-0.03.jpeg', dpi=300, bbox_inches='tight')
plt.close(fig)

# get median and 95 percentiles for results
r_median = np.median(flat_samples[:,0])
r_95pc = np.percentile(flat_samples[:,0], [5, 95])

K_median = np.median(flat_samples[:,1])
K_95pc = np.percentile(flat_samples[:,1], [5, 95])

f = open('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\median_95pc.txt', 'w')
print('r median (95 percentile):', r_median, ' (', r_95pc[0], ', ', r_95pc[1], ')', file=f)
print('K median (95 percentile):', K_median, ' (', K_95pc[0], ', ', K_95pc[1], ')', file=f)
f.close()   

