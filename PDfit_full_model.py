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

np.random.seed(45)

# read in data
df = pd.read_excel(r'C:\Users\cvegvari\Documents\GSK_info\Microbiology\TK-C-Preliminary-Linelist.xlsx', sheet_name='All Isolates',
                   usecols='B,C,D,E,G,H,I,J', skiprows=[0])

df.columns = ['Strain', 'Rep', 'Agent', 'Conc', 'hr0', 'hr2', 'hr4', 'hr8']

df_control = pd.read_excel(r'C:\Users\cvegvari\Documents\GSK_info\Microbiology\TK-C-Preliminary-Linelist.xlsx', sheet_name='All Isolates',
                 usecols='B,C,D,E,L,M,N,O', skiprows=[0])

df_control.columns = ['Strain', 'Rep', 'Agent', 'Conc', 'hr0', 'hr2', 'hr4', 'hr8']

#df['Strain'].unique()
#df['Agent'].unique()

# extract observation
df_ag_strain = df[(df['Strain']=='100042BU') & (df['Agent']=='Gepotidacin')]
df_ag_strain_control = df_control[(df['Strain']=='100042BU') & (df['Agent']=='Gepotidacin')]
dfc = df_ag_strain_control[0:5]
dfc['Conc'] = [0] * 5
df_ag_strain = dfc.append(pd.DataFrame(data=df_ag_strain), ignore_index=True)
logobs = np.asarray(df_ag_strain[['hr0', 'hr2', 'hr4', 'hr8']])
obs = 10 ** logobs
#obs_array = np.array(obs)


# time over which growth occurs
times = np.array([0., 2., 4., 8.])
# repetitions of each experiment (Agent/Strain combination)
num_reps = 5

plt.plot(times, logobs.T)

# define priors
#r = np.random.uniform(low=-1., high=1.)
log10_K = np.random.uniform(low=0., high=6.)
K = 10**log10_K
rmax = np.random.uniform(low=0., high=1.)
s = np.random.uniform(low=-2., high=0.)
#params = np.array([r, K, rmax, s])
params = np.array([rmax, s, K])

C = df_ag_strain['Conc'].to_numpy()

# define model
def growth_conc(params, C):
    rmax = params[0]
    s = params[1]
    #print(s)
    return rmax * np.exp(s * C)

def exp_growth(params, C, times):
    K = params[2]
    r = growth_conc(params, C)
    #print(r)
    return K * np.exp(np.transpose(np.array([r,])) * times)

    

# define log likelihood
def log_like(params, times, C, d):
    B = exp_growth(params, C, times)
    B = np.log10(B)
    #print(B)
    ll = np.sum(np.log(stats.norm.pdf(d, B)))
    #print(ll)
    if not np.isfinite(ll):
        return -np.inf    
    return ll

# determine starting values from ML fitting
nll = lambda *args: - log_like(*args)
initial = np.array(params)
soln = minimize(nll, initial, args=(times, df_ag_strain['Conc'], logobs))


# define prior probability
def log_prior(params):
    rmax, s, K = params
    if (0. < rmax < 1.) and (-2. < s < 0.) and (0. < K < 1e6):
        return 0
    #else:
        #print("inf")
    return -np.inf

# define log probability using prior probability and log likelihood
def log_prob(params, times, C, d):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params, times, C, d)


# draw initial values using ML results
nwalkers = 6
ndim = len(params)
params0 = soln.x + 1e-2 * np.random.randn(nwalkers, ndim)

   
# define sampler
sampler = em.EnsembleSampler(nwalkers, ndim, log_prob, args=(times, C, logobs))

# run MCMC
state = sampler.run_mcmc(params0, 10000, progress=True)

# get MCMC chain for downstream analysis
samples = sampler.get_chain()

# plot chains for all fitted parameters
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ['r_max', 's', 'K']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step number')
#fig.savefig('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\chains_-Ciprofloxacin-4.0.pdf', dpi=300, bbox_inches='tight')
#plt.close(fig)


# check autocorrelation
tau = sampler.get_autocorr_time()
print(tau)

# process chain: discard burn-in, thin by half the autocorrelation time, flatten chain
flat_samples = sampler.get_chain(discard=100, thin=25, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples)
#fig.savefig('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\corner_1100058BU-Ciprofloxacin-4.0.pdf', dpi=300, bbox_inches='tight')
#plt.close(fig)

# get median and 95 percentiles for results
r_median = np.median(flat_samples[:,0])
r_95pc = np.percentile(flat_samples[:,0], [5, 95])

s_median= np.median(flat_samples[:,1])
s_95c = np.percentile(flat_samples[:,1], [5, 95])

K_median = np.median(flat_samples[:,2])
K_95pc = np.percentile(flat_samples[:,2], [5, 95])

#f = open('C:\\Users\\cvegvari\\Documents\\Python Scripts\\PDfit\\median_95pc_100058BU-Ciprofloxacin-4.0.txt', 'w')
#print('r median (95 percentile):', r_median, ' (', r_95pc[0], ', ', r_95pc[1], ')', file=f)
#print('K median (95 percentile):', K_median, ' (', K_95pc[0], ', ', K_95pc[1], ')', file=f)
#f.close()   

