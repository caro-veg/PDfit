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

strain = "100042BU"
agent = "Ciprofloxacin"

# read in data
df = pd.read_excel(r'C:\Users\cvegvari\Documents\GSK_info\Microbiology\TK-C-Preliminary-Linelist.xlsx', sheet_name='All Isolates',
                   usecols='B,C,D,E,G,H,I,J', skiprows=[0])

df.columns = ['Strain', 'Rep', 'Agent', 'Conc', 'hr0', 'hr2', 'hr4', 'hr8']

df_control = pd.read_excel(r'C:\Users\cvegvari\Documents\GSK_info\Microbiology\TK-C-Preliminary-Linelist.xlsx', sheet_name='All Isolates',
                 usecols='B,C,D,E,L,M,N,O', skiprows=[0])

df_control.columns = ['Strain', 'Rep', 'Agent', 'Conc', 'hr0', 'hr2', 'hr4', 'hr8']

fitted_r_data = pd.read_excel(r'D:\GCPKPD\MCMCfits_r_K.xlsx', sheet_name='MCMC')
xagent = agent
if agent == 'Ceftriaxone/Gepotidacin':
    xagent = 'CG'
fitr = fitted_r_data[(fitted_r_data['Strain']==strain) & (fitted_r_data['Agent']==xagent)]

#df['Strain'].unique()
#df['Agent'].unique()

# extract observation
df_ag_strain = df[(df['Strain']==strain) & (df['Agent']==agent)]
df_ag_strain_control = df_control[(df['Strain']==strain) & (df['Agent']==agent)] 
dfc = df_ag_strain_control[0:5]
dfc['Conc'] = [0.] * 5
df_ag_strain = dfc.append(pd.DataFrame(data=df_ag_strain), ignore_index=True)
logobs = np.asarray(df_ag_strain[['hr0', 'hr2', 'hr4', 'hr8']])
obs = 10 ** logobs
lnobs = np.log(obs)

# time over which growth occurs
times = np.array([0., 2., 4., 8.])

plt.plot(times, logobs.T)

########################## ML FIT #############################################

# define priors
#a = np.random.uniform(low=-10., high=10.)
A = fitr.loc[fitr['Conx']=='0x', ['r_median']]
A = A.iloc[0]['r_median']
B = fitr.loc[fitr['Conx']=='10x', ['r_median']]
B = B.iloc[0]['r_median']
#A = 0.21
#B = -1.19
D = A - B
gamma = 0.0068
params = np.array([D, gamma, B])


C = df_ag_strain['Conc'].to_numpy()
B0 = 10 ** df_ag_strain['hr0'].to_numpy()


def r_conc(params, C):
    D, gamma, B = params
    return  D * np.exp(- gamma * C) + B

def exp_growth(params, C, B0, times):
    r = r_conc(params, C)
    return np.log(np.transpose(np.array([B0,]))) + np.transpose(np.array([r,])) * times
    

# define log likelihood
def log_like(params, times, C, B0, d):
    B = exp_growth(params, C, B0, times)
    #ll = np.sum(np.log(stats.norm.pdf(x=d, loc=B)))
    ll = - (d - B) * (d - B) / (2 * 1 * 1)
    ll = np.sum(ll)
    if not np.isfinite(ll):
        return -np.inf    
    return ll

# determine starting values from ML fitting
nll = lambda *args: - log_like(*args)
initial = np.array(params)
soln = minimize(nll, initial, args=(times, C, B0, lnobs))



########################## MCMC ##############################################

# define prior probability
def log_prior(params):
    D, gamma, B = params
    if (0. < D < 100.) and (0. < gamma < 1000.) and (-100. < B < 100.):
        return 0
    return -np.inf

# define log probability using prior probability and log likelihood
def log_prob(params, times, C, B0, d):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params, times, C, B0, d)


# draw initial values using ML results
nwalkers = 8
ndim = len(params)
params0 = soln.x + 1e-2 * np.random.randn(nwalkers, ndim)
   
# define sampler
sampler = em.EnsembleSampler(nwalkers, ndim, log_prob, args=(times, C, B0, lnobs))

# run MCMC
state = sampler.run_mcmc(params0, 100000, progress=True)

# get MCMC chain for downstream analysis
samples = sampler.get_chain()

# plot chains for all fitted parameters
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ['a', 's', 'q']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step number')
fig.savefig('D:\\GCPKPD\\PD_r_conc_fits\\chains_' + strain + '_' + xagent + '.jpeg', dpi=300, bbox_inches='tight')
plt.close(fig)

#plt.plot(samples[:, :, 0], "k", alpha=0.3)
#plt.xlim(0, len(samples))
#plt.xlabel("step number")


# check autocorrelation
tau = sampler.get_autocorr_time()
print(tau)

# process chain: discard burn-in, thin by half the autocorrelation time, flatten chain
flat_samples = sampler.get_chain(discard=100, thin=1000, flat=True)
print(flat_samples.shape)


fig = corner.corner(flat_samples)
fig.savefig('D:\\GCPKPD\\PD_r_conc_fits\\corner_' + strain + '_' + xagent + '.jpeg', dpi=300, bbox_inches='tight')
plt.close(fig)

# get median and 95 percentiles for results
D_median = np.median(flat_samples[:,0])
D_95pc = np.percentile(flat_samples[:,0], [5, 95])

gamma_median= np.median(flat_samples[:,1])
gamma_95pc = np.percentile(flat_samples[:,1], [5, 95])

B_median = np.median(flat_samples[:,2])
B_95pc = np.percentile(flat_samples[:,2], [5, 95])

f = open('D:\\GCPKPD\\PD_r_conc_fits\\median_95pc_' + strain + '_' + xagent + '.txt', 'w')
print('D median (95 percentile):', D_median, ' (', D_95pc[0], ', ', D_95pc[1], ')', file=f)
print('gamma median (95 percentile):', gamma_median, ' (', gamma_95pc[0], ', ', gamma_95pc[1], ')', file=f)
print('B median (95 percentile):', B_median, ' (', B_95pc[0], ', ', B_95pc[1], ')', file=f)
f.close()   

########################## PLOT FIT ###########################################

# Function to plot fitted growth rates (r) by strain and agent
def plot_fitted_r(data, strain, agent, mlfit, C):
    df = data[(data['Strain']==strain) & (data['Agent']==agent)]
    dfMIC = df.iloc[3, :]
    fig, ax = plt.subplots()
    ax.errorbar(x=df['Conc'], y=df['r_median'], 
                 yerr=np.stack([df['r_median'] - df['r_5pc'], df['r_95pc'] - df['r_median']]), fmt='o', capsize=5,
                 mfc='blue', mec='blue', ecolor='blue')
    ax.errorbar(x=dfMIC['Conc'], y=dfMIC['r_median'], 
                 yerr=[[dfMIC['r_median'] - dfMIC['r_5pc']], [dfMIC['r_95pc'] - dfMIC['r_median']]], 
                 fmt='o', capsize=5, mfc='red', mec='red', ecolor='red')
    D, gamma, B = mlfit
    ax.plot(C, D * np.exp(- gamma * C) + B, color="cyan", linestyle=":")
    ax.axhline(y=0, linestyle='--', color='darkgrey')
    ax.text(0.8, 0.9, 'MIC', fontsize=18, color='red', transform=plt.gca().transAxes, va = "top", ha="left")
    #ax.set_ylim([-2.,1.])
    ax.set_title(strain + ' ' + agent, fontsize=18)
    ax.set_ylabel('Fitted growth rate', fontsize=18)
    ax.set_xlabel('Concentration (CFU/ml)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig
    
    
fig = plot_fitted_r(fitted_r_data, strain, xagent, soln.x, C)
fig.savefig('D:\\GCPKPD\\PD_r_conc_fits\\fitted_r_ML_' + strain + '_' + xagent + '.jpeg')
plt.close(fig)

fig = plot_fitted_r(fitted_r_data, strain, xagent, np.array([D_median, gamma_median, B_median]), C)
fig.savefig('D:\\GCPKPD\\PD_r_conc_fits\\fitted_r_MCMC_' + strain + '_' + xagent + '.jpeg')
plt.close(fig)

#fig = plot_fitted_r(fitted_r_data, strain, agent, params, C)
#fig.savefig('D:\\GCPKPD\\PD_r_conc_fits\\params_r_' + strain + '_' + agent + '.jpeg')
#plt.close(fig)
