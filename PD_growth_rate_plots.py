# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:10:36 2019

@author: Carolin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'C:\Users\Carolin\Documents\PythonScripts\PDfit\MCMCfits_r_K.xlsx', sheet_name='MCMC')


# Function to plot fitted growth rates (r) by strain and agent
def plot_fitted_r(data, strain, agent):
    df = data[(data['Strain']==strain) & (data['Agent']==agent)]
    dfMIC = df.iloc[3, :]
    fig, ax = plt.subplots()
    ax.errorbar(x=df['Conc'], y=df['r_median'], 
                 yerr=np.stack([df['r_median'] - df['r_5pc'], df['r_95pc'] - df['r_median']]), fmt='o', capsize=5,
                 mfc='blue', mec='blue', ecolor='blue')
    ax.errorbar(x=dfMIC['Conc'], y=dfMIC['r_median'], 
                 yerr=[[dfMIC['r_median'] - dfMIC['r_5pc']], [dfMIC['r_95pc'] - dfMIC['r_median']]], 
                 fmt='o', capsize=5, mfc='red', mec='red', ecolor='red')
    ax.axhline(y=0, linestyle='--', color='darkgrey')
    ax.text(0.8, 0.9, 'MIC', fontsize=18, color='red', transform=plt.gca().transAxes, va = "top", ha="left")
    ax.set_ylim([-2.,1.])
    ax.set_title(strain + ' ' + agent, fontsize=18)
    ax.set_ylabel('Fitted growth rate', fontsize=18)
    ax.set_xlabel('Concentration (CFU/ml)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('C:\\Users\\Carolin\\Documents\\PythonScripts\\PDfit\\Results\\fitted_r_' + strain + '_' + agent + '.jpeg')
    
plot_fitted_r(data, '100042BU', 'Ciprofloxacin')

strains = ['100042BU', '100058BU', '100058TU', '100064BU', '100064TU', '100080BU', '100085BU']
agents = ['Ceftriaxone', 'CG', 'Ciprofloxacin', 'Gepotidacin']

for st in strains:
    for ag in agents:
        plot_fitted_r(data, st, ag)