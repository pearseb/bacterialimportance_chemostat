# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:42:51 2022


Purpose
-------
    A 0D chemostat model of DOM consumption by bacteria
    
    This model predicts the outcome of a bacterial population consuming DOM
    with variable C:N and NOSC.
    
    Increases in yield:
        - occur as a result of increasing N:C ratio of DOM
        - occur as a result of decreasing NOSC (fresher DOM)
        
    Processes:
        - DOC and DON are produced at a ratio of 6.6 (fresh phytoplankton material)
        - DOC is produced with a NOSC of -1.0, indicating more reduced material
        - NH4 can be taken up to supplement bacterial nitrogen demands at a yield of 1 mol NH4 per mol N-biomass
        - bacterial consumption of DOM increases NOSC of remaining DOM because bacteria
          partly oxidizing material release electrons from C-H

@author: Pearse Buchanan
"""


# %% imports

from numba import jit
from cmocean.tools import lighten
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')

# numerical packages

# print versions of packages
print("python version =", sys.version[:5])
print("numpy version =", np.__version__)
print("pandas version =", pd.__version__)
print("seaborn version =", sb.__version__)
print("matplotlib version =", sys.modules[plt.__package__].__version__)
print("cmocean version =", sys.modules[cmo.__package__].__version__)

os.chdir("/Users/buc146/OneDrive - CSIRO/manuscripts/modelling_DOM_breakdown/scripts/microbial_chemostat_model/")
os.getcwd()

from line_plot import line_plot_DOM, line_plot_DOMlDOMs
from model import DOMbreakdown, DOMlDOMs


# %% model parameters

dil = 0.05  # dilution rate (1/day)
days = 1e3  # number of days to run chemostat
dt = 0.0001  # timesteps per day (days)
timesteps = days/dt     # number of timesteps
out_at_day = 10.0       # output results this often (days)
nn_output = days/out_at_day     # number of entries for output


# %% set pulses (if used)

pulse_O2 = 0.0
pulse_ldoc = 0.0
pulse_sdoc = 0.0
pulse_NH4 = 0.0
pulse_blHet = 0.0
pulse_bsHet = 0.0
pulse_int = 0.0


# %% set nutrient affinities and uptake rates

lHet_mumax = 1.2
rHet_mumax = 0.8
sHet_mumax = 0.4

lHet_p_O2 = 450.0
lHet_Vmax_doc = lHet_mumax * 5.0
lHet_K_ldoc = 100.0

rHet_p_O2 = 450.0
rHet_Vmax_doc = rHet_mumax * 5.0
rHet_K_rdoc = 60.0

sHet_p_O2 = 450.0
sHet_Vmax_doc = sHet_mumax * 5.0
sHet_K_sdoc = 10.0

print("long-chain sharing heterotrophs affinity for lDOC", lHet_Vmax_doc / lHet_K_ldoc)
print("long-chain selfish heterotrophs affinity for rDOC", rHet_Vmax_doc / rHet_K_rdoc)
print("short-chain scavenging heterotrophs affinity for sDOC", sHet_Vmax_doc / sHet_K_sdoc)


# %% set basic parameters


# Chemostat inputs
in_ldoc = 1.0
in_ldon = 1.0 * 16.0/122.0
in_rdoc = 1.0
in_rdon = 1.0 * 1.0/17.0
in_sdoc = 1.0
in_sdon = 1.0 * 1.0/17.0
in_O2 = 100.0
in_NH4 = 0.0
in_CO2 = 0.0
in_blHet = 0.1
in_brHet = 0.1
in_bsHet = 0.1

# bacterial biomass stoichiometry
Cb = 1.0
Hb = 1.4
Ob = 0.4
Nb = 0.2

# long-chain fresh (polymers) DOM stoichiometry
Cl = 1.0
Hl = 1.65
Ol = 0.3
Nl = in_ldon / in_ldoc

# long-chain but reworked (slightly oxidized polymers) DOM stoichiometry (CRAM)
Cr = 1.0
Hr = 1.0
Or = 0.6
Nr = in_rdon / in_rdoc

# short-chain (oligomer/monomer) DOM stoichiometry
Cs = 1.0
Hs = 2.0
Os = 0.5
Ns = in_sdon / in_sdoc

# fraction of electrons routed to biosynthesis
lHet_fele = 0.05
rHet_fele = 0.05
sHet_fele = 0.05

# fraction of substrate partially oxidized to sDOM & rDOM
lHet_alpha = np.arange(0.0,0.96,0.05)
# fraction of partially oxidized substrate to sDOM
lHet_beta = np.arange(0.0,0.96,0.05)

lHet_alpha, lHet_beta = np.meshgrid(lHet_alpha, lHet_beta)

'''

    R* = L * K / (µmax - L)

for bacteria, this typically means:
    
    R* = L * K / (Vmax * yield - L)

from the standard half-reactions assuming complete oxidation of a substrate, the yield is equal to:
    
    yield = f * (eS / eB)
    
    where,  f   = the fraction of electrons routed to biomass synthesis
            eS  = the number of electrons released per mol C substrate
            eB  = the number of electrons contained per mol C biomass


now, from our half-reactions assuming partial oxidation of substrate controlled by alpha, we know that:

        
    yield = f * (eS - eP * alpha) / eB
    
    where,  eP      = the number of electrons contained per mol C product (partially oxidized product)
            alpha   = the fraction of organic matter substrate that is partially oxidized
    
and we can see that as more substrate is partially oxidized, the lower the resulting yield of the organism, 
which means the higher the R* and the less competitive this organism is under steady-state conditions.

Coincidentally, we also see that the O2 consumed per CO2 produced increases as partial oxidation increases,
which aligns with observations low O2 consumption per organic carbon molecule oxidized in marine systems.

'''


def rstar(K, L, Vmax, Y):
    return (K*L) / (Vmax * Y - L)


def ele_potential(c, h, o, n):
    return 4*c + 1*h - 2*o - 3*n


def bac_yield(f, eL, eB, eS, eR, alpha, beta):
    return np.fmin(1.0 - alpha, f * (eL - eS*alpha*beta - eR*alpha*(1-beta)) / eB)


def respQ(Y, f, eL, eS, eR, alpha, beta):
    y_doc = 1.0 / Y
    # Cap electron release to be non-negative
    e_res = np.fmax(0.0, eL - alpha*beta*eS - alpha*(1-beta)*eR)
    y_oxy = e_res * (1.0 - f) / 4.0 / Y
    y_co2 = (1.0 - alpha - Y) / Y
    return y_co2 / y_oxy


eL = ele_potential(Cl, Hl, Ol, Nl)
eB = ele_potential(Cb, Hb, Ob, Nb)
eR = ele_potential(Cr, Hr, Or, Nr)
eS = ele_potential(Cs, Hs, Os, Ns)
eL, eS, eB, eR

y_lHet = bac_yield(lHet_fele, eL, eB, eS, eR, lHet_alpha, lHet_beta)
y_rHet = bac_yield(rHet_fele, eR, eB, 0.0, 0.0, 0.0, 0.0)
y_sHet = bac_yield(sHet_fele, eS, eB, 0.0, 0.0, 0.0, 0.0)

rstar_lHet = rstar(lHet_K_ldoc, dil, lHet_Vmax_doc, y_lHet)
rstar_rHet = rstar(rHet_K_rdoc, dil, rHet_Vmax_doc, y_rHet)
rstar_sHet = rstar(sHet_K_sdoc, dil, sHet_Vmax_doc, y_sHet)

rq_lHet = respQ(y_lHet, lHet_fele, eL, eS, eR, lHet_alpha, lHet_beta)
rq_rHet = respQ(y_rHet, rHet_fele, eR, 0.0, 0.0, 0.0, 0.0)
rq_sHet = respQ(y_sHet, sHet_fele, eS, 0.0, 0.0, 0.0, 0.0)

print("R* of DOC (sharer) =", rstar_lHet)
print("R* of DOC (scavenger) =", rstar_sHet)
print("R* of DOC (selfish) =", rstar_rHet)
print("R* combined =", rstar_lHet + rstar_sHet + rstar_rHet)
print(" ")
print("RQ (CO2/O2) (sharer) =", rq_lHet)
print("RQ (CO2/O2) (scavenger) =", rq_sHet)
print("RQ (CO2/O2) (selfish) =", rq_rHet)


#%% plot the relationship between RQ, partial oxidation and yield

fig = plt.figure(figsize=(10,16))
gs = GridSpec(4, 1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])
ax4 = plt.subplot(gs[3,0])

rstart_tot = rstar_lHet + rstar_sHet + rstar_rHet

p1 = ax1.pcolormesh(lHet_alpha, lHet_beta, y_lHet, cmap=cmo.thermal)
p2 = ax2.pcolormesh(lHet_alpha, lHet_beta, rq_lHet, cmap=cmo.thermal)
p3 = ax3.pcolormesh(lHet_alpha, lHet_beta, rstar_lHet, cmap=cmo.thermal, vmin=0, vmax=200)
p4 = ax4.pcolormesh(lHet_alpha, lHet_beta, rstart_tot, cmap=cmo.thermal, vmin=0, vmax=200)

c1 = ax1.contour(lHet_alpha, lHet_beta, y_lHet, levels=[0.01, 0.02, 0.03, 0.04, 0.05], colors='k')
c2 = ax2.contour(lHet_alpha, lHet_beta, rq_lHet, levels=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], colors='k')
c3 = ax3.contour(lHet_alpha, lHet_beta, rstar_lHet, levels=np.arange(5,100,10), colors='k')
c4 = ax4.contour(lHet_alpha, lHet_beta, rstart_tot, levels=[50], colors='k')

ax1.set_xlabel("partial oxidation (alpha)")
ax2.set_xlabel("partial oxidation (alpha)")
ax3.set_xlabel("partial oxidation (alpha)")
ax4.set_xlabel("partial oxidation (alpha)")

ax1.set_ylabel("fraction of product to sDOM")
ax2.set_ylabel("fraction of product to sDOM")
ax3.set_ylabel("fraction of product to sDOM")
ax4.set_ylabel("fraction of product to sDOM")

cb1 = plt.colorbar(p1, ax=ax1, orientation='vertical', fraction=0.05, aspect=10)
cb2 = plt.colorbar(p2, ax=ax2, orientation='vertical', fraction=0.05, aspect=10)
cb3 = plt.colorbar(p3, ax=ax3, orientation='vertical', fraction=0.05, aspect=10)
cb4 = plt.colorbar(p4, ax=ax4, orientation='vertical', fraction=0.05, aspect=10)

cb1.ax.set_ylabel("yield (mol C-biomass per mol C-substrate) (Sharer)")
cb2.ax.set_ylabel("Respiratory quotient (Sharer)")
cb3.ax.set_ylabel("R* (Sharer)")
cb4.ax.set_ylabel("R* (Combined)")


#%%

'''

However, we also know that scavengers (SAR11) have:
    1) higher affinities for dilute substrate (Clifton et al., 2024)
    2) possibly less affected by grazing due to their small size and cell wall properties that prevent capture on filters

This means that...

'''


# %%

dil = 0.05  # dilution rate (1/day)
days = 2e4  # number of days to run chemostat
dt = 0.0001  # timesteps per day (days)
timesteps = days/dt     # number of timesteps
out_at_day = 10.0       # output results this often (days)
nn_output = days/out_at_day     # number of entries for output

aoa_mumax = 0.5
aoa_y_NH4 = 1.0 / 11.0  # Bayer 2022
aoa_y_O2 = 1.0 / 15.5  # Bayer 2022
aoa_Vmax_NH4 = aoa_mumax / aoa_y_NH4
aoa_p_O2 = 275.0
aoa_K_NH4 = 0.1

lHet_mumax = 1.2
rHet_mumax = 1.2
sHet_mumax = 0.4
lHet_p_O2 = 450.0
rHet_p_O2 = 450.0
sHet_p_O2 = 450.0
lHet_Vmax_doc = lHet_mumax * 5.0
rHet_Vmax_doc = rHet_mumax * 5.0
sHet_Vmax_doc = sHet_mumax * 5.0
lHet_K_doc = 100.0
rHet_K_doc = 100.0
sHet_K_doc = 10.0
lHet_Vmax_NH4 = lHet_mumax * 0.05
rHet_Vmax_NH4 = rHet_mumax * 0.05
sHet_Vmax_NH4 = sHet_mumax * 0.05
lHet_K_NH4 = 0.5
rHet_K_NH4 = 0.5
sHet_K_NH4 = 0.1

print("Affinity of Sharer for lDOC =", lHet_Vmax_doc/lHet_K_doc)
print("Affinity of Selfish for rDOC =", rHet_Vmax_doc/rHet_K_doc)
print("Affinity of Scavenger for sDOC =", sHet_Vmax_doc/sHet_K_doc)

# Chemostat inputs
in_ldoc = 0.2 / dil # 0.2 µM C per day (typical at surface)
in_ldon = in_ldoc * 16./122.
in_rdoc = 0.0
in_rdon = in_rdoc * 1.0/17.0
in_sdoc = 0.0
in_sdon = in_sdoc * 0.0/17.0
in_O2 = 100.0
in_NH4 = 0.0
in_CO2 = 0.0
in_baoa = 0.1
in_blHet = 0.1
in_brHet = 0.1
in_bsHet = 0.1

# bacterial biomass stoichiometry
Cb = 1.0
Hb = 1.4
Ob = 0.4
Nb = 0.2

# lDOM substrate stoichiometry
Cl = 1.0
Hl = 1.65
Ol = 0.3

# rDOM substrate stoichiometry
Cr = 1.0
Hr = 1.0  # Cai et al., 2026 PNAS
Or = 0.6  # Cai et al., 2026 PNAS

# sDOM substrate stoichiometry
Cs = 1.0
Hs = 2.0  # Cai et al., 2026 PNAS
Os = 0.6  # Cai et al., 2026 PNAS

# fraction of electrons routed to biosynthesis
lHet_fele = 0.1 # Zakem et al., 2020 ISME
rHet_fele = 0.05 # Zakem et al., 2020 ISME
sHet_fele = 0.1 # Zakem et al., 2020 ISME

# fraction of substrate partially oxidized
lHet_alpha = 0.50
# fraction of partially oxidized substrate to sDOM
lHet_beta = 0.25
# fraction of lDON that persists as organic
donpersist = 0.40

pulse_ldoc = 0.0
pulse_ldon = 0.0
pulse_rdoc = 0.0
pulse_rdon = 0.0
pulse_sdoc = 0.0
pulse_sdon = 0.0
pulse_blHet = 0.0
pulse_brHet = 0.0
pulse_bsHet = 0.0
pulse_baoa = 0.0
pulse_int = 50.0


# %% run model (test)


ts = {}

# Call main model
res = DOMlDOMs(timesteps, nn_output, dt, dil, out_at_day,
               pulse_O2, pulse_ldoc, pulse_ldon, pulse_rdoc, pulse_rdon, pulse_sdoc, pulse_sdon, pulse_NH4, 
               pulse_baoa, pulse_blHet, pulse_brHet, pulse_bsHet, pulse_int,
               aoa_p_O2, lHet_p_O2, rHet_p_O2, sHet_p_O2,
               lHet_Vmax_doc, rHet_Vmax_doc, sHet_Vmax_doc,
               lHet_K_doc, rHet_K_doc, sHet_K_doc,
               aoa_Vmax_NH4, lHet_Vmax_NH4, rHet_Vmax_NH4, sHet_Vmax_NH4,
               aoa_K_NH4, lHet_K_NH4, rHet_K_NH4, sHet_K_NH4,
               aoa_y_NH4, aoa_y_O2, 
               lHet_alpha, lHet_beta,
               lHet_fele, rHet_fele, sHet_fele,
               Hl, Ol, Hr, Or, Hs, Os, Hb, Ob, Nb, donpersist,
               in_O2, in_ldoc, in_ldon, in_rdoc, in_rdon, in_sdoc, in_sdon, in_NH4, in_CO2,
               in_baoa, in_blHet, in_brHet, in_bsHet,
               flow_through=False)

i = 0
out_ldoc = res[i]; i += 1
out_ldon = res[i]; i += 1
out_rdoc = res[i]; i += 1
out_rdon = res[i]; i += 1
out_sdoc = res[i]; i += 1
out_sdon = res[i]; i += 1
out_O2 = res[i]; i += 1
out_CO2 = res[i]; i += 1
out_NH4 = res[i]; i += 1
out_baoa = res[i]; i += 1
out_blHet = res[i]; i += 1
out_brHet = res[i]; i += 1
out_bsHet = res[i]; i += 1
out_uaoa = res[i]; i += 1
out_ulHet = res[i]; i += 1
out_ulHet_C = res[i]; i += 1
out_ulHet_N = res[i]; i += 1
out_urHet = res[i]; i += 1
out_urHet_C = res[i]; i += 1
out_urHet_N = res[i]; i += 1
out_usHet = res[i]; i += 1
out_usHet_C = res[i]; i += 1
out_usHet_N = res[i]; i += 1
out_ylHet = res[i]; i += 1
out_yrHet = res[i]; i += 1
out_ysHet = res[i]; i += 1
out_c_lHet_ldoc = res[i]; i += 1
out_c_lHet_ldon = res[i]; i += 1
out_c_lHet_O2 = res[i]; i += 1
out_p_lHet_CO2 = res[i]; i += 1
out_p_lHet_sdoc = res[i]; i += 1
out_p_lHet_sdon = res[i]; i += 1
out_p_lHet_rdoc = res[i]; i += 1
out_p_lHet_rdon = res[i]; i += 1
out_p_lHet_NH4 = res[i]; i += 1
out_lHet_RQ = res[i]; i += 1
out_c_rHet_rdoc = res[i]; i += 1
out_c_rHet_rdon = res[i]; i += 1
out_c_rHet_O2 = res[i]; i += 1
out_p_rHet_CO2 = res[i]; i += 1
out_p_rHet_NH4 = res[i]; i += 1
out_rHet_RQ = res[i]; i += 1
out_c_sHet_sdoc = res[i]; i += 1
out_c_sHet_sdon = res[i]; i += 1
out_c_sHet_O2 = res[i]; i += 1
out_p_sHet_CO2 = res[i]; i += 1
out_p_sHet_NH4 = res[i]; i += 1
out_sHet_RQ = res[i]; i += 1

ts['dm_O2'] = res[i]; i += 1
ts['dm_CO2'] = res[i]; i += 1
ts['dm_NH4'] = res[i]; i += 1
ts['dm_ldoc'] = res[i]; i += 1
ts['dm_ldon'] = res[i]; i += 1
ts['dm_rdoc'] = res[i]; i += 1
ts['dm_rdon'] = res[i]; i += 1
ts['dm_sdoc'] = res[i]; i += 1
ts['dm_sdon'] = res[i]; i += 1
ts['du_aoa'] = res[i]; i += 1
ts['du_lHet'] = res[i]; i += 1
ts['du_lHet_C'] = res[i]; i += 1
ts['du_lHet_N'] = res[i]; i += 1
ts['du_rHet'] = res[i]; i += 1
ts['du_rHet_C'] = res[i]; i += 1
ts['du_rHet_N'] = res[i]; i += 1
ts['du_sHet'] = res[i]; i += 1
ts['du_sHet_C'] = res[i]; i += 1
ts['du_sHet_N'] = res[i]; i += 1
ts['db_aoa'] = res[i]; i += 1
ts['db_lHet'] = res[i]; i += 1
ts['db_rHet'] = res[i]; i += 1
ts['db_sHet'] = res[i]; i += 1
ts['dy_lHet'] = res[i]; i += 1
ts['dy_rHet'] = res[i]; i += 1
ts['dy_sHet'] = res[i]; i += 1
ts['dc_lHet_ldoc'] = res[i]; i += 1
ts['dc_lHet_ldon'] = res[i]; i += 1
ts['dc_lHet_O2'] = res[i]; i += 1
ts['dp_lHet_CO2'] = res[i]; i += 1
ts['dp_lHet_sdoc'] = res[i]; i += 1
ts['dp_lHet_sdon'] = res[i]; i += 1
ts['dp_lHet_rdoc'] = res[i]; i += 1
ts['dp_lHet_rdon'] = res[i]; i += 1
ts['dp_lHet_NH4'] = res[i]; i += 1
ts['d_lHet_RQ'] = res[i]; i += 1
ts['dc_rHet_sdoc'] = res[i]; i += 1
ts['dc_rHet_sdon'] = res[i]; i += 1
ts['dc_rHet_O2'] = res[i]; i += 1
ts['dp_rHet_CO2'] = res[i]; i += 1
ts['dp_rHet_NH4'] = res[i]; i += 1
ts['d_rHet_RQ'] = res[i]; i += 1
ts['dc_sHet_sdoc'] = res[i]; i += 1
ts['dc_sHet_sdon'] = res[i]; i += 1
ts['dc_sHet_O2'] = res[i]; i += 1
ts['dp_sHet_CO2'] = res[i]; i += 1
ts['dp_sHet_NH4'] = res[i]; i += 1
ts['d_sHet_RQ'] = res[i]

print("Sharer Vmax * y / L =", out_ylHet[-1] * lHet_Vmax_doc / dil)
print("Selfish Vmax * y / L =", out_yrHet[-1] * rHet_Vmax_doc / dil)
print("Scavenger Vmax * y / L =", out_ysHet[-1] * sHet_Vmax_doc / dil)


# %% figures

line_plot_DOMlDOMs(nn_output, out_ldoc, out_ldon, out_rdoc, out_rdon, out_sdoc, out_sdon, out_O2, out_CO2, out_NH4,
                   out_baoa, out_blHet, out_brHet,out_bsHet,
                   out_uaoa, 
                   out_ulHet, out_ulHet_C, out_ulHet_N,
                   out_urHet, out_urHet_C, out_urHet_N,
                   out_usHet, out_usHet_C, out_usHet_N,
                   out_ylHet, out_yrHet, out_ysHet,
                   out_c_lHet_ldoc, out_c_lHet_ldon, out_c_lHet_O2, out_p_lHet_NH4, out_p_lHet_CO2, out_lHet_RQ,
                   out_c_rHet_rdoc, out_c_rHet_rdon, out_c_rHet_O2, out_p_rHet_NH4, out_p_rHet_CO2, out_rHet_RQ,
                   out_c_sHet_sdoc, out_c_sHet_sdon, out_c_sHet_O2, out_p_sHet_NH4, out_p_sHet_CO2, out_sHet_RQ)

print("Final values")
print("lDOC = ", out_ldoc[-1])
print("lDON = ", out_ldon[-1])
print("l-C:N = ", out_ldoc[-1]/out_ldon[-1])
print("rDOC = ", out_rdoc[-1])
print("rDON = ", out_rdon[-1])
print("r-C:N = ", out_rdoc[-1]/out_rdon[-1])
print("sDOC = ", out_sdoc[-1])
print("sDON = ", out_sdon[-1])
print("s-C:N = ", out_sdoc[-1]/out_sdon[-1])
print("AOA = ", out_baoa[-1])
print("lHet = ", out_blHet[-1])
print("rHet = ", out_brHet[-1])
print("sHet = ", out_bsHet[-1])
print("NH4 = ", out_NH4[-1])
print("CO2 = ", out_CO2[-1])
print("RQ of lHet, rHet and sHet =", out_lHet_RQ[-1], out_rHet_RQ[-1], out_sHet_RQ[-1])
print("Total DOC =", out_ldoc[-1] + out_rdoc[-1] + out_sdoc[-1])



#%%

lHet_alpha = np.arange(0,0.96,0.05)
lHet_beta = np.arange(0,0.96,0.05)

output = {}

for ii,iv in enumerate(lHet_alpha):
    for jj,jv in enumerate(lHet_beta):

        print(" ")
        print("alpha =", iv)
        print("beta =", jv)

        res = DOMlDOMs(timesteps, nn_output, dt, dil, out_at_day,
                       pulse_O2, pulse_ldoc, pulse_ldon, pulse_rdoc, pulse_rdon, 
                       pulse_sdoc, pulse_sdon, pulse_NH4, 
                       pulse_baoa, pulse_blHet, pulse_brHet, pulse_bsHet, pulse_int,
                       aoa_p_O2, lHet_p_O2, rHet_p_O2, sHet_p_O2,
                       lHet_Vmax_doc, rHet_Vmax_doc, sHet_Vmax_doc,
                       lHet_K_doc, rHet_K_doc, sHet_K_doc,
                       aoa_Vmax_NH4, lHet_Vmax_NH4, rHet_Vmax_NH4, sHet_Vmax_NH4,
                       aoa_K_NH4, lHet_K_NH4, rHet_K_NH4, sHet_K_NH4,
                       aoa_y_NH4, aoa_y_O2, 
                       lHet_alpha[ii], lHet_beta[jj],
                       lHet_fele, rHet_fele, sHet_fele,
                       Hl, Ol, Hr, Or, Hs, Os, Hb, Ob, Nb, donpersist,
                       in_O2, in_ldoc, in_ldon, in_rdoc, in_rdon, in_sdoc, in_sdon, in_NH4, in_CO2,
                       in_baoa, in_blHet, in_brHet, in_bsHet,
                       flow_through=False)

        i = 0
        output['alpha_%.2f_beta_%.2f_ldoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ldon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_rdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_rdon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_sdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_sdon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_O2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_CO2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_NH4'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_baoa'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_blHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_brHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_bsHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_uaoa'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ulHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ulHet_C'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ulHet_N'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_urHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_urHet_C'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_urHet_N'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_usHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_usHet_C'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_usHet_N'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ylHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_yrHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_ysHet'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_lHet_ldoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_lHet_ldon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_lHet_O2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_CO2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_sdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_sdon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_rdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_rdon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_lHet_NH4'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_lHet_RQ'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_rHet_rdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_rHet_rdon'%(iv,jv)]= res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_rHet_O2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_rHet_CO2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_rHet_NH4'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_rHet_RQ'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_sHet_sdoc'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_sHet_sdon'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_c_sHet_O2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_sHet_CO2'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_p_sHet_NH4'%(iv,jv)] = res[i]; i += 1
        output['alpha_%.2f_beta_%.2f_sHet_RQ'%(iv,jv)] = res[i]




# %% save output

import pickle 

os.getcwd()
output_file = "../../data/output_DOMlDOMs.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(output, f)


# %% get output

import pickle

os.getcwd()
output_file = "../../data/output_DOMlDOMs.pkl"
with open(output_file, "rb") as f:
    output = pickle.load(f)


#%% process output to get final equilibrium values


lHet_alpha = np.arange(0,0.96,0.05)
lHet_beta = np.arange(0,0.96,0.05)

y_lHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
y_rHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
y_sHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
b_lHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
b_sHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
b_rHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
rq_lHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
rq_sHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
rq_rHet_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
ldoc_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
ldon_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
rdoc_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
rdon_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
sdoc_final = np.zeros((len(lHet_alpha), len(lHet_beta)))
sdon_final = np.zeros((len(lHet_alpha), len(lHet_beta)))


for ii,iv in enumerate(lHet_alpha):
    for jj,jv in enumerate(lHet_beta):
        y_lHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_ylHet'%(iv,jv)][-1]
        y_rHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_yrHet'%(iv,jv)][-1]
        y_sHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_ysHet'%(iv,jv)][-1]
        b_lHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_blHet'%(iv,jv)][-1]
        b_rHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_brHet'%(iv,jv)][-1]
        b_sHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_bsHet'%(iv,jv)][-1]
        rq_lHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_lHet_RQ'%(iv,jv)][-1]
        rq_rHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_rHet_RQ'%(iv,jv)][-1]
        rq_sHet_final[ii,jj] = output['alpha_%.2f_beta_%.2f_sHet_RQ'%(iv,jv)][-1]
        ldoc_final[ii,jj] = output['alpha_%.2f_beta_%.2f_ldoc'%(iv,jv)][-1]
        ldon_final[ii,jj] = output['alpha_%.2f_beta_%.2f_ldon'%(iv,jv)][-1]
        rdoc_final[ii,jj] = output['alpha_%.2f_beta_%.2f_rdoc'%(iv,jv)][-1]
        rdon_final[ii,jj] = output['alpha_%.2f_beta_%.2f_rdon'%(iv,jv)][-1]
        sdoc_final[ii,jj] = output['alpha_%.2f_beta_%.2f_sdoc'%(iv,jv)][-1]
        sdon_final[ii,jj] = output['alpha_%.2f_beta_%.2f_sdon'%(iv,jv)][-1]


# %%


fig = plt.figure(figsize=(8,30))
gs = GridSpec(9, 1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])
ax4 = plt.subplot(gs[3,0])
ax5 = plt.subplot(gs[4,0])
ax6 = plt.subplot(gs[5,0])
ax7 = plt.subplot(gs[6,0])
ax8 = plt.subplot(gs[7,0])
ax9 = plt.subplot(gs[8,0])

doc_tot = ldoc_final + sdoc_final + rdoc_final

p1 = ax1.pcolormesh(lHet_alpha, lHet_beta, y_lHet_final.T, cmap=cmo.thermal)
p2 = ax2.pcolormesh(lHet_alpha, lHet_beta, rq_lHet_final.T, cmap=cmo.thermal)
p3 = ax3.pcolormesh(lHet_alpha, lHet_beta, b_lHet_final.T, cmap=cmo.thermal, vmin=0, vmax=1.0)
p4 = ax4.pcolormesh(lHet_alpha, lHet_beta, (b_sHet_final/b_lHet_final*100).T, cmap=cmo.thermal, vmin=0, vmax=100)
p5 = ax5.pcolormesh(lHet_alpha, lHet_beta, (b_rHet_final/b_lHet_final*100).T, cmap=cmo.thermal, vmin=0, vmax=100)
p6 = ax6.pcolormesh(lHet_alpha, lHet_beta, doc_tot.T, cmap=cmo.thermal, vmin=0, vmax=100)
p7 = ax7.pcolormesh(lHet_alpha, lHet_beta, (ldoc_final/doc_tot*100).T, cmap=cmo.thermal, vmin=0, vmax=100)
p8 = ax8.pcolormesh(lHet_alpha, lHet_beta, (rdoc_final/doc_tot*100).T, cmap=cmo.thermal, vmin=0, vmax=100)
p9 = ax9.pcolormesh(lHet_alpha, lHet_beta, (sdoc_final/doc_tot*100).T, cmap=cmo.thermal, vmin=0, vmax=100)

c1 = ax1.contour(lHet_alpha, lHet_beta, y_lHet_final.T, levels=[0.01, 0.02, 0.03, 0.04, 0.05], colors='k')
c2 = ax2.contour(lHet_alpha, lHet_beta, rq_lHet_final.T, levels=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], colors='k')
c3 = ax3.contour(lHet_alpha, lHet_beta, b_lHet_final.T, levels=np.arange(0.0, 1.01, 0.1), colors='k')
c4 = ax4.contour(lHet_alpha, lHet_beta, (b_sHet_final/b_lHet_final*100).T, levels=np.arange(0.1, 1.0, 0.1)*100, colors='k')
c4 = ax4.contour(lHet_alpha, lHet_beta, (b_sHet_final/b_lHet_final*100).T, levels=[50], colors='firebrick')
c5 = ax5.contour(lHet_alpha, lHet_beta, (b_rHet_final/b_lHet_final*100).T, levels=np.arange(0.1, 1.0, 0.1)*100, colors='k')
c6 = ax6.contour(lHet_alpha, lHet_beta, doc_tot.T, levels=np.arange(10,101,20), colors='k')
c6 = ax6.contour(lHet_alpha, lHet_beta, doc_tot.T, levels=[30,50], colors='firebrick')
c6 = ax6.contour(lHet_alpha, lHet_beta, doc_tot.T, levels=[35,45], colors='goldenrod')
c7 = ax7.contour(lHet_alpha, lHet_beta, (ldoc_final/doc_tot*100).T, levels=np.arange(0.1, 1.0, 0.1)*100, colors='k')
c8 = ax8.contour(lHet_alpha, lHet_beta, (rdoc_final/doc_tot*100).T, levels=np.arange(0.1, 1.0, 0.1)*100, colors='k')
c8 = ax8.contour(lHet_alpha, lHet_beta, (rdoc_final/doc_tot*100).T, levels=[50], colors='firebrick')
c9 = ax9.contour(lHet_alpha, lHet_beta, (sdoc_final/doc_tot*100).T, levels=np.arange(0.1, 1.0, 0.1)*100, colors='k')

ax1.set_xlabel("partial oxidation (alpha)")
ax2.set_xlabel("partial oxidation (alpha)")
ax3.set_xlabel("partial oxidation (alpha)")
ax4.set_xlabel("partial oxidation (alpha)")
ax5.set_xlabel("partial oxidation (alpha)")
ax6.set_xlabel("partial oxidation (alpha)")
ax7.set_xlabel("partial oxidation (alpha)")
ax8.set_xlabel("partial oxidation (alpha)")
ax8.set_xlabel("partial oxidation (alpha)")
ax9.set_xlabel("partial oxidation (alpha)") 

ax1.set_ylabel("fraction of product to sDOM (beta)")
ax2.set_ylabel("fraction of product to sDOM (beta)")
ax3.set_ylabel("fraction of product to sDOM (beta)")
ax4.set_ylabel("fraction of product to sDOM (beta)")
ax5.set_ylabel("fraction of product to sDOM (beta)")
ax6.set_ylabel("fraction of product to sDOM (beta)")
ax7.set_ylabel("fraction of product to sDOM (beta)")
ax8.set_ylabel("fraction of product to sDOM (beta)")
ax9.set_ylabel("fraction of product to sDOM (beta)")

cb1 = plt.colorbar(p1, ax=ax1, orientation='vertical', fraction=0.05, aspect=10)
cb2 = plt.colorbar(p2, ax=ax2, orientation='vertical', fraction=0.05, aspect=10)
cb3 = plt.colorbar(p3, ax=ax3, orientation='vertical', fraction=0.05, aspect=10)
cb4 = plt.colorbar(p4, ax=ax4, orientation='vertical', fraction=0.05, aspect=10)
cb5 = plt.colorbar(p5, ax=ax5, orientation='vertical', fraction=0.05, aspect=10)
cb6 = plt.colorbar(p6, ax=ax6, orientation='vertical', fraction=0.05, aspect=10)
cb7 = plt.colorbar(p7, ax=ax7, orientation='vertical', fraction=0.05, aspect=10)
cb8 = plt.colorbar(p8, ax=ax8, orientation='vertical', fraction=0.05, aspect=10)
cb9 = plt.colorbar(p9, ax=ax9, orientation='vertical', fraction=0.05, aspect=10)

cb1.ax.set_ylabel("Growth yield (Sharer)\n(mol C-biomass per mol C-substrate)")
cb2.ax.set_ylabel("Respiratory quotient (Sharer)")
cb3.ax.set_ylabel("Biomass (Sharer)\n(µM C)")
cb4.ax.set_ylabel("Scavenger % of community")
cb5.ax.set_ylabel("Selfish reworker % of community")
cb6.ax.set_ylabel("DOC (Total)\n(µM C)")
cb7.ax.set_ylabel("Fraction l-DOC")
cb8.ax.set_ylabel("Fraction r-DOC")
cb9.ax.set_ylabel("Fraction s-DOC")


# %% check that the experiments actually came to full equilibrium

ii = 10
jj = 10

iv = lHet_alpha[ii]
jv = lHet_beta[jj]

prefix = f'alpha_{iv:.2f}_beta_{jv:.2f}'


# ------------------------------------------------------------
# Extract selected experiment
# ------------------------------------------------------------

out_ldoc = output[f'{prefix}_ldoc']
out_ldon = output[f'{prefix}_ldon']
out_rdoc = output[f'{prefix}_rdoc']
out_rdon = output[f'{prefix}_rdon']
out_sdoc = output[f'{prefix}_sdoc']
out_sdon = output[f'{prefix}_sdon']

out_O2  = output[f'{prefix}_O2']
out_CO2 = output[f'{prefix}_CO2']
out_NH4 = output[f'{prefix}_NH4']   

out_baoa  = output[f'{prefix}_baoa']
out_blHet = output[f'{prefix}_blHet']
out_brHet = output[f'{prefix}_brHet']
out_bsHet = output[f'{prefix}_bsHet']

out_uaoa = output[f'{prefix}_uaoa']

out_ulHet   = output[f'{prefix}_ulHet']
out_ulHet_C = output[f'{prefix}_ulHet_C']
out_ulHet_N = output[f'{prefix}_ulHet_N']

out_urHet   = output[f'{prefix}_urHet']
out_urHet_C = output[f'{prefix}_urHet_C']
out_urHet_N = output[f'{prefix}_urHet_N']

out_usHet   = output[f'{prefix}_usHet']
out_usHet_C = output[f'{prefix}_usHet_C']
out_usHet_N = output[f'{prefix}_usHet_N']

out_ylHet = output[f'{prefix}_ylHet']
out_yrHet = output[f'{prefix}_yrHet']
out_ysHet = output[f'{prefix}_ysHet']

out_c_lHet_ldoc = output[f'{prefix}_c_lHet_ldoc']
out_c_lHet_ldon = output[f'{prefix}_c_lHet_ldon']
out_c_lHet_O2   = output[f'{prefix}_c_lHet_O2']
out_p_lHet_NH4  = output[f'{prefix}_p_lHet_NH4']
out_p_lHet_CO2  = output[f'{prefix}_p_lHet_CO2']
out_lHet_RQ     = output[f'{prefix}_lHet_RQ']

out_c_rHet_rdoc = output[f'{prefix}_c_rHet_rdoc']
out_c_rHet_rdon = output[f'{prefix}_c_rHet_rdon']
out_c_rHet_O2   = output[f'{prefix}_c_rHet_O2']
out_p_rHet_NH4  = output[f'{prefix}_p_rHet_NH4']
out_p_rHet_CO2  = output[f'{prefix}_p_rHet_CO2']
out_rHet_RQ     = output[f'{prefix}_rHet_RQ']

out_c_sHet_sdoc = output[f'{prefix}_c_sHet_sdoc']
out_c_sHet_sdon = output[f'{prefix}_c_sHet_sdon']
out_c_sHet_O2   = output[f'{prefix}_c_sHet_O2']
out_p_sHet_NH4  = output[f'{prefix}_p_sHet_NH4']
out_p_sHet_CO2  = output[f'{prefix}_p_sHet_CO2']
out_sHet_RQ     = output[f'{prefix}_sHet_RQ']


# ------------------------------------------------------------
# Plot selected experiment
# ------------------------------------------------------------

line_plot_DOMlDOMs(
    nn_output,
    out_ldoc,
    out_ldon,
    out_rdoc,
    out_rdon,
    out_sdoc,
    out_sdon,
    out_O2,
    out_CO2,
    out_NH4,
    out_baoa,
    out_blHet,
    out_brHet,
    out_bsHet,
    out_uaoa,
    out_ulHet,
    out_ulHet_C,
    out_ulHet_N,
    out_urHet,
    out_urHet_C,
    out_urHet_N,
    out_usHet,
    out_usHet_C,
    out_usHet_N,
    out_ylHet,
    out_yrHet,
    out_ysHet,
    out_c_lHet_ldoc,
    out_c_lHet_ldon,
    out_c_lHet_O2,
    out_p_lHet_NH4,
    out_p_lHet_CO2,
    out_lHet_RQ,
    out_c_rHet_rdoc,
    out_c_rHet_rdon,
    out_c_rHet_O2,
    out_p_rHet_NH4,
    out_p_rHet_CO2,
    out_rHet_RQ,
    out_c_sHet_sdoc,
    out_c_sHet_sdon,
    out_c_sHet_O2,
    out_p_sHet_NH4,
    out_p_sHet_CO2,
    out_sHet_RQ
)


# ------------------------------------------------------------
# Print final state
# ------------------------------------------------------------

print("alpha =", iv)
print("beta  =", jv)

print("\nFinal values")

print("lDOC = ", out_ldoc[-1])
print("lDON = ", out_ldon[-1])
print("l-C:N = ", out_ldoc[-1] / out_ldon[-1])

print("rDOC = ", out_rdoc[-1])
print("rDON = ", out_rdon[-1])
print("r-C:N = ", out_rdoc[-1] / out_rdon[-1])

print("sDOC = ", out_sdoc[-1])
print("sDON = ", out_sdon[-1])
print("s-C:N = ", out_sdoc[-1] / out_sdon[-1])

print("AOA  = ", out_baoa[-1])
print("lHet = ", out_blHet[-1])
print("rHet = ", out_brHet[-1])
print("sHet = ", out_bsHet[-1])

print("NH4 = ", out_NH4[-1])
print("CO2 = ", out_CO2[-1])

print(
    "RQ of lHet, rHet and sHet =",
    out_lHet_RQ[-1],
    out_rHet_RQ[-1],
    out_sHet_RQ[-1]
)

print(
    "Total DOC =",
    out_ldoc[-1] + out_rdoc[-1] + out_sdoc[-1]
)

# %%
