# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:51:33 2022

@author: pearseb
"""

import numpy as np

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten


def line_plot_OMZ(nn_output, out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, \
                  out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, \
                  out_rHet, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX):
    
    fig = plt.figure(figsize=(16,6))
    gs = GridSpec(1,4)

    ax1 = plt.subplot(gs[0,0])
    plt.title('External substrates')
    plt.plot(np.arange(nn_output+1), out_Sd, color='k', linestyle='-', label='Sd ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_Sp, color='grey', linestyle='-', label='Sp ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_O2*1e3, color='firebrick', linestyle='-', label='O$_2$ (nM)')
    plt.legend()

    ax2 = plt.subplot(gs[0,1])
    plt.title('Nitrogen species')
    plt.plot(np.arange(nn_output+1), out_NO3, color='k', linestyle='-', label='NO$_3$')
    plt.plot(np.arange(nn_output+1), out_NO2, color='firebrick', linestyle='-', label='NO$_2$')
    plt.plot(np.arange(nn_output+1), out_NH4, color='goldenrod', linestyle='-', label='NH$_4$')
    plt.plot(np.arange(nn_output+1), out_N2, color='royalblue', linestyle='-', label='N$_2$')
    totalN = out_NO3 + out_NO2 + out_NH4 + out_N2 + out_bHet + out_b1Den + out_b2Den + out_b3Den + out_bFac + out_bAOO + out_bNOO + out_bAOX + out_Sd + out_Sp
    plt.plot(np.arange(nn_output+1), totalN, color='k', linestyle=':', linewidth=2.0, label="Total N")
    plt.legend()

    ax3 = plt.subplot(gs[0,2])
    plt.title('Biomasses (uM N)')
    plt.plot(np.arange(nn_output+1), out_bHet, color='k', linestyle='-', label='Het')
    plt.plot(np.arange(nn_output+1), out_bFac, color='k', linestyle='--', label='Fac')
    plt.plot(np.arange(nn_output+1), out_b1Den, color='firebrick', linestyle='-', label='1Den')
    plt.plot(np.arange(nn_output+1), out_b2Den, color='firebrick', linestyle='--', label='2Den')
    plt.plot(np.arange(nn_output+1), out_b3Den, color='firebrick', linestyle=':', label='3Den')
    plt.plot(np.arange(nn_output+1), out_bAOO, color='goldenrod', linestyle='-', label='AOO')
    plt.plot(np.arange(nn_output+1), out_bNOO, color='royalblue', linestyle='-', label='NOO')
    plt.plot(np.arange(nn_output+1), out_bAOX, color='forestgreen', linestyle='-', label='AOX')
    plt.legend()
    
    ax4 = plt.subplot(gs[0,3])
    plt.title('Rates (uM / day)')
    plt.plot(np.arange(nn_output+1), out_rHet, color='k', linestyle='-', label='Heterotrophy')
    plt.plot(np.arange(nn_output+1), out_rO2C, color='k', linestyle='--', label='O$_2$ loss')
    plt.plot(np.arange(nn_output+1), out_r1Den, color='firebrick', linestyle='-', label='NO$_3$ --> NO$_2$')
    plt.plot(np.arange(nn_output+1), out_r2Den, color='firebrick', linestyle='--', label='NO$_2$ --> N$_2$')
    plt.plot(np.arange(nn_output+1), out_r3Den, color='firebrick', linestyle=':', label='NO$_3$ --> N$_2$')
    plt.plot(np.arange(nn_output+1), out_rAOO, color='goldenrod', linestyle='-', label='NH$_4$ --> NO$_2$')
    plt.plot(np.arange(nn_output+1), out_rNOO, color='royalblue', linestyle='-', label='NO$_2$ --> NO$_3$')
    plt.plot(np.arange(nn_output+1), out_rAOX, color='forestgreen', linestyle='-', label='Anammox (NH$_4$ loss)')
    plt.legend(loc='center right', bbox_to_anchor=(1.8,0.5), ncol=1)
    
    plt.subplots_adjust(right=0.825, left=0.05)
    
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    
    return fig




def line_plot_DOM(nn_output, out_doc, out_doh, out_doo, out_don, out_O2, out_CO2, out_NH4, \
                  out_bHet1, out_bHet2, out_bHet3, \
                  out_uHet1, out_uHet2, out_uHet3, \
                  out_yHet1, out_yHet2, out_yHet3, \
                  out_f1_doc, out_r1_doc, out_r1_doh, out_r1_doo, out_r1_don, out_r1_O2, out_p1_NH4, out_p1_CO2, out_RQ1, \
                  out_f2_doc, out_r2_doc, out_r2_doh, out_r2_doo, out_r2_don, out_r2_O2, out_p2_NH4, out_p2_CO2, out_RQ2, \
                  out_f3_doc, out_r3_doc, out_r3_doh, out_r3_doo, out_r3_don, out_r3_O2, out_p3_NH4, out_p3_CO2, out_RQ3):
    
    fig = plt.figure(figsize=(22,16))
    gs = GridSpec(2,4)

    ax1 = plt.subplot(gs[0,0])
    plt.title('Substrates')
    plt.plot(np.arange(nn_output+1), out_doc, color='k', linestyle='-', label='DOC ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_CO2, color='firebrick', linestyle='-', label='CO$_2$ ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_NH4, color='forestgreen', linestyle='-', label='NH$_4$ ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_O2, color='royalblue', linestyle='--', label='O$_2$ ($\mu$M)')
    plt.legend()

    ax2 = plt.subplot(gs[0,1])
    plt.title('Stoichiometry of DOC')
    plt.plot(np.arange(nn_output+1), out_don/out_doc, color='k', linestyle='-', label="N:C")
    plt.plot(np.arange(nn_output+1), out_doh/out_doc, color='k', linestyle='--', label="H:C")
    plt.plot(np.arange(nn_output+1), out_doo/out_doc, color='k', linestyle=':', label="O:C")
    plt.legend()
    
    ax3 = plt.subplot(gs[0,2])
    plt.title('Nominal oxidation state of DOC')
    nosc = 4 - (4*out_doc + out_doh - 2*out_doo - 3*out_don)/out_doc
    plt.plot(np.arange(nn_output+1), nosc, color='k', linestyle='-')
    
    ax4 = plt.subplot(gs[0,3])
    plt.title('Biomasses ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_bHet1, color='k', linestyle='-', label='Primary degrader ')
    plt.plot(np.arange(nn_output+1), out_bHet2, color='k', linestyle='--', label='Secondary degrader')
    plt.plot(np.arange(nn_output+1), out_bHet3, color='k', linestyle=':', label='Scavenger')
    plt.legend()
    
    ax5 = plt.subplot(gs[1,0])
    plt.title('Growth rates (day$^{-1}$)')
    plt.plot(np.arange(nn_output+1), out_uHet1, color='k', linestyle='-', label='Primary degrader')
    plt.plot(np.arange(nn_output+1), out_uHet2, color='k', linestyle='--', label='Secondary degrader')
    plt.plot(np.arange(nn_output+1), out_uHet3, color='k', linestyle=':', label='Scavenger')
    plt.legend()
    
    ax6 = plt.subplot(gs[1,1])
    plt.title('Growth yields (mol C biomass (mol DOC)$^{-1}$)')
    plt.plot(np.arange(nn_output+1), out_yHet1, color='k', linestyle='-', label='Primary degrader')
    plt.plot(np.arange(nn_output+1), out_yHet2, color='k', linestyle='--', label='Secondary degrader')
    plt.plot(np.arange(nn_output+1), out_yHet3, color='k', linestyle=':', label='Scavenger')
    plt.legend()
    
    ax7 = plt.subplot(gs[1,2])
    plt.title('DOC consumption rates ($\mu$M day$^{-1}$)')
    plt.plot(np.arange(nn_output+1), out_r1_doc, color='k', linestyle='-', label='Primary degrader')
    plt.plot(np.arange(nn_output+1), out_r2_doc, color='k', linestyle='--', label='Secondary degrader')
    plt.plot(np.arange(nn_output+1), out_r3_doc, color='k', linestyle=':', label='Scavenger')
    plt.legend()
    
    ax8 = plt.subplot(gs[1,3])
    #plt.title('Respiratory Quotient\n(O$_2$ consumed (CO$_2$ produced)$^{-1}$)')
    #plt.plot(np.arange(nn_output+1), out_RQ1, color='k', linestyle='-', label='Primary degrader')
    #plt.plot(np.arange(nn_output+1), out_RQ2, color='k', linestyle='--', label='Secondary degrader')
    #plt.plot(np.arange(nn_output+1), out_RQ3, color='k', linestyle=':', label='Scavenger')
    plt.title('Fractional availability of DOC')
    plt.plot(np.arange(nn_output+1), out_f1_doc, color='k', linestyle='-', label='Primary degrader')
    plt.plot(np.arange(nn_output+1), out_f2_doc, color='k', linestyle='--', label='Secondary degrader')
    plt.plot(np.arange(nn_output+1), out_f3_doc, color='k', linestyle=':', label='Scavenger')
    plt.legend()
    
    plt.subplots_adjust(right=0.825, left=0.05)
    
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax5.set_xscale('log')
    ax6.set_xscale('log')
    ax7.set_xscale('log')
    ax8.set_xscale('log')
    
    return fig



def line_plot_DOMlDOMs(nn_output, out_ldoc, out_ldon, out_rdoc, out_rdon, out_sdoc, out_sdon,out_O2, out_CO2, out_NH4, \
                       out_baoa, out_blHet, out_brHet, out_bsHet, \
                       out_uaoa, out_ulHet, out_ulHet_C, out_ulHet_N, \
                       out_urHet, out_urHet_C, out_urHet_N, \
                       out_usHet, out_usHet_C, out_usHet_N, \
                       out_ylHet, out_yrHet, out_ysHet, \
                       out_c_lHet_ldoc, out_c_lHet_ldon, out_c_lHet_O2, out_p_lHet_NH4, out_p_lHet_CO2, out_lHet_RQ, \
                       out_c_rHet_rdoc, out_c_rHet_rdon, out_c_rHet_O2, out_p_rHet_NH4, out_p_rHet_CO2, out_rHet_RQ, \
                       out_c_sHet_sdoc, out_c_sHet_sdon, out_c_sHet_O2, out_p_sHet_NH4, out_p_sHet_CO2, out_sHet_RQ):
        
    lHet_Nlim = out_ulHet_N < out_ulHet_C
    rHet_Nlim = out_urHet_N < out_urHet_C
    sHet_Nlim = out_usHet_N < out_usHet_C
    
    fig = plt.figure(figsize=(22,16))
    gs = GridSpec(2,2)

    ax1 = plt.subplot(gs[0,0])
    plt.title('Substrates')
    plt.plot(np.arange(nn_output+1)*10, out_ldoc, color='k', linestyle='-', label='lDOC ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_rdoc, color='k', linestyle=':', label='rDOC ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_sdoc, color='k', linestyle='--', label='sDOC ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_CO2, color='goldenrod', linestyle='-', label='CO$_2$ ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_NH4, color='forestgreen', linestyle='-', label='NH$_4$ ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_O2, color='royalblue', linestyle='--', label='O$_2$ ($\mu$M)')
    plt.legend()

    ax2 = plt.subplot(gs[0,1])
    plt.title('Biomasses ($\mu$M)')
    plt.plot(np.arange(nn_output+1)*10, out_blHet, color='k', linestyle='-', label='Sharer ')
    plt.plot(np.arange(nn_output+1)*10, out_brHet, color='k', linestyle=':', label='Shelfish (reworker)')
    plt.plot(np.arange(nn_output+1)*10, out_bsHet, color='k', linestyle='--', label='Scavenger')
    plt.plot(np.arange(nn_output+1)*10, out_baoa, color='pink', linestyle='-', label='AOA ')
    plt.legend()
    
    ax3 = plt.subplot(gs[1,0])
    plt.title('Growth rates (day$^{-1}$)')
    plt.plot(np.arange(nn_output+1)*10, out_ulHet, color='k', linestyle='-', label='Sharer')
    plt.plot(np.arange(nn_output+1)*10, out_urHet, color='k', linestyle=':', label='Selfish (reworker)')
    plt.plot(np.arange(nn_output+1)*10, out_usHet, color='k', linestyle='--', label='Scavenger')
    plt.plot(np.arange(nn_output+1)*10, out_uaoa, color='pink', linestyle='-', label='AOA')
    plt.legend()
    ax3.plot(np.arange(nn_output+1)*10, out_ulHet_C, color='goldenrod', linestyle='-', label='Sharer (C)') 
    ax3.plot(np.arange(nn_output+1)*10, out_urHet_C, color='goldenrod', linestyle=':', label='Selfish (C)')
    ax3.plot(np.arange(nn_output+1)*10, out_usHet_C, color='goldenrod', linestyle='--', label='Scavenger (C)')
    ax3.plot(np.arange(nn_output+1)*10, out_ulHet_N, color='green', linestyle='-', label='Sharer (N)')
    ax3.plot(np.arange(nn_output+1)*10, out_urHet_N, color='green', linestyle=':', label='Selfish (N)')
    ax3.plot(np.arange(nn_output+1)*10, out_usHet_N, color='green', linestyle='--', label='Scavenger (N)')
    
    ax4 = plt.subplot(gs[1,1])
    plt.title('Growth yields (mol C biomass (mol DOC)$^{-1}$)')
    plt.plot(np.arange(nn_output+1)*10, out_ylHet, color='k', linestyle='-', label='Sharer')
    plt.plot(np.arange(nn_output+1)*10, out_yrHet, color='k', linestyle=':', label='Selfish (reworker)')
    plt.plot(np.arange(nn_output+1)*10, out_ysHet, color='k', linestyle='--', label='Scavenger')
    plt.legend()
    
    #ax7 = plt.subplot(gs[1,2])
    #plt.title('DOC consumption rates ($\mu$M day$^{-1}$)')
    #plt.plot(np.arange(nn_output+1), out_r1_doc, color='k', linestyle='-', label='Primary degrader')
    #plt.plot(np.arange(nn_output+1), out_r2_doc, color='k', linestyle='--', label='Secondary degrader')
    #plt.plot(np.arange(nn_output+1), out_r3_doc, color='k', linestyle=':', label='Scavenger')
    #plt.legend()

    
    plt.subplots_adjust(right=0.825, left=0.05)
    
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    
    return fig




