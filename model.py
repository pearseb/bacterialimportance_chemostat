# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:28:17 2022

Purpose
-------
    A 0D chemostat model of redox reactions occuring in marine OMZs
    
    This model predicts the outcome of microbial populations competing for resources
    in low oxygen zones using the energetics and stoichiometries of the reactions they
    perform.
    
    Sources of competition are:
        - between denitrifying heterotrophs and aerobic heterotrophs for Organic matter
        - between ammonia oxidising archaea and anaerobic ammonium oxidisers for NH4
        - between nitrite oxidising bacteria, denitrifying heterotrophs and anaerobic ammonium oxidisers for NO2
        - between ammonia oxidising archaea, nitrite oxidising bacteria and aerobic heterotrophs for O2
    
    Possible symbioses are:
        - heterotrophs producing NH4 for ammonia oxidising archaea and anammox bacteria
        - ammonia oxidising archaea producing NO2 for denitrifiers
        - nitrite oxidising archaea producing NO3 for denitrifiers
        - denitrifiers producing NO2 for nitrite oxidising bacteria and anammox bacteria
        - anammox bactria producing NO3 for denitrifiers
        - phytoplankton in the deep chlorophyll max producing O2 for heterotrophs??


@author: Emily Zakem & Pearse Buchanan
"""

### imports
import numpy as np
from numba import jit
from pyparsing import Or

@jit(nopython=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
             pulse_Sd, pulse_bHet, pulse_bFac, pulse_O2, pulse_int, \
             po_aer, po_aoo, po_noo, \
             VmaxS, K_s, \
             VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_1DenFac, K_n_Den, \
             VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
             VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
             y_oHet, y_oO2, y_oHetFac, y_oO2Fac, \
             y_n1DenFac, y_n1NO3Fac, y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, \
             y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
             e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, \
             in_Sd, in_Sp, in_O2, in_NO3, in_NO2, in_NH4, \
             in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX):
    '''
    ______________________________________________________
    0D Chemostat model for aerobic and anaerobic processes
    ------------------------------------------------------
    
    INPUTS
    ------
    
        timesteps   :   number of timesteps in run
        nn_output   :   number of times we save output
        dt          :   number of timesteps per day
        dil         :   dilution rate of chemostat
        out_at_day  :   day interval to record output
        
        pulse_Sd    :   instant injection of OM into the model (mmol)
        pulse_bHet  :   instant injection of bHet into the model (mmol)
        pulse_bFac  :   instant injection of bFac into the model (mmol)
        pulse_O2    :   instant injection of O2 into the model (mmol)
        pulse_int   :   interval of pulsing (days)
                        
        po_aer      :   diffusion-limited O2 uptake by aerobic heterotrophs (m^3 / mmol N / day)
        po_aoo      :   diffusion-limited O2 uptake by aerobic ammonia oxidisers (m^3 / mmol N / day)
        po_noo      :   diffusion-limited O2 uptake by aerobic nitrite oxidisers (m^3 / mmol N / day)
        
        VmaxS       :   maximum uptake rate of organic matter (mol orgN / mol N-cell / day )
        K_s         :   half saturation coefficient for organic matter uptake
        VmaxN_1Den  :   maximum uptake rate of NO3 by nitrate reducing bacteria (mol NO3 / mol N-cell / day )
        VmaxN_2Den  :   maximum uptake rate of NO2 by nitrite reducing bacteria (mol NO3 / mol N-cell / day )
        VmaxN_3Den  :   maximum uptake rate of NO3 by fully denitrifying bacteria (mol NO3 / mol N-cell / day )
        VmaxN_1DenFac : maximum uptake rate of NO3 by nitrate reducing facultative bacteria (mol NO3 / mol N-cell / day )
        K_n_Den     :   half saturation coefficient for DIN uptake by heterotrophic bacteria
        VmaxN_AOO   :   maximum uptake rate of NH4 by ammonia oxidisers (mol NH4 / mol N-cell / day )
        K_n_AOO     :   half saturation coefficient for NH4 uptake by ammonia oxidisers
        VmaxN_NOO   :   maximum uptake rate of NO2 by nitrite oxidisers (mol NO2 / mol N-cell / day )
        K_n_NOO     :   half saturation coefficient for NO2 uptake by nitrite oxidisers
        VmaxNH4_AOX :   maximum uptake rate of NH4 by anammox bacteria (mol NH4 / mol N-cell / day )
        VmaxNO2_AOX :   maximum uptake rate of NO2 by anammox bacteria (mol NO2 / mol N-cell / day )
        K_nh4_AOX   :   half saturation coefficient for NH4 uptake by anaerobic ammonium oxidisers
        K_no2_AOX   :   half saturation coefficient for NO2 uptake by anaerobic ammonium oxidisers
                
        y_oHet      :   mol N-biomass / mol N-organics consumed
        y_oO2       :   mol N-biomass / mol O2 reduced
        y_oHetFac   :   mol N-biomass / mol N-organics consumed
        y_oO2Fac    :   mol N-biomass / mol O2 reduced
        y_n1DenFac  :   mol N-biomass / mol N-organics consumed
        y_n1NO3Fac  :   mol N-biomass / mol NO3 reduced
        y_n1Den     :   mol N-biomass / mol N-organics consumed
        y_n1NO3     :   mol N-biomass / mol NO3 reduced (NO3 --> NO2)
        y_n2Den     :   mol N-biomass / mol N-organics consumed
        y_n2NO2     :   mol N-biomass / mol NO2 reduced (NO2 --> N2)
        y_n3Den     :   mol N-biomass / mol N-organics consumed
        y_n3NO3     :   mol N-biomass / mol NO3 reduced (NO3 --> N2)
        y_nAOO      :   mol N-biomass / mol NH4 oxidised
        y_oAOO      :   mol N-biomass / mol O2 reduced
        y_nNOO      :   mol N-biomass / mol NO2 oxidised
        y_oNOO      :   mol N-biomass / mol O2 reduced
        y_nh4AOX    :   mol N-biomass / mol NH4 oxidised
        y_no2AOX    :   mol N-biomass / mol NO2 reduced
        e_n2Den     :   production of N2 by nitrite reducing denitrifiers (mol N2 / mol N-biomass)
        e_n3Den     :   production of N2 by nitrate reducing denitrifiers (mol N2 / mol N-biomass)
        e_no3AOX    :   production of NO3 by anammox bacteria (mol NO3 / mol N-biomass)
        e_n2AOX     :   production of N2 by anammox bacteria (mol N2 / mol N-biomass)
        
        in_Sd       :   initial Sd concentration
        in_Sp       :   initial Sp concentration
        in_O2       :   initial O2 concentration
        in_NO3      :   initial NO3 concentration
        in_NO2      :   initial NO2 concentration
        in_NH4      :   initial NH4 concentration
        in_bHet     :   initial Het concentration of biomass
        in_bFac     :   initial Fac concentration of biomass
        in_b1Den    :   initial 1Den concentration of biomass
        in_b2Den    :   initial 2Den concentration of biomass
        in_b3Den    :   initial 3Den concentration of biomass
        in_bAOO     :   initial AOO concentration of biomass
        in_bNOO     :   initial NOO concentration of biomass
        in_bAOX     :   initial AOX concentration of biomass
        
        
    OUTPUTS
    -------
    
        out_Sd       :   Sd concentration
        out_Sp       :   Sp concentration
        out_O2       :   O2 concentration
        out_NO3      :   NO3 concentration
        out_NO2      :   NO2 concentration
        out_NH4      :   NH4 concentration
        out_N2       :   N2 concentration
        
        out_bHet     :   Het concentration of biomass
        out_bFac     :   Fac concentration of biomass
        out_b1Den    :   1Den concentration of biomass
        out_b2Den    :   2Den concentration of biomass
        out_b3Den    :   3Den concentration of biomass
        out_bAOO     :   AOO concentration of biomass
        out_bNOO     :   NOO concentration of biomass
        out_bAOX     :   AOX concentration of biomass
        
        out_uHet     :   Het growth rate
        out_uFac     :   Fac growth rate
        out_u1Den    :   1Den growth rate
        out_u2Den    :   2Den growth rate
        out_u3Den    :   3Den growth rate
        out_uAOO     :   AOO growth rate
        out_uNOO     :   NOO growth rate
        out_uAOX     :   AOX growth rate

        out_facaer   :   average fraction of aerobic heterotrophy by facultative population
        out_facnoo_lim : average fraction of NO2-limited growth of NOO
        out_rHet     :   Rate of heterotrophy (Org--> NH4) (mmol Org m-3 day-1)
        out_rHetAer  :   Rate of aerobic heterotrophy (Org--> NH4) (mmol Org m-2 day-1)
        out_rO2C     :   Rate of O2 consumption (mmol O2 m-3 day-1)
        out_r1Den    :   Rate of denitrification (NO3 --> NO2) (mmol NO3 m-3 day-1)
        out_r2Den    :   Rate of denitrification (NO2 --> N2) (mmol NO2 m-3 day-1)
        out_r3Den    :   Rate of denitrification (NO3 --> N2) (mmol NO3 m-3 day-1)
        out_rAOO     :   Rate of ammonia oxidation (NH4 --> NO2) (mmol NH4 m-3 day-1)
        out_rNOO     :   Rate of nitrite oxidation (NO2 --> NO3) (mmol NO2 m-3 day-1)
        out_rAOX     :   Rate of anammox (NH4 + 81/70 * NO2 --> 11/70 * NO3 + 69.5/70 * N2) (mmol NH4 m-3 day-1)
    '''
    
    # transfer initial inputs to model variables
    m_Sd = in_Sd
    m_Sp = in_Sp
    m_O2 = in_O2
    m_NO3 = in_NO3
    m_NO2 = in_NO2
    m_NH4 = in_NH4
    m_N2 = 0.0
    m_bHet = in_bHet
    m_bFac = in_bFac
    m_b1Den = in_b1Den
    m_b2Den = in_b2Den
    m_b3Den = in_b3Den
    m_bAOO = in_bAOO
    m_bNOO = in_bNOO
    m_bAOX = in_bAOX
    
    # set the output arrays 
    out_Sd = np.ones((int(nn_output)+1)) * np.nan
    out_Sp = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NO3 = np.ones((int(nn_output)+1)) * np.nan
    out_NO2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_N2 = np.ones((int(nn_output)+1)) * np.nan
    out_bHet = np.ones((int(nn_output)+1)) * np.nan
    out_bFac = np.ones((int(nn_output)+1)) * np.nan
    out_b1Den = np.ones((int(nn_output)+1)) * np.nan
    out_b2Den = np.ones((int(nn_output)+1)) * np.nan
    out_b3Den = np.ones((int(nn_output)+1)) * np.nan
    out_bAOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOO = np.ones((int(nn_output)+1)) * np.nan
    out_bAOX = np.ones((int(nn_output)+1)) * np.nan
    out_uHet = np.ones((int(nn_output)+1)) * np.nan
    out_uFac = np.ones((int(nn_output)+1)) * np.nan
    out_u1Den = np.ones((int(nn_output)+1)) * np.nan
    out_u2Den = np.ones((int(nn_output)+1)) * np.nan
    out_u3Den = np.ones((int(nn_output)+1)) * np.nan
    out_uAOO = np.ones((int(nn_output)+1)) * np.nan
    out_uNOO = np.ones((int(nn_output)+1)) * np.nan
    out_uAOX = np.ones((int(nn_output)+1)) * np.nan
    out_facaer = np.ones((int(nn_output)+1)) * np.nan
    out_facnoo_lim = np.ones((int(nn_output)+1)) * np.nan
    out_maxnox = np.ones((int(nn_output)+1)) * np.nan
    out_rHet = np.ones((int(nn_output)+1)) * np.nan
    out_rHetAer = np.ones((int(nn_output)+1)) * np.nan
    out_rO2C = np.ones((int(nn_output)+1)) * np.nan
    out_r1Den = np.ones((int(nn_output)+1)) * np.nan
    out_r2Den = np.ones((int(nn_output)+1)) * np.nan
    out_r3Den = np.ones((int(nn_output)+1)) * np.nan
    out_rAOO = np.ones((int(nn_output)+1)) * np.nan
    out_rNOO = np.ones((int(nn_output)+1)) * np.nan
    out_rAOX = np.ones((int(nn_output)+1)) * np.nan
    
    # set the array for recording average aerobic or anaerobic activity of facultative population
    interval = int((1/dt * out_at_day))
    facaer = np.ones((interval)) * np.nan
    facnoo_lim = np.ones((interval)) * np.nan
    maxnox = np.ones((interval)) * np.nan
    
    # set the arrays for activities over the final interval of the experiment
    dm_O2 = np.ones((interval)) * np.nan
    dm_NO3 = np.ones((interval)) * np.nan
    dm_NO2 = np.ones((interval)) * np.nan
    dm_NH4 = np.ones((interval)) * np.nan
    dm_N2 = np.ones((interval)) * np.nan
    dm_Sd = np.ones((interval)) * np.nan
    du_Het = np.ones((interval)) * np.nan
    du_Fac = np.ones((interval)) * np.nan
    du_1Den = np.ones((interval)) * np.nan
    du_2Den = np.ones((interval)) * np.nan
    du_3Den = np.ones((interval)) * np.nan
    du_AOO = np.ones((interval)) * np.nan
    du_NOO = np.ones((interval)) * np.nan
    du_AOX = np.ones((interval)) * np.nan
    db_Het = np.ones((interval)) * np.nan
    db_Fac = np.ones((interval)) * np.nan
    db_1Den = np.ones((interval)) * np.nan
    db_2Den = np.ones((interval)) * np.nan
    db_3Den = np.ones((interval)) * np.nan
    db_AOO = np.ones((interval)) * np.nan
    db_NOO = np.ones((interval)) * np.nan
    db_AOX = np.ones((interval)) * np.nan
    dr_Het = np.ones((interval)) * np.nan
    dr_HetAer = np.ones((interval)) * np.nan
    dr_O2C = np.ones((interval)) * np.nan
    dr_1Den = np.ones((interval)) * np.nan
    dr_2Den = np.ones((interval)) * np.nan
    dr_3Den = np.ones((interval)) * np.nan
    dr_AOO = np.ones((interval)) * np.nan
    dr_NOO = np.ones((interval)) * np.nan
    dr_AOX = np.ones((interval)) * np.nan
    
    # record the initial conditions
    i = 0
    out_Sd[i] = m_Sd 
    out_Sp[i] = m_Sp
    out_O2[i] = m_O2
    out_NO3[i] = m_NO3
    out_NO2[i] = m_NO2 
    out_NH4[i] = m_NH4 
    out_N2[i] = m_N2
    out_bHet[i] = m_bHet
    out_bFac[i] = m_bFac
    out_b1Den[i] = m_b1Den
    out_b2Den[i] = m_b2Den
    out_b3Den[i] = m_b3Den 
    out_bAOO[i] = m_bAOO
    out_bNOO[i] = m_bNOO
    out_bAOX[i] = m_bAOX
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        # uptake rates (p)
        p_O2_aer = po_aer * m_O2                            # mol O2 / day
        p_O2_aoo = po_aoo * m_O2                            # mol O2 / day
        p_O2_noo = po_noo * m_O2                            # mol O2 / day
        p_Sp = VmaxS * m_Sp / (K_s + m_Sp)                  # mol Org / day
        p_Sd = VmaxS * m_Sd / (K_s + m_Sd)                  # mol Org / day
        p_FacNO3 = VmaxN_1DenFac * m_NO3 / (K_n_Den + m_NO3)             # mol NO3 / day
        p_1DenNO3 = VmaxN_1Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_2DenNO2 = VmaxN_2Den * m_NO2 / (K_n_Den + m_NO2)               # mol NO2 / day
        p_3DenNO3 = VmaxN_3Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_NH4_AOO = VmaxN_AOO * m_NH4 / (K_n_AOO + m_NH4)   # mol NH4 / day
        p_NO2_NOO = VmaxN_NOO * m_NO2 / (K_n_NOO + m_NO2)   # mol NO2 / day
        p_NH4_AOX = VmaxNH4_AOX * m_NH4 / (K_nh4_AOX + m_NH4)     # mol NH4 / day
        p_NO2_AOX = VmaxNO2_AOX * m_NO2 / (K_no2_AOX + m_NO2)     # mol NO2 / day
        
        # growth rates (u)
        u_Het = np.fmax(0.0, np.fmin(p_Sd * y_oHet, p_O2_aer * y_oO2))          # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
        u_FacO2 = np.fmax(0.0, np.fmin(p_Sd * y_oHetFac, p_O2_aer * y_oO2Fac))  # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
        u_FacNO3 = np.fmax(0.0, np.fmin(p_Sd * y_n1DenFac, p_FacNO3 * y_n1NO3Fac)) # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_1Den = np.fmax(0.0, np.fmin(p_Sd * y_n1Den, p_1DenNO3 * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_2Den = np.fmax(0.0, np.fmin(p_Sd * y_n2Den, p_2DenNO2 * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
        u_3Den = np.fmax(0.0, np.fmin(p_Sd * y_n3Den, p_3DenNO3 * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_AOO = np.fmax(0.0, np.fmin(p_NH4_AOO * y_nAOO, p_O2_aoo * y_oAOO))    # mol NH4 / day * mol Biomass / mol NH4 || mol O2 / day * mol Biomass / mol O2
        u_NOO = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOO, p_O2_noo * y_oNOO))    # mol NO2 / day * mol Biomass / mol NO2 || mol O2 / day * mol Biomass / mol O2
        u_AOX = np.fmax(0.0, np.fmin(p_NO2_AOX * y_no2AOX, p_NH4_AOX * y_nh4AOX)) # mol NO2 / day * mol Biomass / mol NO2 || mol NH4 / day * mol Biomass / mol NH4
        
        # record the limitation of NOO by O2 or NO2
        if (p_NO2_NOO * y_nNOO) > (p_O2_noo * y_oNOO):
            facnoo_lim[int(t % interval)] = 1.0
        else:
            facnoo_lim[int(t % interval)] = 0.0
        
        ### BIOMASS TYPES & THEIR STOICHIOMETRIES (already encoded within yields, which are inputs)
        # 1. Aerobic heterotrophy ( 7.1-OM + 47-O2 --> B + 42-CO2 + 6.1-NH4)
        # 2. Facultative anaerobes ( 8.9-OM + 60-O2 --> B + 53-CO2 + 7.9-NH4)
        # 3. Denitrification ( 7.9-OM + 105-NO3 --> B + 47-CO2 + 6.9-NH4 + 105-NO2)
        # 4. Denitrification ( 7.9-OM + 70-NO2 --> B + 47-CO2 + 6.9-NH4 + 35-N2)
        # 5. Denitrification ( 7.9-OM + 42-NO3 --> B + 47-CO2 + 6.9-NH4 + 21-N2)
        # 6. Ammonia oxidation ( 112-NH4 + 5-CO2 + 162-O2 --> B + 111-NO2)
        # 7. Nitrite oxidation ( 334-NO2 + 5-CO2 + 162-O2 --> B + 334-NO3)
        # 8. Anammox ( 154-NH4 + 216-NO2 + 5-CO2 --> B + 42-NO3 + 163-N2)
        
        ###
        ### Change in state variables per day (ddt)
        ###
        
        # deal with facultative bacteria first
        if u_FacO2 >= u_FacNO3:
            u_Fac = u_FacO2
            ddt_Sd_Fac = u_Fac * m_bFac / y_oHetFac
            ddt_O2_Fac = u_Fac * m_bFac / y_oO2Fac
            ddt_NO3_Fac = 0.0
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_oHetFac - 1)
            facaer[int(t % interval)] = 1.0
        else:
            u_Fac = u_FacNO3
            ddt_Sd_Fac = u_Fac * m_bFac / y_n1DenFac
            ddt_O2_Fac = 0.0
            ddt_NO3_Fac = u_Fac * m_bFac / y_n1NO3Fac
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_n1DenFac - 1)
            facaer[int(t % interval)] = 0.0
            
        
        ### rates
        aer_heterotrophy = u_Het * m_bHet / y_oHet      \
                           + ddt_Sd_Fac * facaer[int(t % interval)]
        heterotrophy = u_Het * m_bHet / y_oHet      \
                       + ddt_Sd_Fac                 \
                       + u_1Den * m_b1Den / y_n1Den \
                       + u_2Den * m_b2Den / y_n2Den \
                       + u_3Den * m_b3Den / y_n3Den       
        oxy_consumption = u_Het * m_bHet / y_oO2    \
                          + ddt_O2_Fac              \
                          + u_AOO * m_bAOO / y_oAOO \
                          + u_NOO * m_bNOO / y_oNOO
        den_nar = u_1Den * m_b1Den / y_n1NO3        \
                  + ddt_NO3_Fac
        den_nir = u_2Den * m_b2Den / y_n2NO2
        den_full = u_3Den * m_b3Den / y_n3NO3
        ammonia_ox = u_AOO * m_bAOO / y_nAOO
        nitrite_ox = u_NOO * m_bNOO / y_nNOO
        anammox_nh4 = u_AOX * m_bAOX / y_nh4AOX
        anammox_no2 = u_AOX * m_bAOX / y_no2AOX
        anammox_no3 = u_AOX * m_bAOX * e_no3AOX
        # record nitrite oxidation rate
        maxnox[int(t % interval)] = nitrite_ox
        
        # Dissolved organic matter (consumed by 1, 2, 3, 4, 5)
        ddt_Sd = dil * (in_Sd - m_Sd) - heterotrophy
                 
        # Particulate organic matter (consumed by 1, 2, 3, 4, 5)
        ddt_Sp = dil * (in_Sp - m_Sp)   # particle associated POM not currently included in model
        
        # Dissolved oxygen (consumed by 1, 2, 6, 7)
        ddt_O2 = dil * (in_O2 - m_O2) - oxy_consumption
                 
        # Nitrate (consumed by 2, 3, 5) (produced by 7, 8)
        ddt_NO3 = dil * (in_NO3 - m_NO3)        \
                 - den_nar                      \
                 - den_full                     \
                 + nitrite_ox                   \
                 + anammox_no3        
        
        # Nitrite (consumed by 4, 7, 8) (produced by 2, 3, 6)
        ddt_NO2 = dil * (in_NO2 - m_NO2)        \
                 - den_nir                      \
                 - nitrite_ox                   \
                 - anammox_no2                  \
                 + den_nar                      \
                 + u_AOO * m_bAOO * (1./y_nAOO - 1)   # because it uses one mol of NH4 for biomass synthesis         
        
        # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5)
        ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                 - ammonia_ox                   \
                 - anammox_nh4                  \
                 - u_NOO * m_bNOO               \
                 + u_Het * m_bHet * (1./y_oHet - 1)    \
                 + ddt_NH4_Fac                         \
                 + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                 + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                 + u_3Den * m_b3Den * (1./y_n3Den - 1)    
        
        # Dinitrogen gas (produced by 3, 4, 8)
        ddt_N2 = dil * (-m_N2)                  \
                 + u_2Den * m_b2Den * e_n2Den   \
                 + u_3Den * m_b3Den * e_n3Den   \
                 + u_AOX * m_bAOX * e_n2AOX
        
        # Biomass of aerobic heterotrophs
        ddt_bHet = dil * (-m_bHet)              \
                   + u_Het * m_bHet 
        
        # Biomass of facultative heterotrophs
        ddt_bFac = dil * (-m_bFac)              \
                   + u_Fac * m_bFac 
        
        # Biomass of nitrate denitrifiers
        ddt_b1Den = dil * (-m_b1Den)            \
                   + u_1Den * m_b1Den
        
        # Biomass of nitrite denitrifiers
        ddt_b2Den = dil * (-m_b2Den)            \
                   + u_2Den * m_b2Den
        
        # Biomass of full denitrifiers
        ddt_b3Den = dil * (-m_b3Den)            \
                   + u_3Den * m_b3Den 
        
        # Biomass of ammonia oxidising archaea
        ddt_bAOO = dil * (-m_bAOO)              \
                   + u_AOO * m_bAOO 
        
        # Biomass of nitrite oxidising bacteria
        ddt_bNOO = dil * (-m_bNOO)              \
                   + u_NOO * m_bNOO
        
        # Biomass of anammox bacteria
        ddt_bAOX = dil * (-m_bAOX)              \
                   + u_AOX * m_bAOX 
        
        
        # apply changes to state variables normalised by timestep (dt = timestep length (per day))
        m_Sd = m_Sd + ddt_Sd * dt
        m_Sp = m_Sp + ddt_Sp * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NO3 = m_NO3 + ddt_NO3 * dt
        m_NO2 = m_NO2 + ddt_NO2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_N2 = m_N2 + ddt_N2 * dt
        m_bHet = m_bHet + ddt_bHet * dt
        m_bFac = m_bFac + ddt_bFac * dt
        m_b1Den = m_b1Den + ddt_b1Den * dt
        m_b2Den = m_b2Den + ddt_b2Den * dt
        m_b3Den = m_b3Den + ddt_b3Den * dt
        m_bAOO = m_bAOO + ddt_bAOO * dt
        m_bNOO = m_bNOO + ddt_bNOO * dt
        m_bAOX = m_bAOX + ddt_bAOX * dt
        
        # Record tracer concentrations at each timestep
        dm_O2[int(t % interval)] = m_O2
        dm_NO3[int(t % interval)] = m_NO3
        dm_NO2[int(t % interval)] = m_NO2
        dm_NH4[int(t % interval)] = m_NH4
        dm_N2[int(t % interval)] = m_N2
        dm_Sd[int(t % interval)] = m_Sd
        
        # Record growth rates at each timestep
        du_Het[int(t % interval)] = u_Het
        du_Fac[int(t % interval)] = u_Fac
        du_1Den[int(t % interval)] = u_1Den
        du_2Den[int(t % interval)] = u_2Den
        du_3Den[int(t % interval)] = u_3Den
        du_AOO[int(t % interval)] = u_AOO
        du_NOO[int(t % interval)] = u_NOO
        du_AOX[int(t % interval)] = u_AOX
        
        # Record biomass at each timestep
        db_Het[int(t % interval)] = m_bHet
        db_Fac[int(t % interval)] = m_bFac
        db_1Den[int(t % interval)] = m_b1Den
        db_2Den[int(t % interval)] = m_b2Den
        db_3Den[int(t % interval)] = m_b3Den
        db_AOO[int(t % interval)] = m_bAOO
        db_NOO[int(t % interval)] = m_bNOO
        db_AOX[int(t % interval)] = m_bAOX
        
        # Record rates at each timestep
        dr_Het[int(t % interval)] = heterotrophy
        dr_HetAer[int(t % interval)] = aer_heterotrophy
        dr_O2C[int(t % interval)] = oxy_consumption
        dr_1Den[int(t % interval)] = den_nar
        dr_2Den[int(t % interval)] = den_nir
        dr_3Den[int(t % interval)] = den_full
        dr_AOO[int(t % interval)] = ammonia_ox
        dr_NOO[int(t % interval)] = nitrite_ox
        dr_AOX[int(t % interval)] = anammox_nh4
        
        
        # pulse OM and O2 into the chemostat
        if (t % int((1/dt * pulse_int))) == 0:
            m_Sd = m_Sd + pulse_Sd
            m_bHet = m_bHet + pulse_bHet
            m_bFac = m_bFac + pulse_bFac
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_Sd[i] = m_Sd 
            out_Sp[i] = m_Sp
            out_O2[i] = m_O2
            out_NO3[i] = m_NO3
            out_NO2[i] = m_NO2 
            out_NH4[i] = m_NH4 
            out_N2[i] = m_N2
            out_bHet[i] = m_bHet
            out_bFac[i] = m_bFac
            out_b1Den[i] = m_b1Den
            out_b2Den[i] = m_b2Den
            out_b3Den[i] = m_b3Den 
            out_bAOO[i] = m_bAOO
            out_bNOO[i] = m_bNOO
            out_bAOX[i] = m_bAOX
            out_uHet[i] = u_Het
            out_uFac[i] = u_Fac
            out_u1Den[i] = u_1Den
            out_u2Den[i] = u_2Den
            out_u3Den[i] = u_3Den 
            out_uAOO[i] = u_AOO
            out_uNOO[i] = u_NOO
            out_uAOX[i] = u_AOX
            out_facaer[i] = np.nanmean(facaer)
            out_facnoo_lim[i] = np.nanmean(facnoo_lim)
            out_maxnox[i] = np.nanmean(maxnox)
            out_rHet[i] = heterotrophy
            out_rHetAer[i] = aer_heterotrophy
            out_rO2C[i] = oxy_consumption
            out_r1Den[i] = den_nar
            out_r2Den[i] = den_nir
            out_r3Den[i] = den_full
            out_rAOO[i] = ammonia_ox
            out_rNOO[i] = nitrite_ox
            out_rAOX[i] = anammox_nh4
            
            
    return [out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, \
            out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, \
            out_uHet, out_uFac, out_u1Den, out_u2Den, out_u3Den, out_uAOO, out_uNOO, out_uAOX, \
            out_facaer, out_facnoo_lim, out_maxnox, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX, \
            dm_O2, dm_NO3, dm_NO2, dm_NH4, dm_N2, dm_Sd, \
            du_Het, du_Fac, du_1Den, du_2Den, du_3Den, du_AOO, du_NOO, du_AOX, \
            db_Het, db_Fac, db_1Den, db_2Den, db_3Den, db_AOO, db_NOO, db_AOX, \
            dr_Het, dr_HetAer, dr_O2C, dr_1Den, dr_2Den, dr_3Den, dr_AOO, dr_NOO, dr_AOX ]
        



@jit(nopython=True)
def DOMbreakdown(timesteps, nn_output, dt, dil, out_at_day, \
                 pulse_O2, pulse_doc, pulse_don, pulse_doh, pulse_doo, pulse_NH4, 
                 pulse_bHet1, pulse_bHet2, pulse_bHet3, pulse_int, \
                 p_O2, Vmax1_doc, Vmax2_doc, Vmax3_doc, \
                 K1_doc, K2_doc, K3_doc, \
                 alpha1, alpha2, alpha3, \
                 f1_ele, f2_ele, f3_ele, \
                 optNOSC1, optNOSC2, optNOSC3, \
                 stdNOSC1, stdNOSC2, stdNOSC3, \
                 Hb, Ob, Nb, Hp_fac, Op_fac, Np_fac, \
                 in_O2, in_doc, in_doh, in_doo, in_don, in_NH4, in_CO2, \
                 in_bHet1, in_bHet2, in_bHet3, 
                 ):
    '''
    ______________________________________________________________________
    0D chemostat model for aerobic degradation and partial oxidation of DOM
    ----------------------------------------------------------------------

    This model simulates three aerobic heterotrophic bacterial populations
    consuming a shared dissolved organic matter (DOM) pool.

    The DOM pool is represented by four elemental tracers:

        DOC : dissolved organic carbon
        DOH : organic hydrogen associated with DOM
        DOO : organic oxygen associated with DOM
        DON : dissolved organic nitrogen

    Bulk DOM stoichiometry is diagnosed as:

        H:C = DOH / DOC
        O:C = DOO / DOC
        N:C = DON / DOC

    and the nominal oxidation state of carbon is:

        NOSC = 2*(O:C) + 3*(N:C) - H:C

    Each bacterial type consumes a fraction of the bulk DOC pool according to
    a Gaussian preference function centred on an optimal NOSC.  The three
    availability fractions are normalised so that the sum of available DOC
    assigned to Het1, Het2 and Het3 equals total DOC.

    Each type converts consumed DOM into:

        - bacterial biomass
        - CO2
        - NH4, or consumes NH4 if organic N is insufficient
        - a partially oxidised DOM product

    The fraction of consumed organic carbon returned as partially oxidised DOM
    is controlled by alpha1, alpha2 and alpha3.  Higher alpha means more
    consumed carbon is returned to DOM and fewer electrons are available for
    respiration and biomass synthesis.

    INPUTS
    ------

    timesteps : int
        Number of model timesteps.

    nn_output : int
        Number of saved output intervals. Output arrays have length nn_output + 1.

    dt : float
        Timestep length in days.

    dil : float
        Chemostat dilution rate in day-1.

    out_at_day : float
        Output saving interval in days.

    pulse_O2 : float
        Amount of O2 added during each pulse.

    pulse_doc : float
        Amount of DOC added during each pulse.

    pulse_doh : float
        Amount of DOH added during each pulse.

    pulse_doo : float
        Amount of DOO added during each pulse.

    pulse_don : float
        Amount of DON added during each pulse.

    pulse_NH4 : float
        Amount of NH4 added during each pulse.

    pulse_bHet1 : float
        Biomass of heterotroph type 1 added during each pulse.

    pulse_bHet2 : float
        Biomass of heterotroph type 2 added during each pulse.

    pulse_bHet3 : float
        Biomass of heterotroph type 3 added during each pulse.

    pulse_int : float
        Pulse interval in days.

    p_O2 : float
        First-order O2 supply or uptake coefficient.

    Vmax_doc : float
        Maximum DOC uptake rate per unit bacterial biomass.

    K_doc : float
        Half-saturation coefficient for DOC uptake.

    alpha1, alpha2, alpha3 : float
        Fraction of consumed DOC returned as partially oxidised DOM by
        heterotroph types 1, 2 and 3.

    f1_ele, f2_ele, f3_ele : float
        Fraction of released electron equivalents routed to biosynthesis for
        heterotroph types 1, 2 and 3.

    optNOSC1, optNOSC2, optNOSC3 : float
        Optimal NOSC values for DOC availability to heterotroph types 1, 2 and 3.

    stdNOSC1, stdNOSC2, stdNOSC3 : float
        Widths of the Gaussian NOSC preference functions for heterotroph types
        1, 2 and 3.

    Hb : float
        Bacterial biomass H:C ratio.

    Ob : float
        Bacterial biomass O:C ratio.

    Nb : float
        Bacterial biomass N:C ratio.

    Hp_fac : float
        Multiplicative factor applied to bulk DOM H:C to define product H:C.
        Values < 1 represent dehydrogenation.

    Op_fac : float
        Multiplicative factor applied to bulk DOM O:C to define product O:C.
        Values > 1 represent oxygen enrichment.

    Np_fac : float
        Multiplicative factor applied to bulk DOM N:C to define product N:C.

    in_O2 : float
        Initial O2 concentration.

    in_doc : float
        Initial DOC concentration.

    in_doh : float
        Initial DOH concentration.

    in_doo : float
        Initial DOO concentration.

    in_don : float
        Initial DON concentration.

    in_NH4 : float
        Initial NH4 concentration.

    in_CO2 : float
        Initial CO2 or DIC concentration.

    in_bHet1 : float
        Initial biomass concentration of heterotroph type 1.

    in_bHet2 : float
        Initial biomass concentration of heterotroph type 2.

    in_bHet3 : float
        Initial biomass concentration of heterotroph type 3.

    OUTPUTS
    -------

    out_doc, out_doh, out_doo, out_don : array
        Saved DOM elemental tracer concentrations.

    out_O2, out_CO2, out_NH4 : array
        Saved inorganic tracer concentrations.

    out_bHet1, out_bHet2, out_bHet3 : array
        Saved bacterial biomass concentrations.

    out_uHet1, out_uHet2, out_uHet3 : array
        Saved specific growth rates of heterotroph types 1, 2 and 3.

    out_yHet1, out_yHet2, out_yHet3 : array
        Saved biomass yields, in mol C biomass per mol DOC consumed.

    out_f1_doc, out_f2_doc, out_f3_doc : array
        Saved fraction DOC availability for heterotroph types 1, 2 and 3.

    out_r1_doc, out_r2_doc, out_r3_doc : array
        Saved DOC consumption rates for heterotroph types 1, 2 and 3.

    out_r1_doh, out_r2_doh, out_r3_doh : array
        Saved DOH consumption rates for heterotroph types 1, 2 and 3.

    out_r1_doo, out_r2_doo, out_r3_doo : array
        Saved DOO consumption rates for heterotroph types 1, 2 and 3.

    out_r1_don, out_r2_don, out_r3_don : array
        Saved DON consumption rates for heterotroph types 1, 2 and 3.

    out_r1_O2, out_r2_O2, out_r3_O2 : array
        Saved O2 consumption rates for heterotroph types 1, 2 and 3.

    out_p1_CO2, out_p2_CO2, out_p3_CO2 : array
        Saved CO2 production rates for heterotroph types 1, 2 and 3.

    out_p1_NH4, out_p2_NH4, out_p3_NH4 : array
        Saved net NH4 production rates for heterotroph types 1, 2 and 3.
        Negative values indicate NH4 uptake.

    out_RQ1, out_RQ2, out_RQ3 : array
        Saved respiratory quotients for heterotroph types 1, 2 and 3,
        defined here as mol O2 consumed per mol CO2 produced.

    dm_* arrays : array
        High-frequency tracer records over the final output interval.

    du_Het* arrays : array
        High-frequency growth-rate records over the final output interval.

    db_Het* arrays : array
        High-frequency biomass records over the final output interval.

    dy_Het* arrays : array
        High-frequency biomass-yield records over the final output interval.

    df*, dr*, dp*, d*RQ arrays : array
        High-frequency DOC availability, rates of removal and production
        terms, and respiratory quotient records over the final output interval.
    '''
    
    epsi = 1e-20
    
    # transfer initial inputs to model variables
    m_doc = in_doc
    m_doh = in_doh
    m_doo = in_doo
    m_don = in_don
    m_O2 = in_O2
    m_CO2 = in_CO2
    m_NH4 = in_NH4
    m_bHet1 = in_bHet1
    m_bHet2 = in_bHet2
    m_bHet3 = in_bHet3
    
    # set the output arrays 
    out_doc = np.ones((int(nn_output)+1)) * np.nan
    out_doh = np.ones((int(nn_output)+1)) * np.nan
    out_doo = np.ones((int(nn_output)+1)) * np.nan
    out_don = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_bHet1 = np.ones((int(nn_output)+1)) * np.nan
    out_bHet2 = np.ones((int(nn_output)+1)) * np.nan
    out_bHet3 = np.ones((int(nn_output)+1)) * np.nan
    out_uHet1 = np.ones((int(nn_output)+1)) * np.nan
    out_uHet2 = np.ones((int(nn_output)+1)) * np.nan
    out_uHet3 = np.ones((int(nn_output)+1)) * np.nan
    out_yHet1 = np.ones((int(nn_output)+1)) * np.nan
    out_yHet2 = np.ones((int(nn_output)+1)) * np.nan
    out_yHet3 = np.ones((int(nn_output)+1)) * np.nan
    out_f1_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r1_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r1_doh = np.ones((int(nn_output)+1)) * np.nan
    out_r1_doo = np.ones((int(nn_output)+1)) * np.nan
    out_r1_don = np.ones((int(nn_output)+1)) * np.nan
    out_r1_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p1_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p1_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_f2_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r2_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r2_doh = np.ones((int(nn_output)+1)) * np.nan
    out_r2_doo = np.ones((int(nn_output)+1)) * np.nan
    out_r2_don = np.ones((int(nn_output)+1)) * np.nan
    out_r2_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p2_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p2_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_f3_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r3_doc = np.ones((int(nn_output)+1)) * np.nan
    out_r3_doh = np.ones((int(nn_output)+1)) * np.nan
    out_r3_doo = np.ones((int(nn_output)+1)) * np.nan
    out_r3_don = np.ones((int(nn_output)+1)) * np.nan
    out_r3_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p3_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p3_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_RQ1 = np.ones((int(nn_output)+1)) * np.nan
    out_RQ2 = np.ones((int(nn_output)+1)) * np.nan
    out_RQ3 = np.ones((int(nn_output)+1)) * np.nan
    
    # set the array for recording average aerobic or anaerobic activity of facultative population
    interval = int((1.0 / dt) * out_at_day)
    if interval < 1:
        interval = 1
    
    # set the arrays for activities over the final interval of the experiment
    dm_O2 = np.ones((interval)) * np.nan
    dm_NH4 = np.ones((interval)) * np.nan
    dm_CO2 = np.ones((interval)) * np.nan
    dm_doc = np.ones((interval)) * np.nan
    dm_doh = np.ones((interval)) * np.nan
    dm_doo = np.ones((interval)) * np.nan
    dm_don = np.ones((interval)) * np.nan
    du_Het1 = np.ones((interval)) * np.nan
    du_Het2 = np.ones((interval)) * np.nan
    du_Het3 = np.ones((interval)) * np.nan
    db_Het1 = np.ones((interval)) * np.nan
    db_Het2 = np.ones((interval)) * np.nan
    db_Het3 = np.ones((interval)) * np.nan
    dy_Het1 = np.ones((interval)) * np.nan
    dy_Het2 = np.ones((interval)) * np.nan
    dy_Het3 = np.ones((interval)) * np.nan
    df1_doc = np.ones((interval)) * np.nan
    dr1_doc = np.ones((interval)) * np.nan
    dr1_doh = np.ones((interval)) * np.nan
    dr1_doo = np.ones((interval)) * np.nan
    dr1_don = np.ones((interval)) * np.nan
    dr1_O2 = np.ones((interval)) * np.nan
    dp1_CO2 = np.ones((interval)) * np.nan
    dp1_NH4 = np.ones((interval)) * np.nan
    d1_RQ = np.ones((interval)) * np.nan
    df2_doc = np.ones((interval)) * np.nan
    dr2_doc = np.ones((interval)) * np.nan
    dr2_doh = np.ones((interval)) * np.nan
    dr2_doo = np.ones((interval)) * np.nan
    dr2_don = np.ones((interval)) * np.nan
    dr2_O2 = np.ones((interval)) * np.nan
    dp2_CO2 = np.ones((interval)) * np.nan
    dp2_NH4 = np.ones((interval)) * np.nan
    d2_RQ = np.ones((interval)) * np.nan
    df3_doc = np.ones((interval)) * np.nan
    dr3_doc = np.ones((interval)) * np.nan
    dr3_doh = np.ones((interval)) * np.nan
    dr3_doo = np.ones((interval)) * np.nan
    dr3_don = np.ones((interval)) * np.nan
    dr3_O2 = np.ones((interval)) * np.nan
    dp3_CO2 = np.ones((interval)) * np.nan
    dp3_NH4 = np.ones((interval)) * np.nan
    d3_RQ = np.ones((interval)) * np.nan
    
    # record the initial conditions
    i = 0
    out_doc[i] = m_doc 
    out_doh[i] = m_doh
    out_doo[i] = m_doo
    out_don[i] = m_don
    out_O2[i] = m_O2
    out_NH4[i] = m_NH4 
    out_CO2[i] = m_CO2
    out_bHet1[i] = m_bHet1
    out_bHet2[i] = m_bHet2
    out_bHet3[i] = m_bHet3
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        # relative fractional availability of DOC
        nosc = 4.0 - (4.0*m_doc + m_doh - 2.0*m_doo - 3*m_don)/(m_doc+epsi)
        f1_doc = 1.0/(stdNOSC1 * np.sqrt(2*np.pi)) * np.exp( -0.5 *((nosc - optNOSC1) / stdNOSC1)**2.0) 
        f2_doc = 1.0/(stdNOSC2 * np.sqrt(2*np.pi)) * np.exp( -0.5 *((nosc - optNOSC2) / stdNOSC2)**2.0) 
        f3_doc = 1.0/(stdNOSC3 * np.sqrt(2*np.pi)) * np.exp( -0.5 *((nosc - optNOSC3) / stdNOSC3)**2.0) 
        m1_doc = m_doc * f1_doc / (f1_doc + f2_doc + f3_doc + epsi)
        m2_doc = m_doc * f2_doc / (f1_doc + f2_doc + f3_doc + epsi)
        m3_doc = m_doc * f3_doc / (f1_doc + f2_doc + f3_doc + epsi)

        # uptake rates (p)
        u_O2 = p_O2 * m_O2                                 # mol O2 / day
        u1_doc = Vmax1_doc * m1_doc / (K1_doc + m1_doc + epsi)  # mol C-Org / day
        u2_doc = Vmax2_doc * m2_doc / (K2_doc + m2_doc + epsi)  # mol C-Org / day
        u3_doc = Vmax3_doc * m3_doc / (K3_doc + m3_doc + epsi)  # mol C-Org / day
        
        # electrons per atom of carbon in biomass (b) and substrate (s)
        def ele_potential(c,h,o,n):
            return 4*c + 1*h - 2*o - 3*n
        eb = ele_potential(1.0, Hb, Ob, Nb)
        Hs = m_doh/(m_doc+epsi); Os = m_doo/(m_doc+epsi); Ns = m_don/(m_doc+epsi)
        Hp = Hs*Hp_fac; Op = Os*Op_fac; Np = Ns*Np_fac 
        es = ele_potential(1.0, Hs, Os, Ns)
        ep = ele_potential(1.0, Hp, Op, Np)
        
        ### residual electrons available to bacteria after some loss of product
        e1_res = np.fmax(0.0, es - alpha1 * ep) # Cap electron release to be non-negative
        e2_res = np.fmax(0.0, es - alpha2 * ep) # Cap electron release to be non-negative
        e3_res = np.fmax(0.0, es - alpha3 * ep) # Cap electron release to be non-negative
        
        ### Sequential partial oxidation of substrate by bacteria gives yields of...
        y1_pB = f1_ele * e1_res/eb 
        y2_pB = f2_ele * e2_res/eb 
        y3_pB = f3_ele * e3_res/eb 
        y1_pB = np.fmin(y1_pB, 1.0 - alpha1) # Cap C yields to a theoretical maximum
        y2_pB = np.fmin(y2_pB, 1.0 - alpha2) # Cap C yields to a theoretical maximum
        y3_pB = np.fmin(y3_pB, 1.0 - alpha3) # Cap C yields to a theoretical maximum
        
        y1_pP = alpha1 / y1_pB
        y2_pP = alpha2 / y2_pB
        y3_pP = alpha3 / y3_pB
        
        y1_rdoc = 1.0/y1_pB
        y1_rO2 = e1_res*(1-f1_ele)/4.0 / y1_pB
        y1_pCO2 = (1.0 - alpha1 - y1_pB) / y1_pB
        y1_pNH4 = (Ns - alpha1*Np - Nb * y1_pB) / y1_pB
  
        y2_rdoc = 1.0/y2_pB
        y2_rO2 = e2_res*(1-f2_ele)/4.0 / y2_pB
        y2_pCO2 = (1.0 - alpha2 - y2_pB) / y2_pB
        y2_pNH4 = (Ns - alpha2*Np - Nb * y2_pB) / y2_pB

        y3_rdoc = 1.0/y3_pB
        y3_rO2 = e3_res*(1-f3_ele)/4.0 / y3_pB
        y3_pCO2 = (1.0 - alpha3 - y3_pB) / y3_pB
        y3_pNH4 = (Ns - alpha3*Np - Nb * y3_pB) / y3_pB

        ### Respiratory quotient (mol O2 consumed per mol CO2 produced)
        RQ1 = y1_rO2 / y1_pCO2
        RQ2 = y2_rO2 / y2_pCO2
        RQ3 = y3_rO2 / y3_pCO2
        
        # growth rates (u)
        g1_O2 = u_O2 / y1_rO2
        g1_C = u1_doc / y1_rdoc
        g1min = np.fmin(g1_C, g1_O2)
        g2_O2 = u_O2 / y2_rO2
        g2_C = u2_doc / y2_rdoc
        g2min = np.fmin(g2_C, g2_O2)
        g3_O2 = u_O2 / y3_rO2
        g3_C = u3_doc / y3_rdoc
        g3min = np.fmin(g3_C, g3_O2)
        mu_Het1 = np.fmax(0.0, g1min)
        mu_Het2 = np.fmax(0.0, g2min)
        mu_Het3 = np.fmax(0.0, g3min)
        
        ### rates of substrate consumption
        oxy_cons_1 = mu_Het1 * m_bHet1 * y1_rO2
        oxy_cons_2 = mu_Het2 * m_bHet2 * y2_rO2
        oxy_cons_3 = mu_Het3 * m_bHet3 * y3_rO2
        doc_cons_1 = mu_Het1 * m_bHet1 * y1_rdoc
        doc_cons_2 = mu_Het2 * m_bHet2 * y2_rdoc
        doc_cons_3 = mu_Het3 * m_bHet3 * y3_rdoc
        doh_cons_1 = doc_cons_1 * Hs # Type 1 consumes bulk substrate stoich
        doh_cons_2 = doc_cons_2 * Hs # Type 2 consumes bulk substrate stoich
        doh_cons_3 = doc_cons_3 * Hs # Type 3 consumes bulk substrate stoich
        doo_cons_1 = doc_cons_1 * Os
        doo_cons_2 = doc_cons_2 * Os
        doo_cons_3 = doc_cons_3 * Os
        don_cons_1 = doc_cons_1 * Ns
        don_cons_2 = doc_cons_2 * Ns
        don_cons_3 = doc_cons_3 * Ns
        # rates of substrate production
        doc_prod_1 = mu_Het1 * m_bHet1 * y1_pP
        doc_prod_2 = mu_Het2 * m_bHet2 * y2_pP
        doc_prod_3 = mu_Het3 * m_bHet3 * y3_pP
        doh_prod_1 = doc_prod_1 * Hp
        doh_prod_2 = doc_prod_2 * Hp
        doh_prod_3 = doc_prod_3 * Hp
        doo_prod_1 = doc_prod_1 * Op
        doo_prod_2 = doc_prod_2 * Op
        doo_prod_3 = doc_prod_3 * Op
        don_prod_1 = doc_prod_1 * Np
        don_prod_2 = doc_prod_2 * Np
        don_prod_3 = doc_prod_3 * Np
        co2_prod_1  = mu_Het1 * m_bHet1 * y1_pCO2
        co2_prod_2  = mu_Het2 * m_bHet2 * y2_pCO2
        co2_prod_3  = mu_Het3 * m_bHet3 * y3_pCO2
        nh4_prod_1  = mu_Het1 * m_bHet1 * y1_pNH4
        nh4_prod_2  = mu_Het2 * m_bHet2 * y2_pNH4
        nh4_prod_3  = mu_Het3 * m_bHet3 * y3_pNH4        
        
        # Dissolved organic carbon, hydrogen, oxygen and nitrogen
        ddt_doc = dil * (in_doc - m_doc) \
                  + doc_prod_1 + doc_prod_2 + doc_prod_3 \
                  - doc_cons_1 - doc_cons_2 - doc_cons_3 
        ddt_doh = dil * (in_doh - m_doh) \
                  + doh_prod_1 + doh_prod_2 + doh_prod_3 \
                  - doh_cons_1 - doh_cons_2 - doh_cons_3
        ddt_doo = dil * (in_doo - m_doo) \
                  + doo_prod_1 + doo_prod_2 + doo_prod_3 \
                  - doo_cons_1 - doo_cons_2 - doo_cons_3
        ddt_don = dil * (in_don - m_don) \
                  + don_prod_1 + don_prod_2 + don_prod_3 \
                  - don_cons_1 - don_cons_2 - don_cons_3
        # Dissolved oxygen
        ddt_O2 = dil * (in_O2 - m_O2) \
                 - oxy_cons_1 - oxy_cons_2 - oxy_cons_3
        # Ammonium 
        ddt_NH4 = dil * (in_NH4 - m_NH4) \
                 + nh4_prod_1 + nh4_prod_2 + nh4_prod_3
        # CO2 (new DIC)
        ddt_CO2 = dil * (in_CO2 - m_CO2) \
                 + co2_prod_1 + co2_prod_2 + co2_prod_3
        # Biomass of aerobic heterotrophs
        ddt_bHet1 = dil * (-m_bHet1) + mu_Het1 * m_bHet1
        ddt_bHet2 = dil * (-m_bHet2) + mu_Het2 * m_bHet2
        ddt_bHet3 = dil * (-m_bHet3) + mu_Het3 * m_bHet3
        
        
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        ### apply changes to state variables ###
        m_doc = m_doc + ddt_doc * dt
        m_doh = m_doh + ddt_doh * dt
        m_doo = m_doo + ddt_doo * dt
        m_don = m_don + ddt_don * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_CO2 = m_CO2 + ddt_CO2 * dt
        m_bHet1 = m_bHet1 + ddt_bHet1 * dt
        m_bHet2 = m_bHet2 + ddt_bHet2 * dt
        m_bHet3 = m_bHet3 + ddt_bHet3 * dt
        
        # Record tracer concentrations at each timestep
        dm_O2[int(t % interval)] = m_O2
        dm_NH4[int(t % interval)] = m_NH4
        dm_CO2[int(t % interval)] = m_CO2
        dm_doc[int(t % interval)] = m_doc
        dm_doh[int(t % interval)] = m_doh
        dm_doo[int(t % interval)] = m_doo
        dm_don[int(t % interval)] = m_don
        
        # Record growth rates at each timestep
        du_Het1[int(t % interval)] = mu_Het1
        du_Het2[int(t % interval)] = mu_Het2
        du_Het3[int(t % interval)] = mu_Het3
        
        # Record biomass at each timestep
        db_Het1[int(t % interval)] = m_bHet1
        db_Het2[int(t % interval)] = m_bHet2
        db_Het3[int(t % interval)] = m_bHet3
        
        # Record biomass yields at each timestep
        dy_Het1[int(t % interval)] = y1_pB
        dy_Het2[int(t % interval)] = y2_pB
        dy_Het3[int(t % interval)] = y3_pB
        
        # Record rates at each timestep
        df1_doc[int(t % interval)] = f1_doc
        dr1_doc[int(t % interval)] = doc_cons_1
        dr1_doh[int(t % interval)] = doh_cons_1
        dr1_doo[int(t % interval)] = doo_cons_1
        dr1_don[int(t % interval)] = don_cons_1
        dr1_O2[int(t % interval)] = oxy_cons_1
        dp1_CO2[int(t % interval)] = co2_prod_1
        dp1_NH4[int(t % interval)] = nh4_prod_1
        d1_RQ[int(t % interval)] = RQ1
        df2_doc[int(t % interval)] = f2_doc
        dr2_doc[int(t % interval)] = doc_cons_2
        dr2_doh[int(t % interval)] = doh_cons_2
        dr2_doo[int(t % interval)] = doo_cons_2
        dr2_don[int(t % interval)] = don_cons_2
        dr2_O2[int(t % interval)] = oxy_cons_2
        dp2_CO2[int(t % interval)] = co2_prod_2
        dp2_NH4[int(t % interval)] = nh4_prod_2
        d2_RQ[int(t % interval)] = RQ2
        df3_doc[int(t % interval)] = f3_doc
        dr3_doc[int(t % interval)] = doc_cons_3
        dr3_doh[int(t % interval)] = doh_cons_3
        dr3_doo[int(t % interval)] = doo_cons_3
        dr3_don[int(t % interval)] = don_cons_3
        dr3_O2[int(t % interval)] = oxy_cons_3
        dp3_CO2[int(t % interval)] = co2_prod_3
        dp3_NH4[int(t % interval)] = nh4_prod_3
        d3_RQ[int(t % interval)] = RQ3
        

        # pulse OM and O2 into the chemostat
        pulse_interval = int((1.0 / dt) * pulse_int)
        if pulse_interval < 1:
            pulse_interval = 1
        if (t % pulse_interval) == 0:
            m_doc = m_doc + pulse_doc
            m_doh = m_doh + pulse_doh
            m_doo = m_doo + pulse_doo
            m_don = m_don + pulse_don
            m_NH4 = m_NH4 + pulse_NH4
            m_bHet1 = m_bHet1 + pulse_bHet1
            m_bHet2 = m_bHet2 + pulse_bHet2
            m_bHet3 = m_bHet3 + pulse_bHet3
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_doc[i] = m_doc 
            out_doh[i] = m_doh
            out_doo[i] = m_doo
            out_don[i] = m_don
            out_O2[i] = m_O2
            out_NH4[i] = m_NH4 
            out_CO2[i] = m_CO2
            out_bHet1[i] = m_bHet1
            out_bHet2[i] = m_bHet2
            out_bHet3[i] = m_bHet3
            out_uHet1[i] = mu_Het1
            out_uHet2[i] = mu_Het2
            out_uHet3[i] = mu_Het3
            out_yHet1[i] = y1_pB
            out_yHet2[i] = y2_pB
            out_yHet3[i] = y3_pB
            out_f1_doc[i] = f1_doc
            out_r1_doc[i] = doc_cons_1
            out_r1_doh[i] = doh_cons_1
            out_r1_doo[i] = doo_cons_1
            out_r1_don[i] = don_cons_1
            out_r1_O2[i] = oxy_cons_1
            out_p1_CO2[i] = co2_prod_1
            out_p1_NH4[i] = nh4_prod_1
            out_RQ1[i] = RQ1
            out_f2_doc[i] = f2_doc
            out_r2_doc[i] = doc_cons_2
            out_r2_doh[i] = doh_cons_2
            out_r2_doo[i] = doo_cons_2
            out_r2_don[i] = don_cons_2
            out_r2_O2[i] = oxy_cons_2
            out_p2_CO2[i] = co2_prod_2
            out_p2_NH4[i] = nh4_prod_2
            out_RQ2[i] = RQ2
            out_f3_doc[i] = f3_doc
            out_r3_doc[i] = doc_cons_3
            out_r3_doh[i] = doh_cons_3
            out_r3_doo[i] = doo_cons_3
            out_r3_don[i] = don_cons_3
            out_r3_O2[i] = oxy_cons_3
            out_p3_CO2[i] = co2_prod_3
            out_p3_NH4[i] = nh4_prod_3
            out_RQ3[i] = RQ3
            
            
            
    return [out_doc, out_doh, out_doo, out_don, out_O2, out_CO2, out_NH4, \
            out_bHet1, out_bHet2, out_bHet3, \
            out_uHet1, out_uHet2, out_uHet3,\
            out_yHet1, out_yHet2, out_yHet3,\
            out_f1_doc, out_r1_doc, out_r1_doh, out_r1_doo, out_r1_don, out_r1_O2, out_p1_CO2, out_p1_NH4, out_RQ1, \
            out_f2_doc, out_r2_doc, out_r2_doh, out_r2_doo, out_r2_don, out_r2_O2, out_p2_CO2, out_p2_NH4, out_RQ2, \
            out_f3_doc, out_r3_doc, out_r3_doh, out_r3_doo, out_r3_don, out_r3_O2, out_p3_CO2, out_p3_NH4, out_RQ3, \
            dm_O2, dm_CO2, dm_NH4, dm_doc, dm_doh, dm_doo, dm_don, \
            du_Het1, du_Het2, du_Het3, \
            db_Het1, db_Het2, db_Het3, \
            dy_Het1, dy_Het2, dy_Het3, \
            df1_doc, dr1_doc, dr1_doh, dr1_doo, dr1_don, dr1_O2, dp1_CO2, dp1_NH4, d1_RQ, \
            df2_doc, dr2_doc, dr2_doh, dr2_doo, dr2_don, dr2_O2, dp2_CO2, dp2_NH4, d2_RQ, \
            df3_doc, dr3_doc, dr3_doh, dr3_doo, dr3_don, dr3_O2, dp3_CO2, dp3_NH4, d3_RQ ]


@jit(nopython=True)
def DOMlDOMs(timesteps, nn_output, dt, dil, out_at_day, \
             pulse_O2, \
             pulse_ldoc, pulse_ldon, \
             pulse_rdoc, pulse_rdon, \
             pulse_sdoc, pulse_sdon, \
             pulse_NH4, \
             pulse_baoa, pulse_blHet, pulse_brHet, pulse_bsHet, pulse_int, \
             aoa_p_O2, lHet_p_O2, rHet_p_O2, sHet_p_O2, \
             lHet_Vmax_doc, rHet_Vmax_doc, sHet_Vmax_doc, \
             lHet_K_ldoc, rHet_K_rdoc, sHet_K_sdoc, \
             aoa_Vmax_NH4, lHet_Vmax_NH4, rHet_Vmax_NH4, sHet_Vmax_NH4, \
             aoa_K_NH4, lHet_K_NH4, rHet_K_NH4, sHet_K_NH4, \
             aoa_y_NH4, aoa_y_O2, lHet_alpha, lHet_beta, \
             lHet_fele, rHet_fele, sHet_fele, \
             Hl, Ol, Hr, Or, Hs, Os, Hb, Ob, Nb, \
             donpersist, \
             in_O2, in_ldoc, in_ldon, \
             in_rdoc, in_rdon, \
             in_sdoc, in_sdon, \
             in_NH4, in_CO2, \
             in_baoa, in_blHet, in_brHet, in_bsHet,
             flow_through=True
             ):
    '''
    ______________________________________________________________________
    0D chemostat model for aerobic degradation and partial oxidation of DOM
    ----------------------------------------------------------------------

    This model simulates three aerobic heterotrophic bacterial populations
    consuming three dissolved organic matter (DOM) pools.

    The DOM pool is represented by three elemental tracers:

        lDOC(N) : long chain dissolved organic carbon (nitrogen)
        rDOC(N) : long chain, reworked dissolved organic carbon (nitrogen)
        sDOC(N) : short chain dissolved organic carbon (nitrogen)
        
    lHet bacteria consume the lDOM pool and partially oxidizes it to sDOM and rDOM
    rHet bacteria consume the rDOM pool and partially oxidizes it to sDOM
    sHet bacteria consume the sDOM pool and fully oxidizes it to CO2
    
    Bacteria converts consumed DOM into:

        - biomass
        - CO2
        - NH4, or consumes NH4 if organic N is insufficient
        - a partially oxidised DOM product in the case of lHet and rHet

    The fraction of consumed organic carbon returned as partially oxidised sDOM
    is controlled by alpha. Higher alpha means more consumed carbon is routed to 
    sDOM and fewer electrons are available for respiration and biomass synthesis.

    INPUTS
    ------

    timesteps : int
        Number of model timesteps.

    nn_output : int
        Number of saved output intervals. Output arrays have length nn_output + 1.

    dt : float
        Timestep length in days.

    dil : float
        Chemostat dilution rate in day-1.

    out_at_day : float
        Output saving interval in days.

    pulse_O2 : float
        Amount of O2 added during each pulse.

    pulse_ldoc : float
        Amount of lDOC added during each pulse.

    pulse_sdoc : float
            Amount of sDOC added during each pulse.
    
    pulse_NH4 : float
        Amount of NH4 added during each pulse.

    pulse_blHet : float
        Biomass of long-chain consuming heterotroph added during each pulse.

    pulse_bsHet : float
            Biomass of short-chain consuming heterotroph added during each pulse.
    
    pulse_int : float
        Pulse interval in days.

    lHet_p_O2, sHet_p_O2 : float
        First-order O2 supply or uptake coefficient.

    lHet_Vmax_doc, sHet_Vmax_doc : float
        Maximum DOC uptake rate per unit bacterial biomass.

    lHet_K_ldoc, lHet_K_rdoc, sHet_K_sdoc : float
        Half-saturation coefficient for DOC uptake.

    lHet_alpha : float
        Fraction of consumed DOC returned as partially oxidised DOM 

    lHet_fele, sHet_fele : float
        Fraction of released electron equivalents routed to biosynthesis
        
    Hl : float
        Substrate H:C ratio.

    Ol : float
        Substrate O:C ratio.

    Hb : float
        Bacterial biomass H:C ratio.

    Ob : float
        Bacterial biomass O:C ratio.

    Nb : float
        Bacterial biomass N:C ratio.

    Hs : float
        short-chain Substrate H:C ratio.

    Os : float
        short-chain Substrate O:C ratio.

    in_O2 : float
        Initial O2 concentration.

    in_ldoc, in_ldon : float
        Initial lDOC, lDON concentration.

    in_sdoc, in_sdon : float
        Initial sDOC, sDON concentration.

    in_NH4 : float
        Initial NH4 concentration.

    in_CO2 : float
        Initial CO2 or DIC concentration.

    in_blHet : float
        Initial biomass concentration of the long-chain consuming heterotroph.

    in_bsHet : float
            Initial biomass concentration of the short-chain consuming heterotroph.
    

    OUTPUTS
    -------

    out_ldoc, out_sdoc : array
        Saved DOC tracer concentrations.

    out_O2, out_CO2, out_NH4 : array
        Saved inorganic tracer concentrations.

    out_blHet, out_bsHet : array
        Saved bacterial biomass concentration.

    out_ulHet, out_usHet : array
        Saved specific growth rate of heterotroph.

    out_ylHet, out_ysHet : array
        Saved biomass yield, in mol C biomass per mol DOC consumed.

    out_c_lHet_ldoc, out_c_sHet_sdoc : array
        Saved DOC consumption rate for heterotroph.

    out_c_lHet_O2, out_c_sHet_O2 : array
        Saved O2 consumption rate for heterotroph.

    out_p_lHet_CO2, out_p_sHet_CO2 : array
        Saved CO2 production rates for heterotroph.

    out_p_lHet_sdoc : array
            Saved net sDOC production rates for heterotroph.
            
    out_p_lHet_NH4, out_p_sHet_NH4 : array
        Saved net NH4 production rates for heterotroph.
        Negative values indicate NH4 uptake.

    out_lHet_RQ, out_sHet_RQ : array
        Saved respiratory quotients for heterotroph types 1, 2 and 3,
        defined here as mol O2 consumed per mol CO2 produced.

    dm_* arrays : array
        High-frequency tracer records over the final output interval.

    du_lHet*, du_sHet* arrays : array
        High-frequency growth-rate records over the final output interval.

    db_lHet*, db_sHet* arrays : array
        High-frequency biomass records over the final output interval.

    dy_lHet*, dy_sHet* arrays : array
        High-frequency biomass-yield records over the final output interval.

    df*, dc*, dp*, d*RQ arrays : array
        High-frequency DOC availability, rates of consumption and production
        terms, and respiratory quotient records over the final output interval.
    '''
    
    xx = 0.0
    if flow_through:
        xx = 1.0
    
    epsi = 1e-20
    
    # transfer initial inputs to model variables
    m_ldoc = in_ldoc
    m_ldon = in_ldon
    m_rdoc = in_rdoc
    m_rdon = in_rdon
    m_sdoc = in_sdoc
    m_sdon = in_sdon
    m_O2 = in_O2
    m_CO2 = in_CO2
    m_NH4 = in_NH4
    m_baoa = in_baoa
    m_blHet = in_blHet
    m_brHet = in_brHet
    m_bsHet = in_bsHet
        
    # set the output arrays 
    out_ldoc = np.ones((int(nn_output)+1)) * np.nan
    out_ldon = np.ones((int(nn_output)+1)) * np.nan
    out_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_baoa = np.ones((int(nn_output)+1)) * np.nan
    out_uaoa = np.ones((int(nn_output)+1)) * np.nan
    out_blHet = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_ylHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_ldoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_ldon = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_lHet_RQ = np.ones((int(nn_output)+1)) * np.nan
    out_brHet = np.ones((int(nn_output)+1)) * np.nan
    out_urHet = np.ones((int(nn_output)+1)) * np.nan
    out_urHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_urHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_yrHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_rHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_rHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_rHet_RQ = np.ones((int(nn_output)+1)) * np.nan
    out_bsHet = np.ones((int(nn_output)+1)) * np.nan
    out_usHet = np.ones((int(nn_output)+1)) * np.nan
    out_usHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_usHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_ysHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_sHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_sHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_sHet_RQ = np.ones((int(nn_output)+1)) * np.nan
        
    
    # set the array for recording average aerobic or anaerobic activity of facultative population
    interval = int((1.0 / dt) * out_at_day)
    if interval < 1:
        interval = 1
    
    # set the arrays for activities over the final interval of the experiment
    dm_O2 = np.ones((interval)) * np.nan
    dm_NH4 = np.ones((interval)) * np.nan
    dm_CO2 = np.ones((interval)) * np.nan
    dm_ldoc = np.ones((interval)) * np.nan
    dm_ldon = np.ones((interval)) * np.nan
    dm_rdoc = np.ones((interval)) * np.nan
    dm_rdon = np.ones((interval)) * np.nan
    dm_sdoc = np.ones((interval)) * np.nan
    dm_sdon = np.ones((interval)) * np.nan
    db_aoa = np.ones((interval)) * np.nan
    du_aoa = np.ones((interval)) * np.nan
    du_lHet = np.ones((interval)) * np.nan
    du_lHet_C = np.ones((interval)) * np.nan
    du_lHet_N = np.ones((interval)) * np.nan
    db_lHet = np.ones((interval)) * np.nan
    dy_lHet = np.ones((interval)) * np.nan
    dc_lHet_ldoc = np.ones((interval)) * np.nan
    dc_lHet_ldon = np.ones((interval)) * np.nan
    dc_lHet_O2 = np.ones((interval)) * np.nan
    dp_lHet_sdoc = np.ones((interval)) * np.nan
    dp_lHet_sdon = np.ones((interval)) * np.nan
    dp_lHet_rdoc = np.ones((interval)) * np.nan
    dp_lHet_rdon = np.ones((interval)) * np.nan
    dp_lHet_CO2 = np.ones((interval)) * np.nan
    dp_lHet_NH4 = np.ones((interval)) * np.nan
    d_lHet_RQ = np.ones((interval)) * np.nan
    du_rHet = np.ones((interval)) * np.nan
    du_rHet_C = np.ones((interval)) * np.nan
    du_rHet_N = np.ones((interval)) * np.nan
    db_rHet = np.ones((interval)) * np.nan
    dy_rHet = np.ones((interval)) * np.nan
    dc_rHet_rdoc = np.ones((interval)) * np.nan
    dc_rHet_rdon = np.ones((interval)) * np.nan
    dc_rHet_O2 = np.ones((interval)) * np.nan
    dp_rHet_CO2 = np.ones((interval)) * np.nan
    dp_rHet_NH4 = np.ones((interval)) * np.nan
    d_rHet_RQ = np.ones((interval)) * np.nan
    du_sHet = np.ones((interval)) * np.nan
    du_sHet_C = np.ones((interval)) * np.nan
    du_sHet_N = np.ones((interval)) * np.nan
    db_sHet = np.ones((interval)) * np.nan
    dy_sHet = np.ones((interval)) * np.nan
    dc_sHet_sdoc = np.ones((interval)) * np.nan
    dc_sHet_sdon = np.ones((interval)) * np.nan
    dc_sHet_O2 = np.ones((interval)) * np.nan
    dp_sHet_CO2 = np.ones((interval)) * np.nan
    dp_sHet_NH4 = np.ones((interval)) * np.nan
    d_sHet_RQ = np.ones((interval)) * np.nan
        
    
    # record the initial conditions
    i = 0
    out_ldoc[i] = m_ldoc
    out_ldon[i] = m_ldon
    out_rdoc[i] = m_rdoc
    out_rdon[i] = m_rdon
    out_sdoc[i] = m_sdoc
    out_sdon[i] = m_sdon
    out_O2[i] = m_O2
    out_NH4[i] = m_NH4 
    out_CO2[i] = m_CO2
    out_baoa[i] = m_baoa
    out_blHet[i] = m_blHet
    out_brHet[i] = m_brHet
    out_bsHet[i] = m_bsHet
        
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        # uptake rates (p)
        u_aoa_O2    = aoa_p_O2 * m_O2                                        # mol O2 / day
        u_aoa_NH4   = aoa_Vmax_NH4 * m_NH4 / (aoa_K_NH4 + m_NH4 + epsi)      # mol N / day
        u_lHet_O2   = lHet_p_O2 * m_O2                                       # mol O2 / day
        u_lHet_ldoc = lHet_Vmax_doc * m_ldoc / (lHet_K_ldoc + m_ldoc + epsi)  # mol C-Org / day
        u_lHet_NH4  = lHet_Vmax_NH4 * m_NH4 / (lHet_K_NH4 + m_NH4 + epsi)    # mol N / day
        u_rHet_O2   = rHet_p_O2 * m_O2                                       # mol O2 / day
        u_rHet_rdoc = rHet_Vmax_doc * m_rdoc / (rHet_K_rdoc + m_rdoc + epsi)  # mol C-Org / day
        u_rHet_NH4  = rHet_Vmax_NH4 * m_NH4 / (rHet_K_NH4 + m_NH4 + epsi)    # mol N / day
        u_sHet_O2   = sHet_p_O2 * m_O2                                       # mol O2 / day
        u_sHet_sdoc = sHet_Vmax_doc * m_sdoc / (sHet_K_sdoc + m_sdoc + epsi)  # mol C-Org / day
        u_sHet_NH4  = sHet_Vmax_NH4 * m_NH4 / (sHet_K_NH4 + m_NH4 + epsi)    # mol N / day

        ### ------------------------------------------------------------------- ###
        ### electrons per atom of carbon in biomass (b) and substrate (l, r, s) ###
        ### ------------------------------------------------------------------- ###
        def ele_potential(c,h,o,n):
            return 4*c + 1*h - 2*o - 3*n
        Nl = m_ldon / np.fmax(epsi, m_ldoc) # dynamic ratio of N:C in lDOM
        Nr = m_rdon / np.fmax(epsi, m_rdoc) # dynamic ratio of N:C in rDOM
        Ns = m_sdon / np.fmax(epsi, m_sdoc) # dynamic ratio of N:C in sDOM
        e_bac = ele_potential(1.0, Hb, Ob, Nb)
        e_ldom = ele_potential(1.0, Hl, Ol, Nl)
        e_rdom = ele_potential(1.0, Hr, Or, Nr)
        e_sdom = ele_potential(1.0, Hs, Os, Ns)
        # Electron potential of the partially oxidized product (sDOM) from lDOM and rDOM
        e_sdom_l = ele_potential(1.0, Hs, Os, Nl * donpersist)
        e_rdom_l = ele_potential(1.0, Hr, Or, Nl * donpersist)
        ### residual electrons available to sharing bacteria after some loss of product
        e_lres = np.fmax(0.0, e_ldom 
                              - lHet_alpha * lHet_beta * e_sdom_l 
                              - lHet_alpha * (1.0 - lHet_beta) * e_rdom_l)


        ### ------------------------------------ ###
        ### compute resource yields and products ###
        ### ------------------------------------ ###
          
        # initalize some values
        y_lHet_cdoc = 0.0; y_rHet_cdoc = 0.0; y_sHet_cdoc = 0.0 # reinitialise for safety
      
        ### Sharing bacteria (do partial oxidation, so produce sDOM and rDOM and CO2)
        y_lHet_pB = np.fmax(0.0, np.fmin(lHet_fele * e_lres/e_bac, 1.0 - lHet_alpha)) # Cap C yields to a theoretical maximum
        if y_lHet_pB > 0.0:
            y_lHet_cdoc = 1.0/y_lHet_pB  # consumption coefficient of lDOC
        y_lHet_coxy = np.fmax(0.0, e_lres - y_lHet_pB * e_bac)/4.0 * y_lHet_cdoc # consumption coefficient of O2
        y_lHet_psdoc = lHet_alpha *lHet_beta * y_lHet_cdoc  # production coefficient of sDOM
        y_lHet_prdoc = lHet_alpha * (1.0 - lHet_beta) * y_lHet_cdoc  # production coefficient of rDOM
        y_lHet_pco2 = (1.0 - lHet_alpha) * y_lHet_cdoc - 1.0
        y_lHet_pnh4 = (Nl - lHet_alpha*Nl*donpersist) * y_lHet_cdoc - Nb  # production coefficient of NH4
        y_lHet_psdon = (lHet_alpha*lHet_beta*Nl*donpersist) * y_lHet_cdoc  # production coefficient of sDON
        y_lHet_prdon = (lHet_alpha*(1.0-lHet_beta)*Nl*donpersist) * y_lHet_cdoc  # production coefficient of rDON
        ### complete oxidation of long-chain, reworked DOM by selfish bacteria
        y_rHet_pB = rHet_fele * e_rdom/e_bac
        if y_rHet_pB > 0.0:
            y_rHet_cdoc = 1.0/y_rHet_pB  # consumption coefficient of rDOC
        y_rHet_coxy = np.fmax(0.0, e_rdom - y_rHet_pB * e_bac)/4.0 * y_rHet_cdoc # consumption coefficient of O2
        y_rHet_pco2 = y_rHet_cdoc - 1.0  # production coefficient of CO2
        y_rHet_pnh4 = (Nr*y_rHet_cdoc - Nb)  # production coefficient of NH4
        ### complete oxidation of short-chain, labile DOM by scavenging bacteria
        y_sHet_pB = sHet_fele * e_sdom/e_bac
        if y_sHet_pB > 0.0:
            y_sHet_cdoc = 1.0/y_sHet_pB  # consumption coefficient of sDOC
        y_sHet_coxy = np.fmax(0.0, e_sdom - y_sHet_pB * e_bac)/4.0 * y_sHet_cdoc # consumption coefficient of O2
        y_sHet_pco2 = y_sHet_cdoc - 1.0  # production coefficient of CO2
        y_sHet_pnh4 = (Ns*y_sHet_cdoc - Nb)  # production coefficient of NH4

        ### Respiratory quotient (mol O2 consumed per mol CO2 produced)
        lHet_RQ = y_lHet_pco2 / y_lHet_coxy
        rHet_RQ = y_rHet_pco2 / y_rHet_coxy
        sHet_RQ = y_sHet_pco2 / y_sHet_coxy

        
        ### ---------------- ###
        ### growth rates (u) ###
        ### ---------------- ###

        # Ammonia oxidizing archaea (AOA)
        aoa_g_O2 = u_aoa_O2 * aoa_y_O2
        aoa_h_N = u_aoa_NH4 * aoa_y_NH4
        mu_aoa = np.fmin(aoa_g_O2, aoa_h_N)
        # Long-chain consuming, Sharing heterotrophs (lHet)
        lHet_g_O2 = u_lHet_O2 / y_lHet_coxy
        lHet_g_C = u_lHet_ldoc / y_lHet_cdoc
        gmin = np.fmin(lHet_g_C, lHet_g_O2)
        if y_lHet_pnh4 < 0.0: 
            lHet_g_N = u_lHet_NH4 / np.fmax(epsi, -y_lHet_pnh4)
            gmin = np.fmin(gmin, lHet_g_N)
        else:
            lHet_g_N = np.nan
        mu_lHet = np.fmax(0.0, gmin)
        # Long-chain consuming, reworking selfish heterotrophs (rHet)
        rHet_g_O2 = u_rHet_O2 / y_rHet_coxy
        rHet_g_C = u_rHet_rdoc / y_rHet_cdoc
        gmin = np.fmin(rHet_g_C, rHet_g_O2)
        if y_rHet_pnh4 < 0.0: 
            rHet_g_N = u_rHet_NH4 / np.fmax(epsi, -y_rHet_pnh4)
            gmin = np.fmin(gmin, rHet_g_N)
        else:
            rHet_g_N = np.nan
        mu_rHet = np.fmax(0.0, gmin)
        # Short-chain consuming, scavenging heterotrophs (sHet)
        sHet_g_O2 = u_sHet_O2 / y_sHet_coxy
        sHet_g_C = u_sHet_sdoc / y_sHet_cdoc
        gmin = np.fmin(sHet_g_C, sHet_g_O2)
        if y_sHet_pnh4 < 0.0: 
            sHet_g_N = u_sHet_NH4 / np.fmax(epsi, -y_sHet_pnh4)
            gmin = np.fmin(gmin, sHet_g_N)
        else:
            sHet_g_N = np.nan
        mu_sHet = np.fmax(0.0, gmin)
                
        ### rates of substrate consumption
        aoa_oxy_cons = mu_aoa * m_baoa / aoa_y_O2
        aoa_nh4_cons = mu_aoa * m_baoa / aoa_y_NH4
        aoa_co2_cons = mu_aoa * m_baoa
        lHet_oxy_cons = mu_lHet * m_blHet * y_lHet_coxy
        lHet_ldoc_cons = mu_lHet * m_blHet * y_lHet_cdoc
        lHet_ldon_cons = mu_lHet * m_blHet * y_lHet_cdoc * Nl
        rHet_oxy_cons = mu_rHet * m_brHet * y_rHet_coxy
        rHet_rdoc_cons = mu_rHet * m_brHet * y_rHet_cdoc
        rHet_rdon_cons = mu_rHet * m_brHet * y_rHet_cdoc * Nr
        sHet_oxy_cons = mu_sHet * m_bsHet * y_sHet_coxy
        sHet_sdoc_cons = mu_sHet * m_bsHet * y_sHet_cdoc
        sHet_sdon_cons = mu_sHet * m_bsHet * y_sHet_cdoc * Ns
        # rates of substrate production
        lHet_sdoc_prod = mu_lHet * m_blHet * y_lHet_psdoc
        lHet_sdon_prod = mu_lHet * m_blHet * y_lHet_psdon
        lHet_rdoc_prod = mu_lHet * m_blHet * y_lHet_prdoc
        lHet_rdon_prod = mu_lHet * m_blHet * y_lHet_prdon
        lHet_co2_prod  = mu_lHet * m_blHet * y_lHet_pco2
        lHet_nh4_prod  = mu_lHet * m_blHet * y_lHet_pnh4
        rHet_co2_prod  = mu_rHet * m_brHet * y_rHet_pco2
        rHet_nh4_prod  = mu_rHet * m_brHet * y_rHet_pnh4
        sHet_co2_prod  = mu_sHet * m_bsHet * y_sHet_pco2
        sHet_nh4_prod  = mu_sHet * m_bsHet * y_sHet_pnh4
                
        # Dissolved organic matter (if flow-through, then dilution is applied)
        ddt_ldoc = dil * (in_ldoc - m_ldoc*xx) - lHet_ldoc_cons
        ddt_ldon = dil * (in_ldon - m_ldon*xx) - lHet_ldon_cons
        ddt_rdoc = dil * (in_rdoc - m_rdoc*xx) + lHet_rdoc_prod - rHet_rdoc_cons
        ddt_rdon = dil * (in_rdon - m_rdon*xx) + lHet_rdon_prod - rHet_rdon_cons
        ddt_sdoc = dil * (in_sdoc - m_sdoc*xx) + lHet_sdoc_prod - sHet_sdoc_cons
        ddt_sdon = dil * (in_sdon - m_sdon*xx) + lHet_sdon_prod - sHet_sdon_cons
        # Dissolved oxygen
        ddt_O2 = dil * (in_O2 - m_O2) - lHet_oxy_cons - rHet_oxy_cons - sHet_oxy_cons - aoa_oxy_cons
        # Ammonium 
        ddt_NH4 = dil * (in_NH4 - m_NH4) + lHet_nh4_prod + rHet_nh4_prod + sHet_nh4_prod - aoa_nh4_cons
        # CO2 (new DIC)
        ddt_CO2 = dil * (in_CO2 - m_CO2) + lHet_co2_prod + rHet_co2_prod + sHet_co2_prod - aoa_co2_cons
        # Biomass of aerobic heterotrophs
        ddt_baoa = dil * (-m_baoa) + mu_aoa * m_baoa
        ddt_blHet = dil * (-m_blHet) + mu_lHet * m_blHet
        ddt_brHet = dil * (-m_brHet) + mu_rHet * m_brHet
        ddt_bsHet = dil * (-m_bsHet) + mu_sHet * m_bsHet
                
        
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        ### apply changes to state variables ###
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        
        m_ldoc = m_ldoc + ddt_ldoc * dt
        m_ldon = m_ldon + ddt_ldon * dt
        m_rdoc = m_rdoc + ddt_rdoc * dt
        m_rdon = m_rdon + ddt_rdon * dt
        m_sdoc = m_sdoc + ddt_sdoc * dt
        m_sdon = m_sdon + ddt_sdon * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_CO2 = m_CO2 + ddt_CO2 * dt
        m_baoa = m_baoa + ddt_baoa * dt
        m_blHet = m_blHet + ddt_blHet * dt
        m_brHet = m_brHet + ddt_brHet * dt
        m_bsHet = m_bsHet + ddt_bsHet * dt

        # Record tracer concentrations at each timestep
        dm_O2[int(t % interval)] = m_O2
        dm_NH4[int(t % interval)] = m_NH4
        dm_CO2[int(t % interval)] = m_CO2
        dm_ldoc[int(t % interval)] = m_ldoc
        dm_rdoc[int(t % interval)] = m_rdoc
        dm_sdoc[int(t % interval)] = m_sdoc
        dm_ldon[int(t % interval)] = m_ldon
        dm_rdon[int(t % interval)] = m_rdon
        dm_sdon[int(t % interval)] = m_sdon
        
        # Record growth rates at each timestep
        du_aoa[int(t % interval)] = mu_aoa
        du_lHet[int(t % interval)] = mu_lHet
        du_lHet_C[int(t % interval)] = lHet_g_C
        du_lHet_N[int(t % interval)] = lHet_g_N
        du_rHet[int(t % interval)] = mu_rHet
        du_rHet_C[int(t % interval)] = rHet_g_C
        du_rHet_N[int(t % interval)] = rHet_g_N
        du_sHet[int(t % interval)] = mu_sHet
        du_sHet_C[int(t % interval)] = sHet_g_C
        du_sHet_N[int(t % interval)] = sHet_g_N
        
        # Record biomass at each timestep
        db_aoa[int(t % interval)] = m_baoa
        db_lHet[int(t % interval)] = m_blHet
        db_rHet[int(t % interval)] = m_brHet
        db_sHet[int(t % interval)] = m_bsHet
        
        # Record biomass yields at each timestep
        dy_lHet[int(t % interval)] = y_lHet_pB
        dy_rHet[int(t % interval)] = y_rHet_pB
        dy_sHet[int(t % interval)] = y_sHet_pB
        
        # Record rates at each timestep
        dc_lHet_ldoc[int(t % interval)] = lHet_ldoc_cons
        dc_lHet_ldon[int(t % interval)] = lHet_ldon_cons
        dc_rHet_rdoc[int(t % interval)] = rHet_rdoc_cons
        dc_rHet_rdon[int(t % interval)] = rHet_rdon_cons
        dc_sHet_sdoc[int(t % interval)] = sHet_sdoc_cons
        dc_sHet_sdon[int(t % interval)] = sHet_sdon_cons
        dc_lHet_O2[int(t % interval)] = lHet_oxy_cons
        dc_rHet_O2[int(t % interval)] = rHet_oxy_cons
        dc_sHet_O2[int(t % interval)] = sHet_oxy_cons
        dp_lHet_CO2[int(t % interval)] = lHet_co2_prod
        dp_rHet_CO2[int(t % interval)] = rHet_co2_prod
        dp_sHet_CO2[int(t % interval)] = sHet_co2_prod
        dp_lHet_NH4[int(t % interval)] = lHet_nh4_prod
        dp_rHet_NH4[int(t % interval)] = rHet_nh4_prod
        dp_sHet_NH4[int(t % interval)] = sHet_nh4_prod
        dp_lHet_sdoc[int(t % interval)] = lHet_sdoc_prod
        dp_lHet_rdoc[int(t % interval)] = lHet_rdoc_prod
        dp_lHet_sdon[int(t % interval)] = lHet_sdon_prod
        dp_lHet_rdon[int(t % interval)] = lHet_rdon_prod
        d_lHet_RQ[int(t % interval)] = lHet_RQ
        d_rHet_RQ[int(t % interval)] = rHet_RQ
        d_sHet_RQ[int(t % interval)] = sHet_RQ

        # pulse OM and O2 into the chemostat
        pulse_interval = int((1.0 / dt) * pulse_int)
        if pulse_interval < 1:
            pulse_interval = 1
        if (t % pulse_interval) == 0:
            m_ldoc = m_ldoc + pulse_ldoc
            m_ldon = m_ldon + pulse_ldon
            m_rdoc = m_rdoc + pulse_rdoc
            m_rdon = m_rdon + pulse_rdon
            m_sdoc = m_sdoc + pulse_sdoc
            m_sdon = m_sdon + pulse_sdon
            m_NH4 = m_NH4 + pulse_NH4
            m_baoa = m_baoa + pulse_baoa
            m_blHet = m_blHet + pulse_blHet
            m_brHet = m_brHet + pulse_brHet
            m_bsHet = m_bsHet + pulse_bsHet
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_ldoc[i] = m_ldoc
            out_ldon[i] = m_ldon
            out_rdoc[i] = m_rdoc
            out_rdon[i] = m_rdon
            out_sdoc[i] = m_sdoc
            out_sdon[i] = m_sdon
            out_O2[i] = m_O2
            out_NH4[i] = m_NH4 
            out_CO2[i] = m_CO2
            out_baoa[i] = m_baoa
            out_blHet[i] = m_blHet
            out_brHet[i] = m_brHet
            out_bsHet[i] = m_bsHet
            out_uaoa[i] = mu_aoa
            out_ulHet[i] = mu_lHet
            out_ulHet_C[i] = lHet_g_C
            out_ulHet_N[i] = lHet_g_N
            out_urHet[i] = mu_rHet
            out_urHet_C[i] = rHet_g_C
            out_urHet_N[i] = rHet_g_N
            out_usHet[i] = mu_sHet
            out_usHet_C[i] = sHet_g_C
            out_usHet_N[i] = sHet_g_N
            out_ylHet[i] = y_lHet_pB
            out_yrHet[i] = y_rHet_pB
            out_ysHet[i] = y_sHet_pB
            out_c_lHet_ldoc[i] = lHet_ldoc_cons
            out_c_lHet_ldon[i] = lHet_ldon_cons
            out_c_lHet_O2[i] = lHet_oxy_cons
            out_p_lHet_CO2[i] = lHet_co2_prod
            out_p_lHet_sdoc[i] = lHet_sdoc_prod
            out_p_lHet_sdon[i] = lHet_sdon_prod
            out_p_lHet_rdoc[i] = lHet_rdoc_prod
            out_p_lHet_rdon[i] = lHet_rdon_prod
            out_p_lHet_NH4[i] = lHet_nh4_prod
            out_lHet_RQ[i] = lHet_RQ
            out_c_rHet_rdoc[i] = rHet_rdoc_cons
            out_c_rHet_rdon[i] = rHet_rdon_cons
            out_c_rHet_O2[i] = rHet_oxy_cons
            out_p_rHet_CO2[i] = rHet_co2_prod
            out_p_rHet_NH4[i] = rHet_nh4_prod
            out_rHet_RQ[i] = rHet_RQ
            out_c_sHet_sdoc[i] = sHet_sdoc_cons
            out_c_sHet_sdon[i] = sHet_sdon_cons
            out_c_sHet_O2[i] = sHet_oxy_cons
            out_p_sHet_CO2[i] = sHet_co2_prod
            out_p_sHet_NH4[i] = sHet_nh4_prod
            out_sHet_RQ[i] = sHet_RQ

                  
    return [out_ldoc, out_ldon, out_rdoc, out_rdon, out_sdoc, out_sdon, out_O2, out_CO2, out_NH4, \
            out_baoa, out_blHet, out_brHet, out_bsHet, \
            out_uaoa, \
            out_ulHet, out_ulHet_C, out_ulHet_N, \
            out_urHet, out_urHet_C, out_urHet_N, \
            out_usHet, out_usHet_C, out_usHet_N, \
            out_ylHet, out_yrHet, out_ysHet, \
            out_c_lHet_ldoc, out_c_lHet_ldon, out_c_lHet_O2, \
            out_p_lHet_CO2, out_p_lHet_sdoc, out_p_lHet_sdon, out_p_lHet_rdoc, out_p_lHet_rdon, out_p_lHet_NH4, out_lHet_RQ, \
            out_c_rHet_rdoc, out_c_rHet_rdon, out_c_rHet_O2, out_p_rHet_CO2, out_p_rHet_NH4, out_rHet_RQ, \
            out_c_sHet_sdoc, out_c_sHet_sdon, out_c_sHet_O2, out_p_sHet_CO2, out_p_sHet_NH4, out_sHet_RQ, \
            dm_O2, dm_CO2, dm_NH4, dm_ldoc, dm_ldon, dm_rdoc, dm_rdon, dm_sdoc, dm_sdon, \
            du_aoa, du_lHet, du_lHet_C, du_lHet_N, du_rHet, du_rHet_C, du_rHet_N, du_sHet, du_sHet_C, du_sHet_N, \
            db_aoa, db_lHet, db_rHet, db_sHet, dy_lHet, dy_rHet, dy_sHet, \
            dc_lHet_ldoc, dc_lHet_ldon, dc_lHet_O2, dp_lHet_CO2, dp_lHet_sdoc, dp_lHet_sdon, dp_lHet_rdoc, dp_lHet_rdon, dp_lHet_NH4, d_lHet_RQ, \
            dc_rHet_rdoc, dc_rHet_rdon, dc_rHet_O2, dp_rHet_CO2, dp_rHet_NH4, d_rHet_RQ, \
            dc_sHet_sdoc, dc_sHet_sdon, dc_sHet_O2, dp_sHet_CO2, dp_sHet_NH4, d_sHet_RQ \
            ]



@jit(nopython=True)
def PhyBacDOM(timesteps, nn_output, dt, dil, out_at_day, \
             pulse_nh4, pulse_doc, pulse_phy, pulse_het, pulse_int, \
             phy_mumax, phy_K_nh4, \
             het_Vmax_doc, het_K_doc, \ 
             het_fele, \
             Hs, Os, Ns, Hb, Ob, Nb, \
             in_doc, in_nh4, \
             in_phy, in_het, \
             flow_through=True
             ):
    '''
    ______________________________________________________________________
    0D chemostat model for algae and bacteria
    ----------------------------------------------------------------------

    This model simulates an algae and a heterotrophic bacterial community
    consuming nutrient and organic matter substrate, respectively.

    Algae convert nutrient into biomass and a small fraction of exuded organic
    substrate. Algae die according to a mortality rate, producing organic 
    matter as well. Bacteria consume the organic matter, remineralising the 
    nutrient.

    INPUTS
    ------

    timesteps : int
        Number of model timesteps.

    nn_output : int
        Number of saved output intervals. Output arrays have length nn_output + 1.

    dt : float
        Timestep length in days.

    dil : float
        Chemostat dilution rate in day-1.

    out_at_day : float
        Output saving interval in days.

    pulse_doc : float
        Amount of DOC added during each pulse.

    pulse_nh4 : float
        Amount of NH4 added during each pulse.

    pulse_het : float
        Biomass of heterotroph added during each pulse.

    pulse_phy : float
        Biomass of algae added during each pulse.
    
    pulse_int : float
        Pulse interval in days.

    phy_mumax : float
        Maximum growth rate of algae.
    
    phy_K_nh4 : float
        Half-saturation coefficient for NH4 uptake.
    
    het_Vmax_doc : float
        Maximum DOC uptake rate per unit bacterial biomass.

    het_K_doc : float
        Half-saturation coefficient for DOC uptake.

    het_fele : float
        Fraction of released electron equivalents routed to biosynthesis
        
    Hb : float
        Bacterial biomass H:C ratio.

    Ob : float
        Bacterial biomass O:C ratio.

    Nb : float
        Bacterial biomass N:C ratio.

    Hs : float
        Substrate H:C ratio.

    Os : float
        Substrate O:C ratio.

    Ns : float
        Substrate N:C ratio.

    in_doc : float
        Initial DOC concentration.

    in_nh4 : float
        Initial NH4 concentration.

    in_het : float
        Initial biomass concentration of the heterotroph.

    in_phy : float
        Initial biomass concentration of the algae.
    

    OUTPUTS
    -------

    out_ldoc, out_sdoc : array
        Saved DOC tracer concentrations.

    out_O2, out_CO2, out_NH4 : array
        Saved inorganic tracer concentrations.

    out_blHet, out_bsHet : array
        Saved bacterial biomass concentration.

    out_ulHet, out_usHet : array
        Saved specific growth rate of heterotroph.

    out_ylHet, out_ysHet : array
        Saved biomass yield, in mol C biomass per mol DOC consumed.

    out_c_lHet_ldoc, out_c_sHet_sdoc : array
        Saved DOC consumption rate for heterotroph.

    out_c_lHet_O2, out_c_sHet_O2 : array
        Saved O2 consumption rate for heterotroph.

    out_p_lHet_CO2, out_p_sHet_CO2 : array
        Saved CO2 production rates for heterotroph.

    out_p_lHet_sdoc : array
            Saved net sDOC production rates for heterotroph.
            
    out_p_lHet_NH4, out_p_sHet_NH4 : array
        Saved net NH4 production rates for heterotroph.
        Negative values indicate NH4 uptake.

    out_lHet_RQ, out_sHet_RQ : array
        Saved respiratory quotients for heterotroph types 1, 2 and 3,
        defined here as mol O2 consumed per mol CO2 produced.

    dm_* arrays : array
        High-frequency tracer records over the final output interval.

    du_lHet*, du_sHet* arrays : array
        High-frequency growth-rate records over the final output interval.

    db_lHet*, db_sHet* arrays : array
        High-frequency biomass records over the final output interval.

    dy_lHet*, dy_sHet* arrays : array
        High-frequency biomass-yield records over the final output interval.

    df*, dc*, dp*, d*RQ arrays : array
        High-frequency DOC availability, rates of consumption and production
        terms, and respiratory quotient records over the final output interval.
    '''
    
    xx = 0.0
    if flow_through:
        xx = 1.0
    
    epsi = 1e-20
    
    # transfer initial inputs to model variables
    m_ldoc = in_ldoc
    m_ldon = in_ldon
    m_rdoc = in_rdoc
    m_rdon = in_rdon
    m_sdoc = in_sdoc
    m_sdon = in_sdon
    m_O2 = in_O2
    m_CO2 = in_CO2
    m_NH4 = in_NH4
    m_baoa = in_baoa
    m_blHet = in_blHet
    m_brHet = in_brHet
    m_bsHet = in_bsHet
        
    # set the output arrays 
    out_ldoc = np.ones((int(nn_output)+1)) * np.nan
    out_ldon = np.ones((int(nn_output)+1)) * np.nan
    out_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_baoa = np.ones((int(nn_output)+1)) * np.nan
    out_uaoa = np.ones((int(nn_output)+1)) * np.nan
    out_blHet = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_ulHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_ylHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_ldoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_ldon = np.ones((int(nn_output)+1)) * np.nan
    out_c_lHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_lHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_lHet_RQ = np.ones((int(nn_output)+1)) * np.nan
    out_brHet = np.ones((int(nn_output)+1)) * np.nan
    out_urHet = np.ones((int(nn_output)+1)) * np.nan
    out_urHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_urHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_yrHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_rdoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_rdon = np.ones((int(nn_output)+1)) * np.nan
    out_c_rHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_rHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_rHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_rHet_RQ = np.ones((int(nn_output)+1)) * np.nan
    out_bsHet = np.ones((int(nn_output)+1)) * np.nan
    out_usHet = np.ones((int(nn_output)+1)) * np.nan
    out_usHet_C = np.ones((int(nn_output)+1)) * np.nan
    out_usHet_N = np.ones((int(nn_output)+1)) * np.nan
    out_ysHet = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_sdoc = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_sdon = np.ones((int(nn_output)+1)) * np.nan
    out_c_sHet_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_sHet_CO2 = np.ones((int(nn_output)+1)) * np.nan
    out_p_sHet_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_sHet_RQ = np.ones((int(nn_output)+1)) * np.nan
        
    
    # set the array for recording average aerobic or anaerobic activity of facultative population
    interval = int((1.0 / dt) * out_at_day)
    if interval < 1:
        interval = 1
    
    # set the arrays for activities over the final interval of the experiment
    dm_O2 = np.ones((interval)) * np.nan
    dm_NH4 = np.ones((interval)) * np.nan
    dm_CO2 = np.ones((interval)) * np.nan
    dm_ldoc = np.ones((interval)) * np.nan
    dm_ldon = np.ones((interval)) * np.nan
    dm_rdoc = np.ones((interval)) * np.nan
    dm_rdon = np.ones((interval)) * np.nan
    dm_sdoc = np.ones((interval)) * np.nan
    dm_sdon = np.ones((interval)) * np.nan
    db_aoa = np.ones((interval)) * np.nan
    du_aoa = np.ones((interval)) * np.nan
    du_lHet = np.ones((interval)) * np.nan
    du_lHet_C = np.ones((interval)) * np.nan
    du_lHet_N = np.ones((interval)) * np.nan
    db_lHet = np.ones((interval)) * np.nan
    dy_lHet = np.ones((interval)) * np.nan
    dc_lHet_ldoc = np.ones((interval)) * np.nan
    dc_lHet_ldon = np.ones((interval)) * np.nan
    dc_lHet_O2 = np.ones((interval)) * np.nan
    dp_lHet_sdoc = np.ones((interval)) * np.nan
    dp_lHet_sdon = np.ones((interval)) * np.nan
    dp_lHet_rdoc = np.ones((interval)) * np.nan
    dp_lHet_rdon = np.ones((interval)) * np.nan
    dp_lHet_CO2 = np.ones((interval)) * np.nan
    dp_lHet_NH4 = np.ones((interval)) * np.nan
    d_lHet_RQ = np.ones((interval)) * np.nan
    du_rHet = np.ones((interval)) * np.nan
    du_rHet_C = np.ones((interval)) * np.nan
    du_rHet_N = np.ones((interval)) * np.nan
    db_rHet = np.ones((interval)) * np.nan
    dy_rHet = np.ones((interval)) * np.nan
    dc_rHet_rdoc = np.ones((interval)) * np.nan
    dc_rHet_rdon = np.ones((interval)) * np.nan
    dc_rHet_O2 = np.ones((interval)) * np.nan
    dp_rHet_CO2 = np.ones((interval)) * np.nan
    dp_rHet_NH4 = np.ones((interval)) * np.nan
    d_rHet_RQ = np.ones((interval)) * np.nan
    du_sHet = np.ones((interval)) * np.nan
    du_sHet_C = np.ones((interval)) * np.nan
    du_sHet_N = np.ones((interval)) * np.nan
    db_sHet = np.ones((interval)) * np.nan
    dy_sHet = np.ones((interval)) * np.nan
    dc_sHet_sdoc = np.ones((interval)) * np.nan
    dc_sHet_sdon = np.ones((interval)) * np.nan
    dc_sHet_O2 = np.ones((interval)) * np.nan
    dp_sHet_CO2 = np.ones((interval)) * np.nan
    dp_sHet_NH4 = np.ones((interval)) * np.nan
    d_sHet_RQ = np.ones((interval)) * np.nan
        
    
    # record the initial conditions
    i = 0
    out_ldoc[i] = m_ldoc
    out_ldon[i] = m_ldon
    out_rdoc[i] = m_rdoc
    out_rdon[i] = m_rdon
    out_sdoc[i] = m_sdoc
    out_sdon[i] = m_sdon
    out_O2[i] = m_O2
    out_NH4[i] = m_NH4 
    out_CO2[i] = m_CO2
    out_baoa[i] = m_baoa
    out_blHet[i] = m_blHet
    out_brHet[i] = m_brHet
    out_bsHet[i] = m_bsHet
        
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        # uptake rates (p)
        u_aoa_O2    = aoa_p_O2 * m_O2                                        # mol O2 / day
        u_aoa_NH4   = aoa_Vmax_NH4 * m_NH4 / (aoa_K_NH4 + m_NH4 + epsi)      # mol N / day
        u_lHet_O2   = lHet_p_O2 * m_O2                                       # mol O2 / day
        u_lHet_ldoc = lHet_Vmax_doc * m_ldoc / (lHet_K_ldoc + m_ldoc + epsi)  # mol C-Org / day
        u_lHet_NH4  = lHet_Vmax_NH4 * m_NH4 / (lHet_K_NH4 + m_NH4 + epsi)    # mol N / day
        u_rHet_O2   = rHet_p_O2 * m_O2                                       # mol O2 / day
        u_rHet_rdoc = rHet_Vmax_doc * m_rdoc / (rHet_K_rdoc + m_rdoc + epsi)  # mol C-Org / day
        u_rHet_NH4  = rHet_Vmax_NH4 * m_NH4 / (rHet_K_NH4 + m_NH4 + epsi)    # mol N / day
        u_sHet_O2   = sHet_p_O2 * m_O2                                       # mol O2 / day
        u_sHet_sdoc = sHet_Vmax_doc * m_sdoc / (sHet_K_sdoc + m_sdoc + epsi)  # mol C-Org / day
        u_sHet_NH4  = sHet_Vmax_NH4 * m_NH4 / (sHet_K_NH4 + m_NH4 + epsi)    # mol N / day

        ### ------------------------------------------------------------------- ###
        ### electrons per atom of carbon in biomass (b) and substrate (l, r, s) ###
        ### ------------------------------------------------------------------- ###
        def ele_potential(c,h,o,n):
            return 4*c + 1*h - 2*o - 3*n
        Nl = m_ldon / np.fmax(epsi, m_ldoc) # dynamic ratio of N:C in lDOM
        Nr = m_rdon / np.fmax(epsi, m_rdoc) # dynamic ratio of N:C in rDOM
        Ns = m_sdon / np.fmax(epsi, m_sdoc) # dynamic ratio of N:C in sDOM
        e_bac = ele_potential(1.0, Hb, Ob, Nb)
        e_ldom = ele_potential(1.0, Hl, Ol, Nl)
        e_rdom = ele_potential(1.0, Hr, Or, Nr)
        e_sdom = ele_potential(1.0, Hs, Os, Ns)
        # Electron potential of the partially oxidized product (sDOM) from lDOM and rDOM
        e_sdom_l = ele_potential(1.0, Hs, Os, Nl * donpersist)
        e_rdom_l = ele_potential(1.0, Hr, Or, Nl * donpersist)
        ### residual electrons available to sharing bacteria after some loss of product
        e_lres = np.fmax(0.0, e_ldom 
                              - lHet_alpha * lHet_beta * e_sdom_l 
                              - lHet_alpha * (1.0 - lHet_beta) * e_rdom_l)


        ### ------------------------------------ ###
        ### compute resource yields and products ###
        ### ------------------------------------ ###
          
        # initalize some values
        y_lHet_cdoc = 0.0; y_rHet_cdoc = 0.0; y_sHet_cdoc = 0.0 # reinitialise for safety
      
        ### Sharing bacteria (do partial oxidation, so produce sDOM and rDOM and CO2)
        y_lHet_pB = np.fmax(0.0, np.fmin(lHet_fele * e_lres/e_bac, 1.0 - lHet_alpha)) # Cap C yields to a theoretical maximum
        if y_lHet_pB > 0.0:
            y_lHet_cdoc = 1.0/y_lHet_pB  # consumption coefficient of lDOC
        y_lHet_coxy = np.fmax(0.0, e_lres - y_lHet_pB * e_bac)/4.0 * y_lHet_cdoc # consumption coefficient of O2
        y_lHet_psdoc = lHet_alpha *lHet_beta * y_lHet_cdoc  # production coefficient of sDOM
        y_lHet_prdoc = lHet_alpha * (1.0 - lHet_beta) * y_lHet_cdoc  # production coefficient of rDOM
        y_lHet_pco2 = (1.0 - lHet_alpha) * y_lHet_cdoc - 1.0
        y_lHet_pnh4 = (Nl - lHet_alpha*Nl*donpersist) * y_lHet_cdoc - Nb  # production coefficient of NH4
        y_lHet_psdon = (lHet_alpha*lHet_beta*Nl*donpersist) * y_lHet_cdoc  # production coefficient of sDON
        y_lHet_prdon = (lHet_alpha*(1.0-lHet_beta)*Nl*donpersist) * y_lHet_cdoc  # production coefficient of rDON
        ### complete oxidation of long-chain, reworked DOM by selfish bacteria
        y_rHet_pB = rHet_fele * e_rdom/e_bac
        if y_rHet_pB > 0.0:
            y_rHet_cdoc = 1.0/y_rHet_pB  # consumption coefficient of rDOC
        y_rHet_coxy = np.fmax(0.0, e_rdom - y_rHet_pB * e_bac)/4.0 * y_rHet_cdoc # consumption coefficient of O2
        y_rHet_pco2 = y_rHet_cdoc - 1.0  # production coefficient of CO2
        y_rHet_pnh4 = (Nr*y_rHet_cdoc - Nb)  # production coefficient of NH4
        ### complete oxidation of short-chain, labile DOM by scavenging bacteria
        y_sHet_pB = sHet_fele * e_sdom/e_bac
        if y_sHet_pB > 0.0:
            y_sHet_cdoc = 1.0/y_sHet_pB  # consumption coefficient of sDOC
        y_sHet_coxy = np.fmax(0.0, e_sdom - y_sHet_pB * e_bac)/4.0 * y_sHet_cdoc # consumption coefficient of O2
        y_sHet_pco2 = y_sHet_cdoc - 1.0  # production coefficient of CO2
        y_sHet_pnh4 = (Ns*y_sHet_cdoc - Nb)  # production coefficient of NH4

        ### Respiratory quotient (mol O2 consumed per mol CO2 produced)
        lHet_RQ = y_lHet_pco2 / y_lHet_coxy
        rHet_RQ = y_rHet_pco2 / y_rHet_coxy
        sHet_RQ = y_sHet_pco2 / y_sHet_coxy

        
        ### ---------------- ###
        ### growth rates (u) ###
        ### ---------------- ###

        # Ammonia oxidizing archaea (AOA)
        aoa_g_O2 = u_aoa_O2 * aoa_y_O2
        aoa_h_N = u_aoa_NH4 * aoa_y_NH4
        mu_aoa = np.fmin(aoa_g_O2, aoa_h_N)
        # Long-chain consuming, Sharing heterotrophs (lHet)
        lHet_g_O2 = u_lHet_O2 / y_lHet_coxy
        lHet_g_C = u_lHet_ldoc / y_lHet_cdoc
        gmin = np.fmin(lHet_g_C, lHet_g_O2)
        if y_lHet_pnh4 < 0.0: 
            lHet_g_N = u_lHet_NH4 / np.fmax(epsi, -y_lHet_pnh4)
            gmin = np.fmin(gmin, lHet_g_N)
        else:
            lHet_g_N = np.nan
        mu_lHet = np.fmax(0.0, gmin)
        # Long-chain consuming, reworking selfish heterotrophs (rHet)
        rHet_g_O2 = u_rHet_O2 / y_rHet_coxy
        rHet_g_C = u_rHet_rdoc / y_rHet_cdoc
        gmin = np.fmin(rHet_g_C, rHet_g_O2)
        if y_rHet_pnh4 < 0.0: 
            rHet_g_N = u_rHet_NH4 / np.fmax(epsi, -y_rHet_pnh4)
            gmin = np.fmin(gmin, rHet_g_N)
        else:
            rHet_g_N = np.nan
        mu_rHet = np.fmax(0.0, gmin)
        # Short-chain consuming, scavenging heterotrophs (sHet)
        sHet_g_O2 = u_sHet_O2 / y_sHet_coxy
        sHet_g_C = u_sHet_sdoc / y_sHet_cdoc
        gmin = np.fmin(sHet_g_C, sHet_g_O2)
        if y_sHet_pnh4 < 0.0: 
            sHet_g_N = u_sHet_NH4 / np.fmax(epsi, -y_sHet_pnh4)
            gmin = np.fmin(gmin, sHet_g_N)
        else:
            sHet_g_N = np.nan
        mu_sHet = np.fmax(0.0, gmin)
                
        ### rates of substrate consumption
        aoa_oxy_cons = mu_aoa * m_baoa / aoa_y_O2
        aoa_nh4_cons = mu_aoa * m_baoa / aoa_y_NH4
        aoa_co2_cons = mu_aoa * m_baoa
        lHet_oxy_cons = mu_lHet * m_blHet * y_lHet_coxy
        lHet_ldoc_cons = mu_lHet * m_blHet * y_lHet_cdoc
        lHet_ldon_cons = mu_lHet * m_blHet * y_lHet_cdoc * Nl
        rHet_oxy_cons = mu_rHet * m_brHet * y_rHet_coxy
        rHet_rdoc_cons = mu_rHet * m_brHet * y_rHet_cdoc
        rHet_rdon_cons = mu_rHet * m_brHet * y_rHet_cdoc * Nr
        sHet_oxy_cons = mu_sHet * m_bsHet * y_sHet_coxy
        sHet_sdoc_cons = mu_sHet * m_bsHet * y_sHet_cdoc
        sHet_sdon_cons = mu_sHet * m_bsHet * y_sHet_cdoc * Ns
        # rates of substrate production
        lHet_sdoc_prod = mu_lHet * m_blHet * y_lHet_psdoc
        lHet_sdon_prod = mu_lHet * m_blHet * y_lHet_psdon
        lHet_rdoc_prod = mu_lHet * m_blHet * y_lHet_prdoc
        lHet_rdon_prod = mu_lHet * m_blHet * y_lHet_prdon
        lHet_co2_prod  = mu_lHet * m_blHet * y_lHet_pco2
        lHet_nh4_prod  = mu_lHet * m_blHet * y_lHet_pnh4
        rHet_co2_prod  = mu_rHet * m_brHet * y_rHet_pco2
        rHet_nh4_prod  = mu_rHet * m_brHet * y_rHet_pnh4
        sHet_co2_prod  = mu_sHet * m_bsHet * y_sHet_pco2
        sHet_nh4_prod  = mu_sHet * m_bsHet * y_sHet_pnh4
                
        # Dissolved organic matter (if flow-through, then dilution is applied)
        ddt_ldoc = dil * (in_ldoc - m_ldoc*xx) - lHet_ldoc_cons
        ddt_ldon = dil * (in_ldon - m_ldon*xx) - lHet_ldon_cons
        ddt_rdoc = dil * (in_rdoc - m_rdoc*xx) + lHet_rdoc_prod - rHet_rdoc_cons
        ddt_rdon = dil * (in_rdon - m_rdon*xx) + lHet_rdon_prod - rHet_rdon_cons
        ddt_sdoc = dil * (in_sdoc - m_sdoc*xx) + lHet_sdoc_prod - sHet_sdoc_cons
        ddt_sdon = dil * (in_sdon - m_sdon*xx) + lHet_sdon_prod - sHet_sdon_cons
        # Dissolved oxygen
        ddt_O2 = dil * (in_O2 - m_O2) - lHet_oxy_cons - rHet_oxy_cons - sHet_oxy_cons - aoa_oxy_cons
        # Ammonium 
        ddt_NH4 = dil * (in_NH4 - m_NH4) + lHet_nh4_prod + rHet_nh4_prod + sHet_nh4_prod - aoa_nh4_cons
        # CO2 (new DIC)
        ddt_CO2 = dil * (in_CO2 - m_CO2) + lHet_co2_prod + rHet_co2_prod + sHet_co2_prod - aoa_co2_cons
        # Biomass of aerobic heterotrophs
        ddt_baoa = dil * (-m_baoa) + mu_aoa * m_baoa
        ddt_blHet = dil * (-m_blHet) + mu_lHet * m_blHet
        ddt_brHet = dil * (-m_brHet) + mu_rHet * m_brHet
        ddt_bsHet = dil * (-m_bsHet) + mu_sHet * m_bsHet
                
        
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        ### apply changes to state variables ###
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        
        m_ldoc = m_ldoc + ddt_ldoc * dt
        m_ldon = m_ldon + ddt_ldon * dt
        m_rdoc = m_rdoc + ddt_rdoc * dt
        m_rdon = m_rdon + ddt_rdon * dt
        m_sdoc = m_sdoc + ddt_sdoc * dt
        m_sdon = m_sdon + ddt_sdon * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_CO2 = m_CO2 + ddt_CO2 * dt
        m_baoa = m_baoa + ddt_baoa * dt
        m_blHet = m_blHet + ddt_blHet * dt
        m_brHet = m_brHet + ddt_brHet * dt
        m_bsHet = m_bsHet + ddt_bsHet * dt

        # Record tracer concentrations at each timestep
        dm_O2[int(t % interval)] = m_O2
        dm_NH4[int(t % interval)] = m_NH4
        dm_CO2[int(t % interval)] = m_CO2
        dm_ldoc[int(t % interval)] = m_ldoc
        dm_rdoc[int(t % interval)] = m_rdoc
        dm_sdoc[int(t % interval)] = m_sdoc
        dm_ldon[int(t % interval)] = m_ldon
        dm_rdon[int(t % interval)] = m_rdon
        dm_sdon[int(t % interval)] = m_sdon
        
        # Record growth rates at each timestep
        du_aoa[int(t % interval)] = mu_aoa
        du_lHet[int(t % interval)] = mu_lHet
        du_lHet_C[int(t % interval)] = lHet_g_C
        du_lHet_N[int(t % interval)] = lHet_g_N
        du_rHet[int(t % interval)] = mu_rHet
        du_rHet_C[int(t % interval)] = rHet_g_C
        du_rHet_N[int(t % interval)] = rHet_g_N
        du_sHet[int(t % interval)] = mu_sHet
        du_sHet_C[int(t % interval)] = sHet_g_C
        du_sHet_N[int(t % interval)] = sHet_g_N
        
        # Record biomass at each timestep
        db_aoa[int(t % interval)] = m_baoa
        db_lHet[int(t % interval)] = m_blHet
        db_rHet[int(t % interval)] = m_brHet
        db_sHet[int(t % interval)] = m_bsHet
        
        # Record biomass yields at each timestep
        dy_lHet[int(t % interval)] = y_lHet_pB
        dy_rHet[int(t % interval)] = y_rHet_pB
        dy_sHet[int(t % interval)] = y_sHet_pB
        
        # Record rates at each timestep
        dc_lHet_ldoc[int(t % interval)] = lHet_ldoc_cons
        dc_lHet_ldon[int(t % interval)] = lHet_ldon_cons
        dc_rHet_rdoc[int(t % interval)] = rHet_rdoc_cons
        dc_rHet_rdon[int(t % interval)] = rHet_rdon_cons
        dc_sHet_sdoc[int(t % interval)] = sHet_sdoc_cons
        dc_sHet_sdon[int(t % interval)] = sHet_sdon_cons
        dc_lHet_O2[int(t % interval)] = lHet_oxy_cons
        dc_rHet_O2[int(t % interval)] = rHet_oxy_cons
        dc_sHet_O2[int(t % interval)] = sHet_oxy_cons
        dp_lHet_CO2[int(t % interval)] = lHet_co2_prod
        dp_rHet_CO2[int(t % interval)] = rHet_co2_prod
        dp_sHet_CO2[int(t % interval)] = sHet_co2_prod
        dp_lHet_NH4[int(t % interval)] = lHet_nh4_prod
        dp_rHet_NH4[int(t % interval)] = rHet_nh4_prod
        dp_sHet_NH4[int(t % interval)] = sHet_nh4_prod
        dp_lHet_sdoc[int(t % interval)] = lHet_sdoc_prod
        dp_lHet_rdoc[int(t % interval)] = lHet_rdoc_prod
        dp_lHet_sdon[int(t % interval)] = lHet_sdon_prod
        dp_lHet_rdon[int(t % interval)] = lHet_rdon_prod
        d_lHet_RQ[int(t % interval)] = lHet_RQ
        d_rHet_RQ[int(t % interval)] = rHet_RQ
        d_sHet_RQ[int(t % interval)] = sHet_RQ

        # pulse OM and O2 into the chemostat
        pulse_interval = int((1.0 / dt) * pulse_int)
        if pulse_interval < 1:
            pulse_interval = 1
        if (t % pulse_interval) == 0:
            m_ldoc = m_ldoc + pulse_ldoc
            m_ldon = m_ldon + pulse_ldon
            m_rdoc = m_rdoc + pulse_rdoc
            m_rdon = m_rdon + pulse_rdon
            m_sdoc = m_sdoc + pulse_sdoc
            m_sdon = m_sdon + pulse_sdon
            m_NH4 = m_NH4 + pulse_NH4
            m_baoa = m_baoa + pulse_baoa
            m_blHet = m_blHet + pulse_blHet
            m_brHet = m_brHet + pulse_brHet
            m_bsHet = m_bsHet + pulse_bsHet
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_ldoc[i] = m_ldoc
            out_ldon[i] = m_ldon
            out_rdoc[i] = m_rdoc
            out_rdon[i] = m_rdon
            out_sdoc[i] = m_sdoc
            out_sdon[i] = m_sdon
            out_O2[i] = m_O2
            out_NH4[i] = m_NH4 
            out_CO2[i] = m_CO2
            out_baoa[i] = m_baoa
            out_blHet[i] = m_blHet
            out_brHet[i] = m_brHet
            out_bsHet[i] = m_bsHet
            out_uaoa[i] = mu_aoa
            out_ulHet[i] = mu_lHet
            out_ulHet_C[i] = lHet_g_C
            out_ulHet_N[i] = lHet_g_N
            out_urHet[i] = mu_rHet
            out_urHet_C[i] = rHet_g_C
            out_urHet_N[i] = rHet_g_N
            out_usHet[i] = mu_sHet
            out_usHet_C[i] = sHet_g_C
            out_usHet_N[i] = sHet_g_N
            out_ylHet[i] = y_lHet_pB
            out_yrHet[i] = y_rHet_pB
            out_ysHet[i] = y_sHet_pB
            out_c_lHet_ldoc[i] = lHet_ldoc_cons
            out_c_lHet_ldon[i] = lHet_ldon_cons
            out_c_lHet_O2[i] = lHet_oxy_cons
            out_p_lHet_CO2[i] = lHet_co2_prod
            out_p_lHet_sdoc[i] = lHet_sdoc_prod
            out_p_lHet_sdon[i] = lHet_sdon_prod
            out_p_lHet_rdoc[i] = lHet_rdoc_prod
            out_p_lHet_rdon[i] = lHet_rdon_prod
            out_p_lHet_NH4[i] = lHet_nh4_prod
            out_lHet_RQ[i] = lHet_RQ
            out_c_rHet_rdoc[i] = rHet_rdoc_cons
            out_c_rHet_rdon[i] = rHet_rdon_cons
            out_c_rHet_O2[i] = rHet_oxy_cons
            out_p_rHet_CO2[i] = rHet_co2_prod
            out_p_rHet_NH4[i] = rHet_nh4_prod
            out_rHet_RQ[i] = rHet_RQ
            out_c_sHet_sdoc[i] = sHet_sdoc_cons
            out_c_sHet_sdon[i] = sHet_sdon_cons
            out_c_sHet_O2[i] = sHet_oxy_cons
            out_p_sHet_CO2[i] = sHet_co2_prod
            out_p_sHet_NH4[i] = sHet_nh4_prod
            out_sHet_RQ[i] = sHet_RQ

                  
    return [out_ldoc, out_ldon, out_rdoc, out_rdon, out_sdoc, out_sdon, out_O2, out_CO2, out_NH4, \
            out_baoa, out_blHet, out_brHet, out_bsHet, \
            out_uaoa, \
            out_ulHet, out_ulHet_C, out_ulHet_N, \
            out_urHet, out_urHet_C, out_urHet_N, \
            out_usHet, out_usHet_C, out_usHet_N, \
            out_ylHet, out_yrHet, out_ysHet, \
            out_c_lHet_ldoc, out_c_lHet_ldon, out_c_lHet_O2, \
            out_p_lHet_CO2, out_p_lHet_sdoc, out_p_lHet_sdon, out_p_lHet_rdoc, out_p_lHet_rdon, out_p_lHet_NH4, out_lHet_RQ, \
            out_c_rHet_rdoc, out_c_rHet_rdon, out_c_rHet_O2, out_p_rHet_CO2, out_p_rHet_NH4, out_rHet_RQ, \
            out_c_sHet_sdoc, out_c_sHet_sdon, out_c_sHet_O2, out_p_sHet_CO2, out_p_sHet_NH4, out_sHet_RQ, \
            dm_O2, dm_CO2, dm_NH4, dm_ldoc, dm_ldon, dm_rdoc, dm_rdon, dm_sdoc, dm_sdon, \
            du_aoa, du_lHet, du_lHet_C, du_lHet_N, du_rHet, du_rHet_C, du_rHet_N, du_sHet, du_sHet_C, du_sHet_N, \
            db_aoa, db_lHet, db_rHet, db_sHet, dy_lHet, dy_rHet, dy_sHet, \
            dc_lHet_ldoc, dc_lHet_ldon, dc_lHet_O2, dp_lHet_CO2, dp_lHet_sdoc, dp_lHet_sdon, dp_lHet_rdoc, dp_lHet_rdon, dp_lHet_NH4, d_lHet_RQ, \
            dc_rHet_rdoc, dc_rHet_rdon, dc_rHet_O2, dp_rHet_CO2, dp_rHet_NH4, d_rHet_RQ, \
            dc_sHet_sdoc, dc_sHet_sdon, dc_sHet_O2, dp_sHet_CO2, dp_sHet_NH4, d_sHet_RQ \
            ]

