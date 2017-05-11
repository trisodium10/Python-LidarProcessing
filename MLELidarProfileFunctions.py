# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:03:03 2017

@author: mhayman
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from time import time,gmtime,strftime
import LidarProfileFunctions as lp


def Num_Gradient(func,x0,step_size=1e-3):
    Gradient = np.zeros(x0.size)
    for ai in range(x0.size):
        xu = x0.copy()
        xl = x0.copy()
        if x0[ai] != 0:
            xu[ai] = x0[ai]*(1+step_size)            
            xl[ai] = x0[ai]*(1-step_size)
#            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
        else:
            xu[ai] = step_size
            xl[ai] = -step_size
#            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
        
        Gradient[ai] = (func(xu)-func(xl))/(xu[ai]-xl[ai])
    return Gradient
    
def LREstimate_buildConst(geo_correct,MolScale,Cam,beta_m,beta_a,range_array):
    geofun = 1/np.interp(range_array,geo_correct['geo_prof'][:,0],geo_correct['geo_prof'][:,1])
    ConstTerms = 1.0/geo_correct['Nprof']*MolScale*(beta_m+Cam*beta_a)*geofun*np.exp(-2.0*np.cumsum(8*np.pi/3*beta_m)*(range_array[1]-range_array[0]))/range_array**2
    return ConstTerms
    

def LREstimateTotalxvalid(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,weights=np.array([1])):
    N = Mprof.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain
    Gc = x[2]   # combinded gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[3:Nx+3])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+3:])  # aerosol backscatter terms
    Molmodel = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Mprof_bg
    Combmodel = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Cprof_bg
    dxvalid = np.diff(xvalid)
    deriv = np.nansum(np.diff(sLR[xvalid])/dxvalid)*lam[0] + np.nansum(np.diff(Baer[xvalid])/dxvalid)*lam[1]
    ErrRet = np.nansum(weights*(Molmodel-Mprof*np.log(Molmodel)))+np.nansum(weights*(Combmodel-Cprof*np.log(Combmodel)))+deriv
    return ErrRet
    
def LREstimateTotalxvalid_prime(x,xvalid,Mprof,Cprof,mol_bs_coeff,Const,lam,Mprof_bg=0,Cprof_bg=0,weights=np.array([1])):
    N = Mprof.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain 
    Gc = x[2]  # combined gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[3:Nx+3])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+3:])  # aerosol backscatter terms

    Molmodel0 = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))
    Molmodel = Molmodel0+Mprof_bg
    
    Combmodel0 = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))
    Combmodel = Combmodel0+Cprof_bg
    
    # useful definitions for gradient calculations
    e0m = weights*(1-Mprof/Molmodel)
    e0c = weights*(1-Cprof/Combmodel)
    grad0m = np.sum(e0m*Molmodel0)-np.cumsum(e0m*Molmodel0)
    grad0c = np.sum(e0c*Combmodel0)-np.cumsum(e0c*Combmodel0)
    
    # lidar ratio gradient terms:
    gradErrS = -2*Baer*sLR*(grad0m+grad0c)
    gradErrS[np.nonzero(np.isnan(gradErrS))] = 0

    # backscatter cross section gradient terms:
    gradErrB = -2*sLR*Baer*(grad0m+grad0c)+Baer*e0m*Gm*Const*Cam*np.exp(-2*np.cumsum(sLR*Baer))+Baer*e0c*Const*Gc*np.exp(-2*np.cumsum(sLR*Baer))
    gradErrB[np.nonzero(np.isnan(gradErrB))] = 0

    # cross talk gradient term
    gradErrCam = np.nansum(e0m*Gm*Const*Baer*np.exp(-2*np.cumsum(sLR*Baer)))

    # combined gain gradient term
    gradErrGc = np.nansum(e0c*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))
    
    # molecular gain gradient term
    gradErrGm = np.nansum(e0m*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))

    # total variance gradient terms
#    gradErr[np.nonzero(np.isnan(gradErr))] = 0
#    gradErr = gradErr[xvalid]
    dxvalid = np.diff(xvalid)
    gradErrStv = np.zeros(Nx)
    gradErrBtv = np.zeros(Nx)
    gradpenS = lam[0]*np.sign(np.diff(sLR[xvalid]))/dxvalid
    gradpenS[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrStv[:-1] = gradErrStv[:-1]-gradpenS
    gradErrStv[1:] = gradErrStv[1:]+gradpenS
    
    gradpenB = lam[0]*np.sign(np.diff(Baer[xvalid]))/dxvalid
    gradpenB[np.nonzero(np.isnan(gradpenS))] = 0
    gradErrBtv[:-1] = gradErrBtv[:-1]-gradpenB
    gradErrBtv[1:] = gradErrBtv[1:]+gradpenB
    
    gradErr = np.zeros(3+2*Nx)
    gradErr[0] = gradErrCam
    gradErr[1] = gradErrGm
    gradErr[2] = gradErrGc
    gradErr[3:Nx+3] = gradErrS[xvalid]+gradErrStv
    gradErr[Nx+3:] = gradErrB[xvalid]+gradErrBtv
    
    return gradErr.flatten()
    
def LREstimateTotalxvalid_Jacobian(x,xvalid,mol_bs_coeff,Const):
    # define delta function matrix
    Jm = np.zeros((Const.size,xvalid.size))
    Jm[xvalid,np.arange(xvalid.size)] = 1
    # define heaviside matrix
    JmI = np.zeros((Const.size,xvalid.size))
    JmI[xvalid+1,np.arange(xvalid.size)] = 1
    N = Const.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain 
    Gc = x[2]  # combined gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[3:Nx+3])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+3:])  # aerosol backscatter terms

    Molmodel0 = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))
#    Molmodel = Molmodel0+Mprof_bg
    
    Combmodel0 = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))
#    Combmodel = Combmodel0+Cprof_bg
    
    
    # lidar ratio gradient terms:
    Jac_sLR_m = -2*Baer[:,np.newaxis]*sLR[:,np.newaxis]*(Molmodel0[:,np.newaxis])*JmI
    Jac_sLR_c = -2*Baer[:,np.newaxis]*sLR[:,np.newaxis]*(Combmodel0[:,np.newaxis])*JmI
#    gradErrS = -2*Baer*sLR*(grad0m+grad0c)
#    gradErrS[np.nonzero(np.isnan(gradErrS))] = 0

    # backscatter cross section gradient terms:
    Jac_Baer_m = -2*Baer[:,np.newaxis]*sLR[:,np.newaxis]*(Molmodel0[:,np.newaxis])*JmI \
                          +(Baer*Gm*Const*Cam*np.exp(-2*np.cumsum(sLR*Baer)))[:,np.newaxis]*Jm
    Jac_Baer_c = -2*Baer[:,np.newaxis]*sLR[:,np.newaxis]*(Combmodel0[:,np.newaxis])*JmI \
                          +(Baer*Const*Gc*np.exp(-2*np.cumsum(sLR*Baer)))[:,np.newaxis]*Jm
#    gradErrB = -2*sLR*Baer*(grad0m+grad0c)+Baer*e0m*Gm*Const*Cam*np.exp(-2*np.cumsum(sLR*Baer))+Baer*e0c*Const*Gc*np.exp(-2*np.cumsum(sLR*Baer))
#    gradErrB[np.nonzero(np.isnan(gradErrB))] = 0

    # cross talk gradient term
    Jac_Cam = (Gm*Const*Baer*np.exp(-2*np.cumsum(sLR*Baer)))[:,np.newaxis]*Jm
#    gradErrCam = np.nansum(e0m*Gm*Const*Baer*np.exp(-2*np.cumsum(sLR*Baer)))

    # combined gain gradient term
    Jac_Gc = ((mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))[:,np.newaxis]*Jm
#    gradErrGc = np.nansum(e0c*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))
    
    # molecular gain gradient term
    Jac_Gm = ((mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))[:,np.newaxis]*Jm
#    gradErrGm = np.nansum(e0m*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer)))
        
#    Jac = Jac_sLR+Jac_Baer+Jac_Cam+Jac_Gc+Jac_Gm
    
    return Jac_sLR_m,Jac_sLR_c,Jac_Baer_m,Jac_Baer_c,Jac_Cam,Jac_Gc,Jac_Gm
    
def ProfilesTotalxvalid(x,xvalid,mol_bs_coeff,Const,Mprof_bg=0,Cprof_bg=0):
    N = mol_bs_coeff.size  # length of profile    
    Nx = xvalid.size
    Cam = x[0]
    Gm = x[1]  # molecular gain
    Gc = x[2]   # combinded gain
    sLR = np.zeros(N)
    sLR[xvalid] = np.exp(x[3:Nx+3])  # lidar ratio terms
    Baer = np.zeros(N)
    Baer[xvalid] = np.exp(x[Nx+3:])  # aerosol backscatter terms
    Molmodel = Gm*(mol_bs_coeff+Cam*Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Mprof_bg
    Combmodel = Gc*(mol_bs_coeff+Baer)*Const*np.exp(-2*np.cumsum(sLR*Baer))+Cprof_bg
    return Molmodel,Combmodel
    
def MLE_Estimate_OptProp(MolRaw,CombRaw,Beta_A,geo_data,sonde_pres,sonde_temp,sonde_range, \
        tlim=np.array([np.nan,np.nan]),minSNR=2.0, \
        lam=np.array([4,0.000001]),sLRinit=np.array([-24.5,-8.5]),dG=0.04, \
        fitfilt=True):
    """
    MLE_Estimate_OptProp(
    Maximum Likelihood Estimator for aerosol backscatter, extinction and lidar ratio
    Performs optimization on one profile at a time and loops through

    Inputs:
        MolRaw - the raw nonlinear count corrected molecular profile of type LidarProfile
        CombRaw - the raw nonlinear count corrected Combined profile of type LidarProfile
        Beta_A - directly calculated aerosol backscatter cross section
        geo_data - library from loading the geo cal file
        sonde_pres - raw pressure profile from sonde or model
        sonde_temp - raw temperature profile from sonde or model
        sonde_range - array of sonde range corresponding to pressure and temperature
        tlim - array containing the desired start and stop times (in seconds)
            if not used, the entirity of the profile will be processed
        minSNR - minimum directly retrieved aerosol backscatter SNR to count the bin
            as a valid aerosol detection.  defaults to 2 if not provided.
        lam - array of total variance penalties for
            lam[0] - lidar ratio estimate
            lam[1] - aerosol backscatter estimate
        sLRinit - array containing assumed functional relationship between aerosol
            backscatter coefficient and lidar ratio for initial guess in the 
            estimater.  The formula for the initial guess is:
            sLR0 = (sLRinit[1]*np.log10(aer_backscatter)-sLRinit[0])*dR
        dG - allowed gain variation between profiles
        fitfilt - if the fit looks bad, don't include it in xvalid_mle

    Outputs:    
    
    """
    # copy all the profiles so we don't change any of them in the routnine
    MolRawE = MolRaw.copy()
    CombRawE = CombRaw.copy()
    Beta_A_E = Beta_A.copy()
    
    dR = MolRaw.mean_dR  # store the profile range resolution    
    
    # slice the time on the profiles if specific limits are given
    
    if not any(np.isnan(tlim)):
        MolRawE.slice_time(tlim)
        CombRawE.slice_time(tlim)
        Beta_A_E.slice_time(tlim)
    
    
    # Estimate Profile Backgrounds
    Nprof = MolRawE.NumProfList
    FitMol = MolRawE.NumProfList[:,np.newaxis]*MolRawE.profile
    FitMol_bg = MolRawE.NumProfList*np.nanmean(MolRawE.profile[:,-50:],axis=1)
    
    FitComb = CombRawE.NumProfList[:,np.newaxis]*CombRawE.profile
    FitComb_bg = CombRawE.NumProfList*np.nanmean(CombRawE.profile[:,-50:],axis=1)  
    
    # pad aerosol profile if it has been fewer range bins than the raw photon counts    
    FitAer = Beta_A_E.profile
    FitAer = np.hstack((FitAer,np.zeros((FitAer.shape[0],FitMol.shape[1]-FitAer.shape[1]))))
    FitAer[np.nonzero(np.isnan(FitAer))[0]]=0
    
    # Estimate molecular backscatter
    Tsonde = np.interp(CombRawE.range_array,sonde_range,sonde_temp)
    Psonde = np.interp(CombRawE.range_array,sonde_range,sonde_pres)
    # note the operating wavelength of the lidar is 532 nm
    beta_m_sonde = 5.45*(550.0/CombRawE.wavelength)**4*1e-32*Psonde/(Tsonde*lp.kB)

    # build the profile constant terms that are repeated in each profile.
    # These account for system efficiency, geometric overlap, molcular extinction
    # 1/r^2 loss.  The function called here also includes molecular backscatter
    # so we have to divide that term out after the fact.
    ConstTerms = Nprof[:,np.newaxis]*LREstimate_buildConst(geo_data,1,0.0,beta_m_sonde,np.ones(FitComb.shape[1]),MolRawE.range_array)
    ConstTerms = ConstTerms/beta_m_sonde[np.newaxis,:]
    
    # Pre allocate solution variables
    CamList = np.zeros(CombRawE.time.size)
    GmList = np.zeros(CombRawE.time.size)
    GcList = np.zeros(CombRawE.time.size)
    
    # Pre allocate the profile variables
    sLR_mle = Beta_A_E.copy()
    sLR_mle.descript = 'Maximum Likelihood Estimate of Aerosol Lidar Ratio sr'
    sLR_mle.label = 'Aerosol Lidar Ratio'
    sLR_mle.profile_type = 'Aerosol Lidar Ratio [$sr$]'
    sLR_mle.profile_variance = sLR_mle.profile_variance*0.0

    beta_a_mle = Beta_A_E.copy()
    beta_a_mle.descript = 'Maximum Likelihood Estimate of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    beta_a_mle.label = 'Aerosol Backscatter Coefficient'
    beta_a_mle.profile_type = 'Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]'
    beta_a_mle.profile_variance = beta_a_mle.profile_variance*0.0
    
    alpha_a_mle = Beta_A_E.copy()
    alpha_a_mle.descript = 'Maximum Likelihood Estimate of Aerosol Extinction Coefficient in m^-1'
    alpha_a_mle.label = 'Aerosol Extinction Coefficient'
    alpha_a_mle.profile_type = 'Aerosol Extinction Coefficient [$m^{-1}$]'
    alpha_a_mle.profile_variance = alpha_a_mle.profile_variance*0.0
    
    fit_mol_mle = MolRawE.copy()
    fit_mol_mle.descript = 'Maximum Likelihood Estimate of ' + fit_mol_mle.descript
    fit_comb_mle = CombRawE.copy()
    fit_comb_mle.descript = 'Maximum Likelihood Estimate of ' + fit_comb_mle.descript
    
    xvalid_mle = Beta_A_E.copy()
    xvalid_mle.descript = 'Maximum Likelihood Estimated Data Points'
    xvalid_mle.label = 'MLE Data Points'
    xvalid_mle.profile_type = 'MLE Data Points'
    xvalid_mle.profile = xvalid_mle.profile*0
    
    
    for pI in range(FitMol.shape[0]):
       # Determine valid aerosol data points based on the SNR of the directly calculated aerosol profile
        xvalid = np.nonzero((Beta_A_E.profile[pI,:]/np.sqrt(Beta_A_E.profile_variance[pI,:])).flatten()>minSNR)[0]
        xvalid_mle.profile[pI,xvalid] = 1
        
        # setup the optimization functions for this profile
        FitProfMol = lambda x: 1e-3*LREstimateTotalxvalid(x,xvalid,FitMol[pI,:],FitComb[pI,:],beta_m_sonde,ConstTerms[pI,:],lam,Mprof_bg=FitMol_bg[pI],Cprof_bg=FitComb_bg[pI])
        FitProfMolDeriv = lambda x: 1e-3*LREstimateTotalxvalid_prime(x,xvalid,FitMol[pI,:],FitComb[pI,:],beta_m_sonde,ConstTerms[pI,:],lam,Mprof_bg=FitMol_bg[pI],Cprof_bg=FitComb_bg[pI])
        
        # setup the general profile bounds
        bndsP = np.zeros((3+2*xvalid.size,2))
        bndsP[3:(3+xvalid.size),1] = np.log(1e2*dR)    # maximum lidar ratio (natural log)
        
        bndsP[(3+xvalid.size):,0] = np.log(1e-12)   # minimum aerosol backscatter (natural log), absolute definition
        bndsP[(3+xvalid.size):,1] = np.log(1e-1)    # maximum aerosol backscatter (natural log), absolute definition
        
#        bndsP[(3+xvalid.size):,0] = np.log(Beta_A_E.profile[pI,xvalid]-np.sqrt(Beta_A_E.profile_variance[pI,xvalid]))   # minimum aerosol backscatter (natural log), profile definition
#        bndsP[np.nonzero(np.isnan(bndsP))] = np.log(1e-12)
#        bndsP[(3+xvalid.size):,1] = np.log(1e-1)    # maximum aerosol backscatter (natural log), profile definition
        
        bndsP[0,0] = 0.0                            # minimum aerosol to molecular cross talk
        bndsP[0,1] = 0.2                            # maximum aerosol to molecular cross talk
        bndsP[1,0] = 0.6                            # minimum molecular channel gain
        bndsP[1,1] = 0.9                            # maximum molecular channel gain
        bndsP[2,0] = 0.9                            # minimum combined channel gain
        bndsP[2,1] = 1.2                            # maximum combined channel gain
        
        # setup initial guess for optimization
        x0 = np.ones(3+2*xvalid.size); # allocate the array
        x0[3:(3+xvalid.size)] = np.log((sLRinit[1]*np.log10(FitAer[pI,xvalid])-sLRinit[0])*dR)  # set lidar ratio initial guess
        x0[(3+xvalid.size):] = np.log(FitAer[pI,xvalid])  # set aerosol backscatter initial guess to the directly calculated values
        
        # if this is the first run, use general values for initial guesses of cross talk and gain
        # otherwise base those on the previously determined values
        if pI == 0:
            x0[0] = 0.1  # Cam
            x0[1] = 0.7545  # Gt
            x0[2] = 1.0104
        else:
    #        # use previous solution to seed the next time step
    #        xoverlap = np.nonzero(xvalid2D[pI,:]*xvalid2D[pI-1,:])[0]
    #        xlast = np.nonzero(xvalid2D[pI-1,:])[0]
    #        sLR0 = np.log(sLRinitial*dR)*np.ones(FitComb.shape[1])
    #        sLR0[xlast] = sLR2D[pI-1,xlast]
    #        x0[3:(3+xvalid.size)] = sLR0[xvalid]
            
            # use previous coefficients
            x0[0] = CamList[pI-1]
            x0[1] = GmList[pI-1]
            x0[2] = GcList[pI-1]
            # dynamically adjust the bounds on gain terms
            # this version does not impose absolute limits
            bndsP[1,0] = GmList[pI-1]-dG
            bndsP[1,1] = GmList[pI-1]+dG
            bndsP[2,0] = GcList[pI-1]-dG
            bndsP[2,1] = GcList[pI-1]+dG      
        
        # run the optimizor        
        wMol = scipy.optimize.fmin_slsqp(FitProfMol,x0,bounds=bndsP,fprime=FitProfMolDeriv ,iter=1000) # fprime=FitProfMolDeriv ,disp=0
        
        # store the optimizor solutions
        beta_a_mle.profile[pI,xvalid] = np.exp(wMol[(3+xvalid.size):])
        sLR_mle.profile[pI,xvalid] = np.exp(wMol[3:(3+xvalid.size)])/dR
        
        CamList[pI] = wMol[0]
        GmList[pI] = wMol[1]
        GcList[pI] = wMol[2]
        
#        aer_bs_sol = np.zeros(FitComb.size)
#        aer_bs_sol[xvalid] = np.exp(wMol[(3+xvalid.size):])
#        sLRsol = np.zeros(FitComb.size)
#        sLRsol[xvalid] = np.exp(wMol[3:(3+xvalid.size)])
        
    #    Cam = wMol[0]
    #    Gm = wMol[1]
    #    Gc = wMol[2]
        
        # store the fits obtained from MLE
        fit_mol_mle.profile[pI,:],fit_comb_mle.profile[pI,:] = \
            ProfilesTotalxvalid(wMol,xvalid,beta_m_sonde,ConstTerms[pI,:],Mprof_bg=FitMol_bg[pI],Cprof_bg=FitComb_bg[pI])
    
    # estimate extinction based on solutions from aerosol backscatter and lidar ratio
    alpha_a_mle.profile = beta_a_mle.profile*sLR_mle.profile
    
    # calculate RMS profile fit error
    ProfileErrorMol = np.sqrt(np.sum((FitMol-fit_mol_mle.profile)**2,axis=1))
    ProfileErrorComb = np.sqrt(np.sum((FitComb-fit_comb_mle.profile)**2,axis=1))
    ProfileError = np.sqrt(ProfileErrorMol**2+ProfileErrorComb**2)
    
    if fitfilt:
        # attempt to remove profiles where the fit was bad
        pbad = np.nonzero(np.abs(np.diff(ProfileError)/ProfileError[:-1])>3)[0]+1
        xvalid_mle.profile[pbad,:]= 0
        pbad = np.nonzero(np.abs(np.diff(ProfileError)/ProfileError[1:])>3)[0]
        xvalid_mle.profile[pbad,:]= 0
        
    return beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb,fit_mol_mle,fit_comb_mle