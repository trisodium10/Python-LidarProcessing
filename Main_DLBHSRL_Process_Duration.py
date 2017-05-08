# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:45:42 2017

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import LidarProfileFunctions as lp
import MLELidarProfileFunctions as mle
import scipy.interpolate

from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime

import glob

def ProfEstimateSVD2D(x,Prof,Psvd,lam):
    x = x.reshape(Prof.shape)
    deriv= np.nansum(np.diff(x,axis=0))*lam[0]+np.nansum(np.diff(x,axis=1))*lam[1]
    ErrRet = np.nansum((x+Psvd).flatten()-Prof.flatten()*np.log(x+Psvd).flatten())+deriv
    return ErrRet
def ProfEstimateSVD2D_prime(x,Prof,Psvd,lam):
    x = x.reshape(Prof.shape)
    gradErr = 1-Prof/(x+Psvd)
    gradErr[np.nonzero(np.isnan(gradErr))] = 0
    
    gradpen = lam[0]*np.sign(np.diff(x,axis=0))
    gradpen[np.nonzero(np.isnan(gradpen))] = 0
    gradErr[:-1,:] = gradErr[:-1,:]-gradpen
    gradErr[1:,:] = gradErr[1:,:]+gradpen
    
    gradpen = lam[1]*np.sign(np.diff(x,axis=1))
    gradpen[np.nonzero(np.isnan(gradpen))] = 0
    gradErr[:,:-1] = gradErr[:,:-1]-gradpen
    gradErr[:,1:] = gradErr[:,1:]+gradpen

    return gradErr.flatten()

#filePath = '/scr/eldora1/MSU_h2o_data/2016/161209NF/18/'
#fileName1 = 'Online_Raw_Data.dat'  # molecular
#fileName2 = 'Offline_Raw_Data.dat'  # combined

#Day = 4
#Month = 1
#Year = 2017
#HourLim = np.array([0,24])  # Limits on the processing time
#
#
# Set the process chunk in year, month, day and duration
# function format:
# generate_WVDIAL_day_list(startYr,startMo,startDay,startHr=0,duration=0.0,stopYr=0,stopMo=0,stopDay=0,stopHr=24)
# duration is set in hours
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,28,startHr=6.0,duration=1.0) #,stopYr=0,stopMo=0,stopDay=0,stopHr=24):
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,28,startHr=13.5,stopYr=2017,stopMo=1,stopDay=2)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,27,startHr=19.85,stopHr=19.95)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,19,startHr=22.0,duration=1.0)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,21,startHr=19.3,duration=1.5)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,17,startHr=10.0,duration=1.0)

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,27,startHr=13.75,duration=0.3)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,2,25,startHr=3,duration=1.0)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,2,22,startHr=8.0,duration=3.0)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,2)

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,4,13,startHr=9.12,stopHr=11.76)  # WV-DIAL RD Correction ,startHr=5,duration=4.0
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,4,12,duration=2*24)  # WV-DIAL RD Correction ,startHr=5,duration=4.0

Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,4,11,startHr=1,duration=5)  # WV-DIAL RD Correction ,startHr=5,duration=4.0
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,4,18,startHr=4.5,stopHr=5.4)  # WV-DIAL RD Correction ,startHr=5,duration=4.0

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,12,stopDay=19)

plotAsDays = False
getMLE_extinction = False
run_MLE = False
runKlett = False

save_as_nc = True
save_figs = True

nctag = 'with_geo'  # additional tag for netcdf and figure filename

run_geo_cal = False

ProcStart = datetime.date(Years[0],Months[0],Days[0])

DateLabel = ProcStart.strftime("%A %B %d, %Y")

MaxAlt = 12e3 #12e3

KlettAlt = 14e3  # altitude where Klett inversion starts

tres = 0.5*60.0  # resolution in time points (2 sec)
zres = 1.0  # resolution in altitude points (75 m)

use_diff_geo = False   # no diff geo correction after April ???
use_geo = True

use_mask = False
SNRmask = 0.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

#kB = 1.3806504e-23;
#c = 3e8

MCSbins = 280*2  # number of bins in a range resolved profile,  280-typical became 1400 on 2/22/2017
BinWidth = 250e-9 # MCS timing bin width in seconds.  typically 500e-9 before April ?.  250e-9 after April ?
dR = BinWidth*lp.c/2  # profile range resolution (500e-9*c/2)-typical became 100e-9*c/2 on 2/22/2017
dt = 2  # profile accumulation time
Roffset = ((1.25+0.5)-0.5/2)*150  # offset in range

BGIndex = -50; # negative number provides an index from the end of the array
Cam = 0.00 # Cross talk of aerosols into the molecular channel - 0.005 on Dec 21 2016 after 18.5UTC
            # 0.033 found for 4/18/2017 11UTC extinction test case

save_data_path = '/h/eol/mhayman/HSRL/DLBHSRL/Processed_Data/'
save_fig_path = '/h/eol/mhayman/HSRL/DLBHSRL/Processed_Data/Plots/'
sonde_path = '/scr/eldora1/HSRL_data/'


#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161212.npz'
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161219_2.npz'
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161221_2.npz'
diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161227.npz'  #Provided by Scott
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20170301.npz'  #Provided by Scott
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20170222.npz'  #Update with Zeeman calibration

diff_geo_data = np.load(diff_geo_file)
diff_geo_corr = diff_geo_data['diff_geo_prof']
#diff_lim_index = 14  # correction is not applied below this index - used on 20161219_2
#diff_geo_corr[:diff_lim_index] = 1.2/(1.0/9*2.25*1.1*1.25/1.39)

#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161219.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161213.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161227.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161227.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170119.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170120_0.npz'  # obtained using OD, includes variance in profile
geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170413.npz'  # obtained using OD, includes variance in profile
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170418.npz'  # obtained using OD, includes variance in profile

if use_geo:
    geo_data = np.load(geo_file)
    geo_corr = geo_data['geo_prof']
    geo_corr0 = geo_corr[100,1]  # normalize to bin 100
    geo_corr[:,1] = geo_corr[:,1]/geo_corr0
    if any('sonde_scale' in s for s in geo_data.keys()):
        sonde_scale=geo_data['sonde_scale']/geo_corr0
    else:
        sonde_scale=1.0/geo_corr0
else:
    sonde_scale=1.0

basepath = '/scr/eldora1/MSU_h2o_data/'
FieldLabel = 'NF'
MolFileBase = 'Online_Raw_Data.dat'
CombFileBase = 'Offline_Raw_Data.dat'

# define time grid for lidar signal processing to occur
MasterTime = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600+tres,tres)

if save_as_nc or save_figs:
    ncfilename0 = lp.create_ncfilename('DLBHSRL',Years,Months,Days,Hours,tag=nctag)
    ncfilename = save_data_path+ncfilename0
    figfilename = save_fig_path + ncfilename0[:-3] + nctag

firstFile = True

for dayindex in range(Years.size):

    if Days[dayindex] < 10:
        DayStr = '0' + str(Days[dayindex])
    else:
        DayStr = str(Days[dayindex])
        
    if Months[dayindex] < 10:
        MonthStr = '0' + str(Months[dayindex])
    else:
        MonthStr = str(Months[dayindex])
    
    YearStr = str(Years[dayindex])
    HourLim = Hours[:,dayindex]
    
    # calculate the time offset due to being a different day than we started with
    if firstFile:
        deltat_0 = 0;
    else:
        deltat_0_date = datetime.date(Years[dayindex],Months[dayindex],Days[dayindex])-datetime.date(Years[0],Months[0],Days[0])
        deltat_0 = deltat_0_date.days
        
    FilePath0 = basepath + YearStr + '/' + YearStr[-2:] + MonthStr + DayStr + FieldLabel
    
    SubDirs = glob.glob(FilePath0+'/*/')
    
    for idir in range(len(SubDirs)):
        Hour = np.double(SubDirs[idir][-3:-1])
        if Hour >= np.floor(HourLim[0]) and Hour <= HourLim[1]:
            loadfile_mol = SubDirs[idir]+MolFileBase
            loadfile_comb = SubDirs[idir]+CombFileBase
            Hour = np.double(SubDirs[idir][-3:-1])
            
            #### LOAD NETCDF DATA ####
            mol_data,mol_vars = lp.read_WVDIAL_binary(loadfile_mol,MCSbins)
            hi_data,hi_vars = lp.read_WVDIAL_binary(loadfile_comb,MCSbins)
                   
            
            timeDataM =3600*24*(np.remainder(mol_vars[0,:],1)+deltat_0)
            timeDataT =3600*24*(np.remainder(hi_vars[0,:],1)+deltat_0)
            
            shots_m = np.ones(np.shape(timeDataM))*np.mean(mol_vars[6,:])
            shots_t = np.ones(np.shape(timeDataT))*np.mean(mol_vars[6,:])
            
            itimeBad = np.nonzero(np.diff(timeDataM)<0)[0]        
            if itimeBad.size > 0:
                timeDataM[itimeBad+1] = timeDataM[itimeBad]+dt
                
            itimeBad = np.nonzero(np.diff(timeDataT)<0)[0]        
            if itimeBad.size > 0:
                timeDataT[itimeBad+1] = timeDataT[itimeBad]+dt
            
    #        print timeDataM.size
    #        print timeDataT.size        
            
            # load profile data
            if firstFile:
    #            timeMaster = np.arange(np.floor(np.min((timeDataM[0],timeDataT[0]))),np.floor(np.min((timeDataM[0],timeDataT[0]))),tres)
                
                Molecular = lp.LidarProfile(mol_data.T,timeDataM,label='Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=-Roffset/dR,lidar='DLB-HSRL',shot_count=shots_m,binwidth=BinWidth,StartDate=ProcStart)
#                RemMol = Molecular.time_resample(delta_t=tres,t0=HourLim[0]*3600,update=True,remainder=True)
                RemMol = Molecular.time_resample(tedges=MasterTime,update=True,remainder=True)
                
                CombHi = lp.LidarProfile(hi_data.T,timeDataT,label='Total Backscatter Channel',descript = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns',bin0=-Roffset/dR,lidar='DLB-HSRL',shot_count=shots_t,binwidth=BinWidth,StartDate=ProcStart)
#                RemCom = CombHi.time_resample(delta_t=tres,t0=HourLim[0]*3600,update=True,remainder=True)
                RemCom = CombHi.time_resample(tedges=MasterTime,update=True,remainder=True)
                
                firstFile = False
                
                
                
            else:
#                timeMaster = np.arange(Molecular.time[0]+tres,np.floor(np.min((timeDataM[0],timeDataT[0]))),tres)
                if np.size(RemMol.time) > 0:
                    MolTmp = lp.LidarProfile(mol_data.T,timeDataM,label='Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=0,lidar='DLB-HSRL',shot_count=shots_m,binwidth=BinWidth,StartDate=ProcStart)
                    MolTmp.cat_time(RemMol)
    #                mol_data = np.hstack((RemMol.profile.T,mol_data))
    #                MolTmp = lp.LidarProfile(mol_data.T,dt*np.arange(mol_data.shape[1])+RemMol.time[-1]+Molecular.mean_dt,label='Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=0,lidar='DLB-HSRL')
#                    RemMol = MolTmp.time_resample(delta_t=tres,t0=(Molecular.time[-1]+tres),update=True,remainder=True)
                    RemMol = MolTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    Molecular.cat_time(MolTmp,front=False)
                    
                    ComTmp = lp.LidarProfile(hi_data.T,timeDataT,label='Total Backscatter Channel',descript = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns',bin0=0,lidar='DLB-HSRL',shot_count=shots_t,binwidth=BinWidth,StartDate=ProcStart)
                    ComTmp.cat_time(RemCom)
    #                hi_data = np.hstack((RemCom.profile.T,hi_data))
    #                ComTmp = lp.LidarProfile(hi_data.T,dt*np.arange(hi_data.shape[1])+RemCom.time[-1]+CombHi.mean_dt,label='Total Backscatter Channel',descript = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns',bin0=0,lidar='DLB-HSRL')
#                    RemCom = ComTmp.time_resample(delta_t=tres,t0=(CombHi.time[-1]+tres),update=True,remainder=True)
                    RemCom = ComTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    CombHi.cat_time(ComTmp,front=False)
                else:
                    MolTmp = lp.LidarProfile(mol_data.T,timeDataM,label='Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=0,lidar='DLB-HSRL',shot_count=shots_m,binwidth=BinWidth,StartDate=ProcStart)
    #                MolTmp = lp.LidarProfile(mol_data.T,dt*np.arange(mol_data.shape[1])+Molecular.time[-1]+Molecular.mean_dt,label='Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=0,lidar='DLB-HSRL')
#                    RemMol = MolTmp.time_resample(delta_t=tres,t0=(Molecular.time[-1]+tres),update=True,remainder=True)
                    RemMol = MolTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    Molecular.cat_time(MolTmp,front=False)
                    
                    ComTmp = lp.LidarProfile(hi_data.T,timeDataT,label='Total Backscatter Channel',descript = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns',bin0=0,lidar='DLB-HSRL',shot_count=shots_t,binwidth=BinWidth,StartDate=ProcStart)                
    #                ComTmp = lp.LidarProfile(hi_data.T,dt*np.arange(hi_data.shape[1])+CombHi.time[-1]+CombHi.mean_dt,label='Total Backscatter Channel',descript = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns',bin0=0,lidar='DLB-HSRL')
#                    RemCom = ComTmp.time_resample(delta_t=tres,t0=(CombHi.time[-1]+tres),update=True,remainder=True)
                    RemCom = ComTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    CombHi.cat_time(ComTmp,front=False)
                    
            
#            print(Molecular.profile.shape)
#            print(CombHi.profile.shape)


#plt.figure(figsize=(15,5)); 
#plt.pcolor(Molecular.time/3600,Molecular.range_array*1e-3, np.log10(1e9*Molecular.profile.T/Molecular.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Molecular Count Rate [Hz]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(CombHi.time/3600,CombHi.range_array*1e-3, np.log10(1e9*CombHi.profile.T/CombHi.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Combined Count Rate [Hz]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)


# Update the HourLim definition to account for multiple days.  Plots use this
# to display only the desired plot portion.
HourLim = np.array([Hours[0,0],Hours[1,-1]+deltat_0*24])

Molecular.slice_time(HourLim*3600)
MolRaw = Molecular.copy()
#Molecular.nonlinear_correct(38e-9);
Molecular.bg_subtract(BGIndex)




CombHi.slice_time(HourLim*3600)
CombRaw = CombHi.copy()
CombHi.nonlinear_correct(29.4e-9);
CombHi.bg_subtract(BGIndex)


#ncfilename = '/h/eol/mhayman/write_py_netcdf3.nc'
#CombHi.write2nc(ncfilename)
#aer_beta_dlb.write2nc(ncfilename)



#plt.figure(); 
#plt.pcolor(np.log(Molecular.profile_variance).T);

#MolInt = Molecular.copy();
#MolInt.time_integrate();
#CombInt = CombHi.copy();
#CombInt.time_integrate();
#plt.figure(); plt.semilogy(np.sqrt(CombInt.profile_variance.flatten())); plt.semilogy(np.sqrt(MolInt.profile_variance.flatten()));


#plt.figure(figsize=(15,5)); 
#plt.pcolor(Molecular.time/3600,Molecular.range_array*1e-3, np.log10(1e9*Molecular.profile.T/Molecular.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Molecular Count Rate [Hz] (BG Subtracted)')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(CombHi.time/3600,CombHi.range_array*1e-3, np.log10(1e9*CombHi.profile.T/CombHi.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Combined Count Rate [Hz] (BG Subtracted)')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)

if CombHi.time.size > Molecular.time.size:
    CombHi.slice_time_index(time_lim=np.array([0,Molecular.time.size]))
elif CombHi.time.size < Molecular.time.size:
    Molecular.slice_time_index(time_lim=np.array([0,CombHi.time.size-1]))

# mask based on raw counts - remove points where there are < 2 counts
if use_mask:
    NanMask = np.logical_or(Molecular.profile < 2.0,CombHi.profile < 2.0)
    Molecular.profile = np.ma.array(Molecular.profile,mask=NanMask)
    CombHi.profile = np.ma.array(CombHi.profile,mask=NanMask)
#    Molecular.profile[NanMask] = np.nan
#    CombHi.profile[NanMask] = np.nan



#Molecular.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_geo:
#    Molecular.geo_overlap_correct(geo_corr)
Molecular.range_correct();
Molecular.slice_range(range_lim=[0,MaxAlt])
Molecular.range_resample(delta_R=zres*dR,update=True)
Molecular.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#Molecular.conv(1.5,2.0)  # regrid by convolution
MolRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin

#CombHi.energy_normalize(TransEnergy*EnergyNormFactor)
if use_diff_geo:
    CombHi.diff_geo_overlap_correct(diff_geo_corr,geo_reference='mol')
#if use_geo:
#    CombHi.geo_overlap_correct(geo_corr)
CombHi.range_correct()
CombHi.slice_range(range_lim=[0,MaxAlt])
CombHi.range_resample(delta_R=zres*dR,update=True)
CombHi.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#CombHi.conv(1.5,2.0)  # regrid by convolution

# if running the Klett inversion, the raw data is used and we need it to be on the same range grid as the sondes
if runKlett:
    CombRaw.slice_range(range_lim=[0,MaxAlt])
    CombRaw.range_resample(delta_R=zres*dR,update=True)

CombRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin


#plt.figure(); 
#plt.plot(np.sum(CombHi.profile[3600:,:],axis=0)/np.sum(Molecular.profile[3600:,:],axis=0),'.');

if use_diff_geo:
    #MolGain = 2.00;  # GeoCorrect prior to 12/12/2016
#    MolGain = 2.25;  # Correction after 12/12/2016
#    MolGain = 1.20;  # Correction after 12/19/2016
#    MolGain = 1.0/0.397# /64.8#2.68  # Correction after 12/21/2016
#    MolGain = 1.58/1.13  # Gain used for Scott's profile from 12/27 and additional clear data in surrounding days
    MolGain = 1.33  # Gain updated 1/18/2016 based in very clear integrated retrievals between 0-16UTC
#    MolGain = 2.8789  # Gain used for Scott's profile from 3/1/2017 and additional clear data in surrounding days
else:
#    MolGain = 1.0/9*2.25*1.1*1.25  # No diff_geo
#    MolGain = 1.0/9*2.25*1.1*1.25/1.39 #*1.76  # No diff_geo After Dec. 19, 2016
#    MolGain = 1.3/9*2.25*1.1*1.25/1.39 # # No diff_geo starting Dec. 21, 2016
#    MolGain = 1.2821          # after Dec. 21, 2016 18.5 UTC - switched to 70/30 splitter
    MolGain = 1.33          # after April. 14, 2017 mode scrambler
#    MolGain = 1.0
    



Molecular.gain_scale(MolGain)

# Correct Molecular Cross Talk
if Cam > 0:
    lp.FilterCrossTalkCorrect(Molecular,CombHi,Cam,smart=True)
#    Molecular.profile = 1.0/(1-Cam)*(Molecular.profile-CombHi.profile*Cam);

lp.plotprofiles([CombHi,Molecular])


### Grab Sonde Data  -- This segment is depricated but temp and pressure data are still used in outdated functions (e.g. extinction)
sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
#sonde_index = 2*Days[-1]
sonde_index = 22
#(Man or SigT)
f = netcdf.netcdf_file(sondefilename, 'r')
TempDat = f.variables['tpSigT'].data.copy()  # Kelvin
PresDat = f.variables['prSigT'].data.copy()*100.0  # hPa - convert to Pa (or Man or SigT)
SondeTime = f.variables['relTime'].data.copy() # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
SondeAlt = f.variables['htSigT'].data.copy()  # geopotential altitude in m
StatElev = f.variables['staElev'].data.copy()  # launch elevation in m
f.close()

TempDat[np.nonzero(np.logical_or(TempDat < 173.0, TempDat > 373.0))] = np.nan;
PresDat[np.nonzero(np.logical_or(PresDat < 1.0*100, PresDat > 1500.0*100))] = np.nan;

sonde_index = np.min([np.shape(SondeAlt)[0]-1,sonde_index])
# Obtain sonde data for backscatter coefficient estimation
Tsonde = np.interp(CombHi.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
Psonde = np.interp(CombHi.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],PresDat[sonde_index,:])



# note the operating wavelength of the lidar is 532 nm
beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*lp.kB)


beta_mol_sonde,sonde_time,sonde_index_prof = lp.get_beta_m_sonde(Molecular,Years,Months,Days,sonde_path,interp=True)
#beta_mol_sonde.gain_scale(sonde_scale)

#plt.figure(); plt.semilogx(beta_m_sonde/np.nanmean(Molecular.profile,axis=0),Molecular.range_array)
if sonde_scale == 1.0:
    Mol_Beta_Scale = 1.36*0.925e-6*2.49e-11*Molecular.mean_dt/(Molecular.time[-1]-Molecular.time[0])  # conversion from profile counts to backscatter cross section
else:
    Mol_Beta_Scale = 1.0/sonde_scale    

BSR = (CombHi.profile)/Molecular.profile

#beta_bs = BSR*beta_m_sonde[np.newaxis,:]  # total backscatter including molecules
beta_bs = BSR*beta_mol_sonde.profile

#aer_beta_bs = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
#aer_beta_bs[np.nonzero(aer_beta_bs <= 0)] = 1e-10;


## Depricated aerosol backscatter retrieval.  Better if there are no sondes to use as reference.
#aer_beta_dlb = lp.Calc_AerosolBackscatter(Molecular,CombHi,Temp=Tsonde,Pres=Psonde)

# Latest aerosol backscatter retrival using 2D sonde profiles
aer_beta_dlb = lp.AerosolBackscatter(Molecular,CombHi,beta_mol_sonde)


#### Dynamic Integration ####
## Dynamically integrate in layers to obtain lower resolution only in areas that need it for better
## SNR.  Returned values are
## dynamically integrated molecular profile, dynamically integrated combined profile, dynamically integrated aerosol profile, resolution in time, resolution in altitude
#MolLayer,CombLayer,aer_layer_dlb,layer_t,layer_z = lp.AerBackscatter_DynamicIntegration(Molecular,CombHi,Temp=Tsonde,Pres=Psonde,num=3,snr_th=1.2,sigma = np.array([1.5,1.0]))

MolInt = Molecular.copy();
MolInt.time_integrate();
CombInt = CombHi.copy();
CombInt.time_integrate();
sonde_int = beta_mol_sonde.copy()
sonde_int.time_integrate();
#aer_beta_dlb_int = lp.Calc_AerosolBackscatter(MolInt,CombInt,Tsonde,Psonde)
aer_beta_dlb_int = lp.AerosolBackscatter(MolInt,CombInt,sonde_int)
lp.plotprofiles([aer_beta_dlb_int])


if use_geo:
    Molecular.geo_overlap_correct(geo_corr)
    CombHi.geo_overlap_correct(geo_corr)

# Obtain low pass filtered instances of the profiles for extinction and 
# MLE initialization
MolLP = Molecular.copy()
MolLP.conv(4,2)
CombLP = CombHi.copy()
CombLP.conv(4,2)
aer_beta_LP = lp.AerosolBackscatter(MolLP,CombLP,beta_mol_sonde)

Extinction,OptDepth,ODmol = lp.Calc_Extinction(MolLP, MolConvFactor=Mol_Beta_Scale, Temp=Tsonde, Pres=Psonde, AerProf=aer_beta_dlb)

## Atmospheric transmission calculation
#Transmission = Molecular.profile.copy()/beta_m_sonde[np.newaxis,:]*0.7729729

############## Eventually fold into LidarProfileFunctions
#aerfit = aer_beta_dlb.profile
##aerfit[np.nonzero(aerfit<0)] = 0
##aerfit[np.nonzero(np.isnan(aerfit))] = np.nan
#aerfit_std = np.sqrt(aer_beta_dlb.profile_variance)
#ODfit = OptDepth.profile-ODmol[np.newaxis,:]
#ODvar = OptDepth.profile_variance
#
#
#aerfit[np.nonzero(np.isnan(aerfit))] = 0
#x_invalid = np.nonzero(aerfit < 3.3*aerfit_std)
#
#sLROpt,extinctionOpt,ODOpt,ODbiasOpt = lp.Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=30,maxLR=1e5,minLR=aer_beta_dlb.mean_dR,lam=np.array([10.0,3.0]),grad_gain=1e-1,max_iterations=200,optout=-1)
if getMLE_extinction:
    ExtinctionMLE,OptDepthMLE,LidarRatioMLE,ODbiasProf,xMLE_fit = lp.Retrieve_Ext_MLE(OptDepth,aer_beta_dlb,ODmol,overlap=3,lam=np.array([10.0,3.0]),max_iterations=10000)

if run_MLE and use_geo:
#    beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb,fit_mol_mle,fit_comb_mle = \
#            mle.MLE_Estimate_OptProp(MolRaw,CombRaw,aer_beta_LP,geo_data,PresDat[sonde_index,:],TempDat[sonde_index,:],SondeAlt[sonde_index,:]-StatElev[sonde_index], \
#            minSNR=1.0,dG=0.04,fitfilt=True)
    beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb,fit_mol_mle,fit_comb_mle = \
            mle.MLE_Estimate_OptProp(MolRaw,CombRaw,aer_beta_LP,geo_data,sonde_path, \
            minSNR=1.0,dG=0.04,fitfilt=True)
    # merge mle backscatter data with direct retrieval
    beta_a_merge = aer_beta_dlb.copy()
    iMLE = np.nonzero(xvalid_mle.profile)
    beta_a_merge.profile[iMLE] = beta_a_mle.profile[iMLE]
    # trim fit profile ranges so they mesh with current processed ranges
    fit_mol_mle.slice_range_index(range_lim=[0,aer_beta_dlb.profile.shape[1]])
    fit_comb_mle.slice_range_index(range_lim=[0,aer_beta_dlb.profile.shape[1]])
    
#    if use_mask:
#        NanMask = np.logical_or(Molecular.profile < 4.0,CombHi.profile < 4.0)
#        beta_a_merge = np.ma.array(beta_merge,mask=NanMask)

#plt.figure(); 
#plt.semilogx(aer_beta_dlb_int.range_array,aer_beta_dlb)

#OptDepth = np.log(CombHi.profile/beta_bs)*0.5
#
#extinction = -np.diff(np.log(Molecular.profile/beta_m_sonde[np.newaxis,:]),axis=1)*0.5
#
#extinction2 = -np.diff(np.log(CombHi.profile/beta_bs),axis=1)*0.5

### Run Klett Inversion for comparision
#,geo_corr=np.array([])
if runKlett:
    aer_beta_klett = lp.Klett_Inv(CombRaw,KlettAlt,Temp=Tsonde,Pres=Psonde,avgRef=False,BGIndex=BGIndex,geo_corr=geo_corr,Nmean=40,kLR=1.05)
#    diff_aer_beta = np.ma.array(aer_beta_dlb.profile-aer_beta_klett.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))
    diff_aer_beta = np.ma.array(aer_beta_klett.profile/aer_beta_dlb.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))

#if use_mask:
##    CountMask = np.logical_or(Molecular.SNR() < countLim,CombHi.SNR() < countLim)
##    CountMask = np.logical_or(CountMask,np.isnan(aer_beta_dlb.profile))
#    CountMask = np.isnan(aer_beta_dlb.profile)
##    CountMask = np.logical_or(CountMask, aer_beta_dlb.SNR() < SNRmask)
#    aer_beta_dlb.profile = np.ma.array(aer_beta_dlb.profile, mask=CountMask)
##    aer_beta_dlb.profile[np.nonzero(aer_beta_dlb.SNR() < SNRmask)] = np.nan
##    Extinction.profile[np.nonzero(Extinction.SNR() < SNRmask)] = np.nan

#ExtMask = Extinction.SNR() < 1.0
#ExtMask = np.logical_or(ExtMask,aer_beta_dlb.profile < 1e-6)
#Extinction.profile = np.ma.array(Extinction.profile,mask=ExtMask)

#ncfilename = '/h/eol/mhayman/write_py_netcdf3.nc'
if save_as_nc:
    CombHi.write2nc(ncfilename)
    Molecular.write2nc(ncfilename)
    aer_beta_dlb.write2nc(ncfilename)
    beta_mol_sonde.write2nc(ncfilename)
    if run_MLE and use_geo:
#        CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb
        beta_a_mle.write2nc(ncfilename)
        alpha_a_mle.write2nc(ncfilename)
        sLR_mle.write2nc(ncfilename)
        xvalid_mle.write2nc(ncfilename)
        fit_mol_mle.write2nc(ncfilename)
        fit_comb_mle.write2nc(ncfilename)

if plotAsDays:
    time_scale = 3600*24.0
else:
    time_scale = 3600.0

lp.pcolor_profiles([Molecular,CombHi],climits=[[8,12],[8,12]],plotAsDays=plotAsDays)

# set y limits to nearest half km in profile range span
ylimits = 2.0*np.array([np.round(0.5e-3*aer_beta_dlb.range_array[0]),np.round(0.5e-3*aer_beta_dlb.range_array[-1])])
# scale figure dimensions based on time and altitude dimensions
time_span = (aer_beta_dlb.time[-1]-aer_beta_dlb.time[0])/3600  # time domain of plotted data
range_span = ylimits[1]-ylimits[0]  # range domain of plotted data
#range_span = (aer_beta_dlb.range_array[-1]-aer_beta_dlb.range_array[0])*1e-3

# adjust title line based on the amount of plotted time data
if time_span < 8.0:
    # short plots (in time)
    line_char = '\n'  # include a newline to fit full title
    y_top_edge = 1.2  # top edge set for double line title
    title_font_size = 12  # use larger title font
elif time_span <= 16.0:
    # medium plots (in time)
    line_char = ' '  # no newline in title
    y_top_edge = 0.9  # top edge set for single line title
    title_font_size = 12  # use smaller title font
else:
    # long plots (in time)
    line_char = ' '  # no newline in title
    y_top_edge = 0.9  # top edge set for single line title
    title_font_size = 16  # use larger title font

max_len = 18.0
min_len = 2.0
max_h = 8.0
min_h = 0.2
x_left_edge =1.0
x_right_edge = 2.0
y_bottom_edge = 0.6
#y_top_edge = 1.2 # 0.9 for single line title, 1.8 for double line title

ax_len = np.max(np.array([np.min(np.array([max_len,time_span*18.0/24.0])),min_len])) # axes length
ax_h = np.max(np.array([np.min(np.array([max_h,range_span*2.1/12])),min_h]))  # axes height
fig_len = x_left_edge+x_right_edge+ax_len  # figure length
fig_h =y_bottom_edge+y_top_edge+ax_h  # figure height

# axes sizes for time/range resolved profiles
axlims = [x_left_edge/fig_len,y_bottom_edge/fig_h,1-x_right_edge/fig_len,1-y_top_edge/fig_h]

# axes sizing for vertically stacked plots
axlim2 = [[x_left_edge/fig_len,y_bottom_edge/fig_h/2.0+0.5,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/2.0], \
    [x_left_edge/fig_len,y_bottom_edge/fig_h/2.0,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/2.0]]


#plt.figure(figsize=(15,5)); # plt.figure(figsize=(15,5))
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
    
#fig = plt.figure(figsize=(20,3)); # plt.figure(figsize=(15,5))
#ax = plt.axes([0.05, 0.2, 0.9, 0.7])  # [xstart,ystart,xwid,ywid] - as fractions of the window size
fig = plt.figure(figsize=(fig_len,fig_h)); # plt.figure(figsize=(15,5))
ax = plt.axes(axlims)  # [xstart,ystart,xwid,ywid] - as fractions of the window size
im = plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
plt.ylim(ylimits)
plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
plt.title(DateLabel + ', ' + aer_beta_dlb.lidar+ line_char + 'Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]',fontsize=title_font_size)
plt.ylabel('Altitude AGL [km]')
if plotAsDays:
    plt.xlabel('Days [UTC]')
    plt.xlim(HourLim/24.0)
else:
    plt.xlabel('Time [UTC]')
    plt.xlim(HourLim)
divider = make_axes_locatable(ax)
#cax = divider.append_axes("right",size="1%",pad=0.2)
cax = divider.append_axes("right",size=0.1,pad=0.2)
plt.colorbar(im,cax=cax)

if save_figs:
    plt.savefig(figfilename+'_AerosolBackscatter.png')

if runKlett:
#    plt.figure(figsize=(15,5)); 
#    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(aer_beta_klett.profile.T));
#    plt.colorbar()
#    plt.clim([-9,-3])
#    plt.title(DateLabel + ', ' + 'Klett Estimated' + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#    plt.ylabel('Altitude [km]')
#    if plotAsDays:
#        plt.xlabel('Days [UTC]')
#        plt.xlim(HourLim/24.0)
#    else:
#        plt.xlabel('Time [UTC]')
#        plt.xlim(HourLim)
        
    fig = plt.figure(figsize=(20,3)); # plt.figure(figsize=(15,5))
    ax = plt.axes([0.05, 0.2, 0.9, 0.7])
    im = plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(aer_beta_klett.profile.T));
    plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
    plt.title(DateLabel + ', ' + 'Klett Estimated' + 'Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    #plt.ylim([0,10])
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="1%",pad=0.2)
    plt.colorbar(im,cax=cax)
        
    plt.figure(figsize=(15,5)); 
    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(diff_aer_beta.T));
    plt.colorbar()
    plt.clim([-1,1])
    plt.title(DateLabel + ', ' + 'Logarithmic Difference in' + ' Aerosol Backscatter Coefficient')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)


#plt.figure(figsize=(15,5)); 
#plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile.T));
#plt.colorbar()
#plt.clim([8,12])  # plt.clim([9,12])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Attenuated Backscatter [Counts]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
    
fig = plt.figure(figsize=(fig_len,fig_h)); # plt.figure(figsize=(15,5))
ax = plt.axes(axlims)  # [xstart,ystart,xwid,ywid] - as fractions of the window size
im = plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile.T));
plt.clim([8,12])
plt.title(DateLabel + ', ' + CombHi.lidar +line_char+ 'Attenuated Backscatter [Counts]',fontsize=title_font_size)
plt.ylabel('Altitude AGL [km]')
if plotAsDays:
    plt.xlabel('Days [UTC]')
    plt.xlim(HourLim/24.0)
else:
    plt.xlabel('Time [UTC]')
    plt.xlim(HourLim)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size=0.1,pad=0.2)
plt.colorbar(im,cax=cax)
if save_figs:
    plt.savefig(figfilename+'_CombHi.png')

if getMLE_extinction:
    plt.figure(figsize=(15,5)); 
    plt.pcolor(ExtinctionMLE.time/time_scale,ExtinctionMLE.range_array*1e-3, np.log10(ExtinctionMLE.profile.T));
    plt.colorbar()
    plt.clim([-7,-3])
    plt.title(DateLabel + ', ' + ExtinctionMLE.lidar + line_char+'Aerosol Extinction Coefficient [$m^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
        
    plt.figure(figsize=(15,5)); 
    plt.pcolor(LidarRatioMLE.time/time_scale,LidarRatioMLE.range_array*1e-3, (xMLE_fit*LidarRatioMLE.profile).T);
    plt.colorbar()
    plt.clim([0,100])
    plt.title(DateLabel + ', ' + ExtinctionMLE.lidar + line_char+'Lidar Ratio [$sr$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)

#plt.figure(figsize=(15,5)); 
#plt.pcolor(Extinction.time/3600,Extinction.range_array*1e-3, np.log10(Extinction.profile.T));
#plt.colorbar()
#plt.clim([-4,-2])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Extinction Coefficient [$m^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)

#plt.figure(); 
#plt.pcolor(OptDepth.time/3600,OptDepth.range_array*1e-3, np.log10(OptDepth.profile.T));
#plt.colorbar()
#plt.clim([0,3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Optical Depth')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)


#plt.figure(figsize=(15,5)); 
#plt.pcolor(Extinction.time/3600,Extinction.range_array*1e-3, np.log10(Extinction.profile/aer_beta_dlb.profile).T);
#plt.colorbar()
##plt.clim([-4,-2])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Lidar Ratio [$sr$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)


#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,2)
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-8.5,-4])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#    
#plt.subplot(2,1,1)
#plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile).T);  # /(CombHi.binwidth_ns*1e-9*tres*7e3)
#plt.colorbar()
#plt.clim([9,12])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Attenuated Backscatter [Counts]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)



fig = plt.figure(figsize=(fig_len,2*fig_h)); # plt.figure(figsize=(15,5))
ax = plt.axes(axlim2[0])  # [xstart,ystart,xwid,ywid] - as fractions of the window size
im = plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
plt.ylim(ylimits)
plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
plt.title(DateLabel + ', ' + CombHi.lidar+ line_char + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]',fontsize=title_font_size)
plt.ylabel('Altitude AGL [km]')
if plotAsDays:
    plt.xlabel('Days [UTC]')
    plt.xlim(HourLim/24.0)
else:
    plt.xlabel('Time [UTC]')
    plt.xlim(HourLim)
divider = make_axes_locatable(ax)
#cax = divider.append_axes("right",size="1%",pad=0.2)
cax = divider.append_axes("right",size=0.1,pad=0.2)
plt.colorbar(im,cax=cax)

ax = plt.axes(axlim2[1]) 
im = plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile.T));
plt.ylim(ylimits)
plt.clim([8,12])
plt.title(DateLabel + ', ' + CombHi.lidar+ line_char + ' Attenuated Backscatter [Counts]',fontsize=title_font_size)
plt.ylabel('Altitude AGL [km]')
if plotAsDays:
    plt.xlabel('Days [UTC]')
    plt.xlim(HourLim/24.0)
else:
    plt.xlabel('Time [UTC]')
    plt.xlim(HourLim)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right",size=0.1,pad=0.2)
plt.colorbar(im,cax=cax)
if save_figs:
    plt.savefig(figfilename+'_AerosolBackscatter_and_CombinedBackscatter.png')
#
#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,1)
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#
#plt.subplot(2,1,2)
#plt.pcolor(aer_layer_dlb.time/time_scale,aer_layer_dlb.range_array*1e-3, np.log10(aer_layer_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)

if getMLE_extinction:
    # initialization dependent on the aerosol backscatter
    lrplt1 = np.arange(100)
    aer01 = -75.0*lrplt1/1000.0-2
    
    # constant initialization
#    lrplt = np.array([1500,1500])/75.0
    lrplt = np.array([50,50])*75.0
    aer0 = np.array([-4,-9])
    
    plt.figure(); 
    plt.plot((LidarRatioMLE.profile*xMLE_fit).flatten(),np.log10(aer_beta_dlb.profile.flatten()),'.',label='retrieved data'); 
    plt.plot(lrplt,aer0,'g--',label='constant initialization'); 
    plt.plot(lrplt1,aer01,'r--',label='aerosol initialization'); 
    plt.xlabel('lidar ratio [sr]'); 
    plt.ylabel(r'$log_{10}\beta_a$ [$m^{-1}sr^{-1}$]'); 
    plt.grid(b=True); 
    plt.legend();

if runKlett:
    plt.figure(figsize=(15,10)); 
    plt.subplot(3,1,1)
    plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
    plt.colorbar()
    plt.clim([-9,-3])
    plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
#        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
#        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])
    
    plt.subplot(3,1,2)
    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(aer_beta_klett.profile.T));
    plt.colorbar()
    plt.clim([-9,-3])
    plt.title(DateLabel + ', ' + 'Klett Estimated' + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
#        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
#        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])

    plt.subplot(3,1,3)
    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(diff_aer_beta.T));
    plt.colorbar()
    plt.clim([-1,1])
    plt.title(DateLabel + ', ' + 'Logarithmic Difference in' + ' Aerosol Backscatter Coefficient')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])

if run_MLE and use_geo:    
    ### Plot Both MLE and Direct Retrievals ###
    fig = plt.figure(figsize=(20,6)); # plt.figure(figsize=(15,5))
    #plt.subplot(2,1,1)
    ax = plt.axes([0.05, 0.5+0.1, 0.9, (0.7)/2])
    im = plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
    plt.ylim([0,12])
    plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
    plt.title(DateLabel + ', ' + CombHi.lidar + ' Direct Calculated Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
    #    plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
    #    plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="1%",pad=0.2)
    plt.colorbar(im,cax=cax)
    #plt.subplot(2,1,2)
    ax = plt.axes([0.05, 0.1, 0.9,0.7/2])
    im = plt.pcolor(beta_a_merge.time/time_scale,beta_a_merge.range_array*1e-3, np.log10(beta_a_merge.profile.T));
    plt.ylim([0,12])
    plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
    plt.title(DateLabel + ', ' + beta_a_merge.lidar + ' MLE Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="1%",pad=0.2)
    plt.colorbar(im,cax=cax)
    if save_figs:
        plt.savefig(figfilename+'_Both_MLE_Direct_AerosolBackscatter.png')
    
    
    fig = plt.figure(figsize=(20,3)); # plt.figure(figsize=(15,5))
    ax = plt.axes([0.05, 0.2, 0.9, 0.7])
    im = plt.pcolor(beta_a_mle.time/time_scale,beta_a_mle.range_array*1e-3, np.log10(beta_a_mle.profile.T));
    plt.ylim([0,12])
    plt.clim([-8,-4]) # Aerosol Enhanced limits plt.clim([-7.5,-6])
    plt.title(DateLabel + ', ' + beta_a_merge.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="1%",pad=0.2)
    plt.colorbar(im,cax=cax)
    if save_figs:
        plt.savefig(figfilename+'_Merged_MLE_AerosolBackscatter.png')
        
#    beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb

    plt.figure()
    plt.subplot(3,1,1)
    plt.pcolor(beta_a_mle.time/time_scale,beta_a_mle.range_array*1e-3,np.log10(beta_a_mle.profile).T)
    plt.ylim([0,14])
    plt.clim([-8,-4])
    plt.colorbar()
    plt.title('Backscatter [$m^{-1} sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    plt.subplot(3,1,2)
    plt.pcolor(alpha_a_mle.time/time_scale,alpha_a_mle.range_array*1e-3,np.log10(alpha_a_mle.profile).T)
    plt.ylim([0,14])
    plt.clim([-7,-3])
    plt.colorbar()
    plt.ylabel('Altitude [km]')
    plt.title('Extinction [$m^{-1}$]')
    plt.subplot(3,1,3)
    plt.pcolor(sLR_mle.time/time_scale,sLR_mle.range_array*1e-3,sLR_mle.profile.T)
    plt.ylim([0,14])
    plt.clim([1,50])
    plt.colorbar()
    plt.title('Lidar Ratio [$sr$]')
    plt.ylabel('Altitude [km]')
    plt.xlabel('Time [h-UTC]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    
    plt.figure(); 
    plt.semilogy(beta_a_mle.time/time_scale,ProfileErrorMol)
    plt.semilogy(beta_a_mle.time/time_scale,ProfileErrorComb)
    plt.grid(b=True)
    plt.ylabel('Profile RMS Error')
    plt.legend(('Molecular','Combined'))
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    
    
    plt.figure()
    plt.plot(beta_a_mle.time/time_scale,GmList)
    plt.plot(beta_a_mle.time/time_scale,GcList)
    plt.grid(b=True)
    plt.legend(('Molecular Gain','Combined Gain'))
    plt.ylabel('$G$')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    
    plt.figure()
    plt.plot(beta_a_mle.time/time_scale,CamList)
    plt.grid(b=True)
    plt.ylabel('$C_{a}$')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)


plt.show()


#np.savez('20170121_Ext1_5h_2',range_array=LidarRatioMLE.range_array,time_array=LidarRatioMLE.time,LidarRatio=LidarRatioMLE.profile,Extinction=ExtinctionMLE.profile,aer_beta_dlb=aer_beta_dlb.profile,xMLE_fit=xMLE_fit)





#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,2)
#plt.pcolor(aer_layer_dlb.time/3600,aer_layer_dlb.range_array*1e-3, (layer_z.T+1.0)*aer_layer_dlb.mean_dR);
#plt.colorbar()
##plt.clim([-8.5,-4])
#plt.title(DateLabel + ', ' + aer_layer_dlb.lidar + ' Range Resolution [m]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.subplot(2,1,1)
#plt.pcolor(aer_layer_dlb.time/3600,aer_layer_dlb.range_array*1e-3, (layer_t.T+1.0)*aer_layer_dlb.mean_dt);  # /(CombHi.binwidth_ns*1e-9*tres*7e3)
#plt.colorbar()
##plt.clim([9,12])
#plt.title(DateLabel + ', ' + aer_layer_dlb.lidar + ' Time Resolution [seconds]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)

## Plot SNR
#plt.figure(); 
#plt.pcolor(aer_beta_dlb.time/3600,aer_beta_dlb.range_array*1e-3, np.log10((aer_beta_dlb.profile/np.sqrt(aer_beta_dlb.profile_variance)).T));
#plt.colorbar()
##plt.clim([-8,-3])
#plt.title(CombHi.lidar + ' Aerosol Backscatter Coeffient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [h-UT]')



"""
Diff Geo Correction
"""

##timecal = np.array([1.5*3600,6*3600])
#MolCal = Molecular.copy();
#CombCal = CombHi.copy();
#
##MolCal.slice_time(timecal)
##CombCal.slice_time(timecal)
#
#MolCal.time_integrate();
#CombCal.time_integrate();
##
#plt.figure();
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(np.max((MolCal.profile/CombCal.profile).T,axis=1),'k--',linewidth=2)
#plt.plot((MolCal.profile/CombCal.profile).flatten())
#
#FitProf = np.max((MolCal.profile/CombCal.profile).T,axis=1)
#FitProf2 = np.mean((MolCal.profile/CombCal.profile).T,axis=1)
##f1 = np.concatenate((np.arange(1,8),np.arange(42,61)));
#f1 = np.arange(1,61)
##f1 = np.arange(1,100)
##f1 = np.array([6,7,43,44])
#
##pfit1 = np.polyfit(f1,np.log(FitProf[f1]),10)
##pfit1 = np.polyfit(f1,np.log(FitProf[f1]),2)
#
##np.interp(np.arange(8,43),f1,FitProf[f1]);
#finterp = scipy.interpolate.interp1d(f1,FitProf[f1],kind='cubic')
#finterpL = scipy.interpolate.interp1d(f1,FitProf[f1])
#Lweight = 0.7
#
##f2 = np.concatenate((np.arange(50,100),np.arange(130,200)));
##FitOrder = 4
#f2 = np.arange(61,108)
#FitOrder = 0
#
#pfit1 = np.polyfit(f2,FitProf2[f2],FitOrder)
#
#xf = np.arange(f1[0],f1[-1])
#xf2 = np.arange(50,200)
#plt.figure()
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(FitProf,'k--',linewidth=2)
#plt.plot(xf,(1-Lweight)*finterp(xf)+Lweight*finterpL(xf),'k-',linewidth=2)
#plt.plot(xf2,np.polyval(pfit1,xf2),'k-',linewidth=2)
#
#x0 = np.arange(np.size(FitProf))
#diff_geo_prof = np.zeros(np.size(FitProf))
#diff_geo_prof[0] = 1
#diff_geo_prof[f1[0]:f1[-1]] = (1-Lweight)*finterp(x0[f1[0]:f1[-1]])+Lweight*finterpL(x0[f1[0]:f1[-1]])
#diff_geo_prof[f2[0]:f1[-1]] = 0.5*((1-Lweight)*finterp(x0[f2[0]:f1[-1]])+Lweight*finterpL(x0[f2[0]:f1[-1]])) + 0.5*(np.polyval(pfit1,x0[f2[0]:f1[-1]]))
#diff_geo_prof[f1[-1]:] = np.polyval(pfit1,x0[f1[-1]:])
#
#plt.figure();
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(diff_geo_prof,'k-',linewidth= 2)
#
#np.savez('diff_geo_DLB_20170223',diff_geo_prof=diff_geo_prof,Day=Day,Month=Month,Year=Year,HourLim=HourLim)





"""
Geo Overlap
can only be run if the profile is run at minimum z-resolution, does not have
a geo correction already imparted and is run over the full range of the 
lidar.
"""

## 12/13/2016 - 10-12 UT
#
## 12/26/2016 - 20.4-1.2 UT
#

if run_geo_cal and not use_geo and zres==1.0 and MaxAlt > 25:
    Mol_Beta_Scale = 1.0
    
    plt.figure(); 
    plt.semilogx(Mol_Beta_Scale*MolInt.profile.flatten(),MolInt.range_array)
    plt.semilogx(beta_m_sonde,CombHi.range_array)
    plt.grid(b=True)
    
    # add a correction for aerosol loading
    # extinction term actually includes molecular extinction as well
    calLR = 50  # assumed aerosol lidar ratio
    calLRvar = 20**2  # assumed variance in the assumed lidar ratio
    aerExt = np.exp(-2*aer_beta_dlb_int.mean_dR*np.cumsum(calLR*aer_beta_dlb_int.profile.flatten()+8*np.pi/3*beta_m_sonde))
    
    # propagate error in assumed extinction
    varExt = calLRvar*(-2*aer_beta_dlb_int.mean_dR*np.cumsum(aer_beta_dlb_int.profile.flatten())*aerExt)**2 \
        + aer_beta_dlb_int.profile_variance.flatten()*(-2*aer_beta_dlb_int.range_array[-1]*calLR*aerExt)**2
    
    
    plt.figure();
    plt.plot(beta_m_sonde/(MolInt.profile.flatten()))
    plt.plot(beta_m_sonde*aerExt/(MolInt.profile.flatten()))
    
    ## Set constant above 47th bin
    #geo_prof = np.ones(np.size(MolInt.profile))
    #geo_prof[0:47] = (beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))[np.newaxis,0:47]
    #geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis]))
    
    ## Run a linear fit above 65th bin
    geo_prof = beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten())
    xfit = np.arange(MolInt.profile[0,200:400].size)
    yfit = geo_prof[200:400]
    wfit = 1.0/np.sqrt(MolInt.profile_variance[0,200:400].flatten())
    wfit[0] = 10*np.max(wfit)
    #pfit = lp.polyfit_with_fixed_points(1,xfit,yfit, np.array([0]) ,np.array([geo_prof[200]]))
    pfit = np.polyfit(xfit,yfit,1,w=wfit)
    xprof = np.arange(MolInt.profile[0,200:].size)
    geo_prof[200:] = np.polyval(pfit,xprof)
    
    
    
    var_geo_prof =  varExt*(beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))**2 \
        +  MolInt.profile_variance.flatten()*(beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten()**2))**2
    
    
    #geo_prof = np.hstack((MolGeo.range_array[:,np.newaxis],GeoFromOD,geo_corr_var))
    geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis],var_geo_prof[:,np.newaxis]))
    
    plt.figure(); 
    plt.plot(beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten()))
    plt.plot(geo_prof[:,1])
    plt.plot(np.sqrt(var_geo_prof))
    
#    np.savez('/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170413',geo_prof=geo_prof,Day=Days,Month=Months,Year=Years,HourLim=HourLim,Hours=Hours,Mol_Beta_Scale=Mol_Beta_Scale,tres=tres,zres=zres,Nprof=Molecular.time.size)




#plt.figure()
#plt.semilogy(beta_a_mle.time/time_scale,FitMol_bg)
#plt.semilogy(beta_a_mle.time/time_scale,FitComb_bg)
#plt.grid(b=True)
#plt.ylabel('Background Counts')
#plt.legend(('Molecular Background','Combined Background'))
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)

"""
SVD Denoising
"""
#
#
#
#NpcaM = 3
#NpcaC = 18
#
#FitMol0 = MolRaw.profile
#MeanFitMol0 = np.mean(FitMol0,axis=0)
#
#[u,s,v] = np.linalg.svd((FitMol0-MeanFitMol0).T);
#
#weights = np.dot(u[:,:NpcaM].T,(FitMol0-MeanFitMol0).T).T
#
#Molsvd = np.dot(u[:,:NpcaM],weights.T).T
#
##FitComb0 = CombRaw.profile
##MeanFitComb0 = np.mean(FitComb0,axis=0)
##
##[uC,sC,vC] = np.linalg.svd((FitComb0-MeanFitComb0).T);
##
##weightsC = np.dot(u[:,:NpcaC].T,(FitComb0-MeanFitComb0).T).T
##
##Combsvd = np.dot(uC[:,:NpcaC],weightsC.T).T
#
#### Optimization ###
#
#FitMol = MolRaw.profile
#
#FitComb = CombRaw.profile
#
#stopIndex = MolRaw.range_array.size
#
#MolEst = Molsvd+MeanFitMol0[np.newaxis,:]
##CombEst = Combsvd+MeanFitComb0[np.newaxis,:]
#
#lamMol = np.array([0.1,1e-3])
##lamCom = np.array([0.01,10e-3])
#
#FitProfMol = lambda x: ProfEstimateSVD2D(x,FitMol,MolEst,lamMol)
#FitProfMolDeriv = lambda x: ProfEstimateSVD2D_prime(x,FitMol,MolEst,lamMol)
#
##bndsP = np.zeros((FitMol.size,2))
##bndsP[:,1] = np.max(FitMol)*20
#
#x0 = np.random.rand(FitMol.size)-0.5
#wMol = scipy.optimize.fmin_slsqp(FitProfMol,x0,fprime=FitProfMolDeriv,iter=300)
#
##FitProfComb = lambda x: ProfEstimateSVD2D(x,FitComb,CombEst,lamCom)
##FitProfCombDeriv = lambda x: ProfEstimateSVD2D_prime(x,FitComb,CombEst,lamCom)
##
###bndsP = np.zeros((FitComb.size,2))
###bndsP[:,1] = np.max(FitComb)*20
##
##x0 = np.random.rand(FitComb.size)-0.5
##wCom = scipy.optimize.fmin_slsqp(FitProfComb,x0,fprime=FitProfCombDeriv,iter=500)
##
##(wCom.reshape(FitComb.shape)+CombEst)
#
#
#####  Process SVD Profiles ####
#MolSVD = lp.LidarProfile(wMol.reshape(FitMol.shape)+MolEst,MolRaw.time,label='SVD Denoised Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=-Roffset/dR,lidar='DLB-HSRL')
##CombSVD = lp.LidarProfile(wCom.reshape(FitComb.shape)+CombEst,CombRaw.time,label='SVD Denoised Combined Backscatter Channel',descript = 'Unpolarization\nCombined Backscatter Returns',bin0=-Roffset/dR,lidar='DLB-HSRL')
#CombSVD = CombRaw.copy()
#
#MolSVD.nonlinear_correct(38e-9);
#MolSVD.bg_subtract(BGIndex)
#
#CombSVD.nonlinear_correct(29.4e-9);
#CombSVD.bg_subtract(BGIndex)
#
#
#if CombSVD.time.size > Molecular.time.size:
#    CombSVD.slice_time_index(time_lim=np.array([0,MolSVD.time.size]))
#elif CombSVD.time.size < MolSVD.time.size:
#    MolSVD.slice_time_index(time_lim=np.array([0,CombSVD.time.size]))
#
##MolSVD.range_correct();
#MolSVD.slice_range(range_lim=[0,MaxAlt])
#MolSVD.range_resample(delta_R=zres*dR,update=True)
##Molecular.conv(2.0,1.0)  # regrid by convolution
#
##CombHi.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_diff_geo:
#    CombSVD.diff_geo_overlap_correct(diff_geo_corr,geo_reference='mol')
##if use_geo:
##    CombHi.geo_overlap_correct(geo_corr)
##CombSVD.range_correct()
#CombSVD.slice_range(range_lim=[0,MaxAlt])
#CombSVD.range_resample(delta_R=zres*dR,update=True)
##CombHi.conv(2.0,1.0)  # regrid by convolution
#
#
#MolSVD.gain_scale(MolGain)
#
## Correct Molecular Cross Talk
#if Cam > 0:
#    lp.FilterCrossTalkCorrect(MolSVD,CombSVD,Cam,smart=True)
##    Molecular.profile = 1.0/(1-Cam)*(Molecular.profile-CombHi.profile*Cam);
#
##lp.plotprofiles([CombSVD,MolSVD])
#
#
#
### note the operating wavelength of the lidar is 532 nm
##beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*kB)
#
##plt.figure(); plt.semilogx(beta_m_sonde/np.nanmean(Molecular.profile,axis=0),Molecular.range_array)
#if sonde_scale == 1.0:
#    Mol_Beta_Scale = 1.36*0.925e-6*2.49e-11*MolSVD.mean_dt/(MolSVD.time[-1]-MolSVD.time[0])  # conversion from profile counts to backscatter cross section
#else:
#    Mol_Beta_Scale = 1.0/sonde_scale    
#
#BSR = (CombSVD.profile)/MolSVD.profile
#
#beta_bs = BSR*beta_m_sonde[np.newaxis,:]  # total backscatter including molecules
##aer_beta_bs = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
##aer_beta_bs[np.nonzero(aer_beta_bs <= 0)] = 1e-10;
#
#aer_beta_dlb_svd = lp.Calc_AerosolBackscatter(MolSVD,CombSVD,Temp=Tsonde,Pres=Psonde,beta_sonde_scale=sonde_scale)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(aer_beta_dlb_svd.time/time_scale,aer_beta_dlb_svd.range_array*1e-3, np.log10(aer_beta_dlb_svd.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + aer_beta_dlb_svd.lidar + ' Aerosol Backscatter Coefficient (SVD) [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#    
#plt.show()
#
