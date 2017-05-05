# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:45:55 2016

@author: mhayman
"""
import numpy as np
import scipy.special.lambertw
from scipy.special import wofz
import matplotlib.pyplot as plt
import scipy.signal
import datetime

from mpl_toolkits.axes_grid1 import make_axes_locatable

import os.path

#from scipy.io import netcdf
import netCDF4 as nc4

kB = 1.3806504e-23;     # Boltzman Constant
Mair = 28.95*1.66053886e-27;  # Average mass of molecular air
N_A = 6.0221413e23; #Avagadro's number, mol^-1
mH2O = 18.018; # mass water molecule g/mol
c = 2.99792458e8  # speed of light in a vacuum

"""
LidarProfile Class provides functionality of consolidating operations performed
on single profiles and storing all instances of those operations in a single
spot (i.e. background subtracted, etc)

LidarProfile(profile,time,label=none,descript=none)
profile - data read from raw data
time - corresponding time axis for profile
label - optional string label for profile.  This is used to help set the 
    operational parameters of the system
descript - longer string that can be used to provide a detailed profile 
    description

range can be calculed from the profile bin widths

Methods are written in order in which they are expected to be applied.
"""
class LidarProfile():
    def __init__(self,profile,time,label='none',descript='none',lidar='none',bin0=0,shot_count=np.array([]),binwidth=0,wavelength=0):
        if hasattr(profile,'binwidth_ns'):
            if profile.binwidth_ns == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.binwidth_ns = 50
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.binwidth_ns = 500  
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.binwidth_ns = 500
                    self.wavelength = 780.24e-9
                else:
                    self.binwidth_ns = 50
                    self.wavelength = 532e-9
            else:
                self.binwidth_ns = profile.binwidth_ns
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.wavelength = 780.24e-9
                else:
                    self.wavelength = 532e-9
#            self.raw_profile = profile             # unprocessed profile - how do I copy a netcdf_variable?
#            self.raw_profile.data = profile.data.copy()
#            self.raw_profile.dimensions = profile.dimensions
#            self.raw_profile.binwidth_ns = self.binwidth_ns
                    
            self.profile = profile.data
        else:
            if binwidth == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.binwidth_ns = 50
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.binwidth_ns = 500  
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.binwidth_ns = 250
                else:
                    # default to GV-HSRL
                    self.binwidth_ns = 50   
            else:
                self.binwidth_ns = binwidth*1e9
            if wavelength == 0:
                if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                    self.wavelength = 532e-9
                elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                    self.wavelength = 828e-9
                elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                    self.wavelength = 780.24e-9
                else:
                    # default to GV-HSRL
                    self.wavelength = 532e-9
            else:
                self.wavelength = wavelength
                    #profile_data,profile_dimensions=('time','bincount'),profile_binwidth_ns=0
            self.profile = profile         # highest processessed level of lidar profile - updates at each processing stage
            
##            self.raw_profile = profile             # unprocessed profile - how do I copy a netcdf_variable?
#            self.raw_profile.data = profile.copy()
#            self.raw_profile.dimensions = profile.shape
#            self.raw_profile.binwidth_ns = self.binwidth_ns
        
#        self.dimensions = self.profile    # time and altitude dimensions from netcdf
#        self.binwidth_ns = profile.binwidth_ns  # binwidth - used to calculate bin ranges
        self.time = time.copy()                        # 
        self.label = label                      # label for profile (used in error messages to identify the profile)
        self.descript = descript                # description of profile
        self.lidar = lidar                      # name of lidar corresponding to this profile
        
        self.ProcessingStatus = ['Raw Data']    # status of highest level of lidar profile - updates at each processing stage
        self.profile_variance = self.profile+1  # variance of the highest level lidar profile      
        self.mean_dt = np.mean(np.diff(time))   # average profile integration time
        self.range_array = (np.arange(np.shape(self.profile)[1])-bin0)*self.binwidth_ns*1e-9*c/2  # array containing the range bin data
        self.diff_geo_Refs = [];                # list containing the differential geo overlap reference sources (answers: differential to what?)
        self.profile_type = 'Photon Counts'     # measurement type of the profile (either 'Photon Counts' or 'Photon Arrival Rate [Hz]')
        self.bin0 = bin0                        # bin corresponding to range = 0        
        
        self.bg = np.array([])                  # profile background levels
        self.bg_var = np.array([])              # variance of background levels
        self.mean_dR = c*self.binwidth_ns*1e-9/2  # binwidth in range [m]
        
        self.NumProfList = np.ones(np.shape(self.profile)[0])    # number of raw profile accumlations in each profile
        
        if shot_count.size == 0:
            if (lidar == 'GV-HSRL' or lidar == 'gv-hsrl'):
                laser_freq = 4e3
            elif  (lidar == 'WV-DIAL' or lidar == 'wv-dial'):
                laser_freq = 7e3
            elif  (lidar == 'DLB-HSRL' or lidar == 'dlb-hsrl'):
                laser_freq = 7e3
            else:
                laser_freq = 4e3
            self.shot_count = np.concatenate((np.array([self.mean_dt]),np.diff(time)))*laser_freq
        else:
            self.shot_count = shot_count
        
        # Check for dimensions or set altitude/time dimension indices?        
    
    def set_bin0(self,bin0):
        # set the bin number corresponding to range = 0.  Float or integer are accepted.
        self.bin0 = bin0
        self.range_array = (np.arange(np.shape(self.profile)[1])-self.bin0)*self.binwidth_ns*1e-9*c/2
        self.ProcessingStatus.extend(['Reset Bin 0'])
        
    def nonlinear_correct(self,deadtime,laser_shot_count=0,std_deadtime=2e-9,override=False,productlog=False,newstats=False):
        # Apply nonlinear counting correction to data
        # Requires detector dead time
        # User can provide an array of the laser shots fired for each time bin
        #   otherwise it will estimate the count based on a 4kHz laser repition
        #   rate and the average integration bin duration.
    
        # override skips errors due to order in which corrections are applied
        if any('Background Subtracted' in s for s in self.ProcessingStatus):
            print ('Warning: Nonlinear correction is being applied AFTER background subtracting %s.'  %self.label)
            print ('   Nonlinear correction is generally applied BEFORE backsground subtracting.')
            print ('   Applying correction anyway.')
        
        if any('Nonlinear CountRate Correction' in s for s in self.ProcessingStatus) and not override:
            print ('Warning:  Attempted Nonlinear correction on %s after it has already been applied.' %self.label)
            print ('   Skipping this step.')
            
        else:
            if productlog:
                self.profile_count_rate(update=True)
                prodlog = scipy.special.lambertw(-deadtime*self.profile)
                self.profile=-prodlog/deadtime
                self.profile_variance = self.profile_variance*(prodlog/(deadtime*self.profile*(1+prodlog)))
                self.profile_to_photon_counts(update=True)
            elif newstats:
                bintime = self.binwidth_ns*1e-9
                self.profile = self.profile/self.shot_count[:,np.newaxis]
                self.profile = (1+self.profile*(bintime-self.profile*deadtime)/(bintime+self.profile*deadtime))*(2*bintime)**self.profile \
                    /(((1+(bintime-self.profile*deadtime)/(bintime+self.profile*deadtime)))**self.profile*(bintime+self.profile*deadtime)**(self.profile-1)*(bintime+self.profile*deadtime))
                second_moment = -4*self.profile*(self.profile**2-1)*bintime*deadtime+(bintime+self.profile*deadtime)**2*(2+self.profile**2+3*self.profile*np.sqrt(1-4*self.profile*deadtime*bintime/(bintime-self.profile*deadtime)**2)) \
                    /(bintime-self.profile*deadtime)**4
                self.profile_variance = second_moment-self.profile**2
                
                self.profile= self.profile*bintime*self.shot_count[:,np.newaxis]
                self.profile_variance= self.profile_variance*bintime**2*self.shot_count[:,np.newaxis]**2
                
                    
            else:
                CorrectionFactor = 1.0/(1-deadtime*self.profile_count_rate())
                CorrectionFactor[np.nonzero(CorrectionFactor < 0)] = np.nan;  # if the correction factor goes negative, just set it to 1.0 or NaN
            
                self.profile = self.profile*CorrectionFactor
                self.profile_variance = self.profile_variance*CorrectionFactor**4  # power of 4 due to count dependance in denominator of the correction factor
                self.profile_variance = std_deadtime**2*self.profile**4*(self.binwidth_ns*1e-9)**2/(self.binwidth_ns*1e-9-self.profile*deadtime)**4
            self.ProcessingStatus.extend(['Nonlinear CountRate Correction for dead time %.1f ns'%(deadtime*1e9)])
        
    def bg_subtract(self,start_index,stop_index=-1):
        # HSRL usually uses preshot data for background subtraction.  That
        # should be added to this function
        # Estimate background based on indicies passed to the method,
        # then create a background subtracted profile
        self.bg = np.nanmean(self.profile[:,start_index:stop_index],axis=1)
        # This needs to be adjusted to account for any nans in each profile (only divide by the number of non-nans)
        self.bg_var = np.nansum(self.profile_variance[:,start_index:stop_index],axis=1)/(np.shape(self.profile_variance[:,start_index:stop_index])[1])**2
        self.profile = self.profile-self.bg[:,np.newaxis]
        self.profile_variance = self.profile_variance+self.bg_var[:,np.newaxis]
        self.ProcessingStatus.extend(['Background Subtracted over [%.2f, %.2f] m'%(self.range_array[start_index],self.range_array[stop_index])])
        
        
    def energy_normalize(self,PulseEnergy,override=False):
        # Normalize each time bin to the corresponding transmitted energy
        # passed in as PulseEnergy
        
        if (any('Transmit Energy Normalized' in s for s in self.ProcessingStatus) and not override):
            print ('Warning:  Attempted Energy Normalization on %s after it has already been applied.' %self.label)
            print ('   Skipping this step.')
        else:
            # Only execute Energy Normalization if the profile has been background
            # subtracted.
            if any('Background Subtracted' in s for s in self.ProcessingStatus):
#                PulseEnergy = PulseEnergy/np.nanmean(PulseEnergy)      # normalize to averge pulse energy to preserve count-rate information
                # Averge Energy normalization needs to happen outside of the routine to maintain uniformity across all profiles and time data
                self.profile = self.profile/PulseEnergy[:,np.newaxis]       # normalize each vertical profile
                self.profile_variance = self.profile_variance/PulseEnergy[:,np.newaxis]**2
                self.ProcessingStatus.extend(['Transmit Energy Normalized'])
            else:
                print ('Error: Cannot energy normalize on profile %s.\n   Profile must be background subtracted first' %self.label)
        
    def geo_overlap_correct(self,geo_correction):
        # Apply a geometric overlap correction to the recorded profile
        if any('Geometric Overlap Correction' in s for s in self.ProcessingStatus):
            print ('Warning:  Attempted Geometric Overlap Correction on %s after it has already been applied.' %self.label)
            print ('   Applying correction anyway.')

        geo_correction[np.nonzero(np.isnan(geo_correction[:,1]))[0],1] = 0  # set value to zero if it is a nan
        #print ('Geometric Overlap Correction initiated for %s but processing code is not complete.' %self.label)
        geo_corr = np.interp(self.range_array,geo_correction[:,0],geo_correction[:,1])[np.newaxis,:]        
        if geo_correction.shape[1] > 2:
#            geo_corr_var = (0.05*geo_corr)**2
            geo_corr_var = np.interp(self.range_array,geo_correction[:,0],geo_correction[:,2])[np.newaxis,:]+(0.5*geo_corr)**2     
#            self.profile_variance = self.profile_variance*geo_corr**2+geo_corr_var*self.profile**2
        else:
            geo_corr_var = (0.5*geo_corr)**2
#            self.profile_variance = self.profile_variance*geo_corr**2
        self.profile_variance = self.profile_variance*geo_corr**2+geo_corr_var*self.profile**2
        self.profile = self.profile*geo_corr
        self.ProcessingStatus.extend(['Geometric Overlap Correction Applied'])
    
    def diff_geo_overlap_correct(self,diff_geo_correction,geo_reference = 'none'):
        # Apply a differential geometric overlap correction to this profile.
        # An optional label allows the user to define the reference channel 
        # (what this channel is being compared to).

        #print ('Differential Geometric Overlap Correction initiated for %s but processing code is not complete.' %self.label)
        self.profile = self.profile*diff_geo_correction[np.newaxis,:]
        self.profile_variance = self.profile_variance*diff_geo_correction[np.newaxis,:]**2
        self.diff_geo_Refs.extend([geo_reference])
        self.ProcessingStatus.extend(['Differential Geometric Correction'])
    def range_correct(self):
        # apply correction for 1/r^2 loss in the profile
        self.profile = self.profile*self.range_array**2
        self.profile_variance = self.profile_variance*self.range_array**4
        self.ProcessingStatus.extend(['Applied R^2 Range Correction'])
        
    def profile_count_rate(self,laser_shot_count=0,update=False):
        # Calculate the profile a photon arrival rate instead of photon counts.
        # if an array of laser shots per profile is not passed, the number of
        # shots are assumed based on the profile integration time and a 4kHz
        # laser rep rate.
        #
        # if update=True, the profile in the class is updated.  Otherwise
        # the profile count rates are returned.
    
        if self.profile_type == 'Photon Counts':
            if not type(laser_shot_count).__module__==np.__name__:
                laser_shot_count = self.shot_count
                
#            if np.size(laser_shot_count) != np.shape(self.profile)[0] or laser_shot_count <= 0:
#                if (self.lidar == 'GV-HSRL' or self.lidar == 'gv-hsrl'):
#                    laser_shot_count = 4e3*self.mean_dt
#                elif  (self.lidar == 'WV-DIAL' or self.lidar == 'wv-dial'):
#                    laser_shot_count = 7e3*2.0
#                elif  (self.lidar == 'DLB-HSRL' or self.lidar == 'dlb-hsrl'):
#                    laser_shot_count = 7e3*2.0
#                else:
#                    laser_shot_count = 4e3*self.mean_dt
#                bin_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count)
#                var_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count)**2
#            else:
#                bin_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
#                var_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2
                
            bin_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
            var_count_rate = self.profile/((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])**2
            
            if update:
                # update the processed profile, background and profile_type
                self.profile = bin_count_rate
                self.profile_variance = var_count_rate
                self.bg = self.bg/((self.binwidth_ns*1e-9)*laser_shot_count)**2
                self.bg_var/((self.binwidth_ns*1e-9)*laser_shot_count)**2
                self.profile_type = 'Photon Arrival Rate [Hz]'
            else:
                return bin_count_rate
        else:
            if not update:
                return self.profile
    def profile_to_photon_counts(self,laser_shot_count=0,update=False):
        # Calculate the photon counts corresponding to a profile currently defined in Photon Arrival Rate [Hz]        
        # if an array of laser shots per profile is not passed, the number of
        # shots are assumed based on the profile integration time and a 4kHz
        # laser rep rate.
        #
        # if update=True, the profile in the class is updated.  Otherwise
        # the profile count rates are returned.        
        
        
        if self.profile_type == 'Photon Arrival Rate [Hz]':
            if laser_shot_count == 0:
                if (self.lidar == 'GV-HSRL' or self.lidar == 'gv-hsrl'):
                    laser_shot_count = 4e3*self.mean_dt
                elif  (self.lidar == 'WV-DIAL' or self.lidar == 'wv-dial'):
                    laser_shot_count = 7e3*2.0
                elif  (self.lidar == 'DLB-HSRL' or self.lidar == 'dlb-hsrl'):
                    laser_shot_count = 7e3*2.0
                else:
                    laser_shot_count = 4e3*self.mean_dt
                
                bin_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count)
                var_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count)**2
            else:
                bin_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
                var_counts = self.profile*((self.binwidth_ns*1e-9)*laser_shot_count[:,np.newaxis])
                
            if update:
                # update the processed profile and profile_type
                self.profile = bin_counts
                self.profile_variance = var_counts
                self.profile_type = 'Photon Counts'
            else:
                return bin_counts
        else:
            if not update:
                return self.profile
        
    def time_resample(self,tedges=np.array([]),delta_t=0,i=1,t0=np.nan,update=False,remainder=False,average=True):
        # note that background data does not get adjusted in this routine
#        print ('time_resample() initiated for %s but no processing code has been written for this.' %self.label)
        if tedges.size > 0:
#            tedges = self.time[0]+np.arange(1,np.int((self.time[-1]-self.time[0])/delta_t)+1)*delta_t
#            if tedges.size > 0:            
            itime = np.digitize(self.time,tedges)
            # Only run if the profiles fit in the master timing array (tedges), otherwise everything is a remainder
            if np.sum((itime>0)*(itime<tedges.size))!=0:
                
#                iremain = np.nonzero(self.time > tedges[-1])[0]
                iremain = np.int(np.max(itime))  # the remainder starts at the maximum bin where data exists
                iremainList = np.nonzero(self.time > tedges[iremain-1])[0]
                iprofstart = np.int(np.max(np.array([1,np.min(itime)])))
#                print('start index: %d\nstop index: %d'%(iprofstart,iremain))
                
#                profNew = np.zeros((np.size(tedges)-1,self.profile.shape[1]))
#                timeNew = 0.5*tedges[1:]+0.5*tedges[:-1] 
                profNew = np.zeros((iremain-iprofstart,self.profile.shape[1]))
                timeNew = -np.diff(tedges[iprofstart-1:iremain])*0.5+tedges[iprofstart:iremain]
#                itimeNew = np.arange(iprofstart,iremain)
                var_profNew = np.zeros(profNew.shape)
                shot_countNew = np.zeros(timeNew.shape)
                self.NumProfList = np.zeros(timeNew.shape)
                          
    
#                for ai in range(0,np.size(tedges)-1):
                for ai in range(np.size(timeNew)):
                    if hasattr(self.profile,'mask') and average:
                        NumProf = np.nansum(np.logical_not(self.profile[itime == ai+iprofstart,:].mask),axis=0)
                        NumProf[np.nonzero(NumProf==0)] = 1.0
                    elif average:
                        NumProf = self.profile[itime == ai+iprofstart,:].shape[0]
                        if NumProf == 0:
                            NumProf = 1.0
                    else:
                        NumProf = 1.0
                    profNew[ai,:] = np.nansum(self.profile[itime == ai+iprofstart,:],axis=0)/NumProf
                    var_profNew[ai,:] = np.nansum(self.profile_variance[itime == ai+iprofstart,:],axis=0)/NumProf**2 
                    shot_countNew[ai] =1.0* np.nansum(self.shot_count[itime == ai+iprofstart])/NumProf
                    self.NumProfList[ai] = NumProf
                    
                if remainder:
                    RemainderProfile = self.copy();
                    RemainderProfile.profile = self.profile[iremainList,:].copy()
                    RemainderProfile.profile_variance = self.profile_variance[iremainList,:].copy()
                    RemainderProfile.time = self.time[iremainList].copy()
                    RemainderProfile.shot_count = self.shot_count[iremainList].copy()
                    
                if update:
                    self.profile = profNew.copy()
                    self.profile_variance = var_profNew.copy()
                    self.time = timeNew.copy()
                    self.mean_dt = np.mean(np.diff(timeNew))
                    self.shot_count = shot_countNew.copy()
#                    self.Nprof = NumProfList
                    self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            elif remainder:
                RemainderProfile = self.copy()
                self.profile = np.array([])
                self.profile_variance = np.array([])
                self.time = np.array([])
                self.mean_dt = np.array([])
                self.shot_count = np.array([])
                    
            if remainder:
                return RemainderProfile
        ############  Functions below this point need updated to include self.shot_count updates
        elif delta_t != 0:
            if np.isnan(t0):
                tedges = self.time[0]+np.arange(1,np.int((self.time[-1]-self.time[0])/delta_t)+1)*delta_t
            else:
                tedges = t0+np.arange(1,np.int((self.time[-1]-t0)/delta_t)+1)*delta_t
            if tedges.size > 0:            
                itime = np.digitize(self.time,tedges)
                
                iremain = np.nonzero(self.time > tedges[-1])[0]
                
                profNew = np.zeros((np.size(tedges)-1,self.profile.shape[1]))
                var_profNew = np.zeros(profNew.shape)
                timeNew = 0.5*tedges[1:]+0.5*tedges[:-1]  
                NumProfList = np.zeros(timeNew.shape)
    
                for ai in range(0,np.size(tedges)-1):
                    if hasattr(self.profile,'mask') and average:
                        NumProf = np.nansum(np.logical_not(self.profile[itime == ai,:].mask),axis=0)
                        NumProf[np.nonzero(NumProf==0)] = 1.0
                    elif average:
                        NumProf = self.profile[itime == ai,:].shape[0]
                        if NumProf == 0:
                            NumProf = 1.0
                    else:
                        NumProf = 1.0
                    profNew[ai,:] = np.nansum(self.profile[itime == ai,:],axis=0)/NumProf
                    var_profNew[ai,:] = np.nansum(self.profile_variance[itime == ai,:],axis=0)/NumProf**2    
                    NumProfList[ai] = NumProf
                    
                if remainder:
                    RemainderProfile = self.copy();
                    RemainderProfile.profile = self.profile[iremain,:]
                    RemainderProfile.profile_variance = self.profile_variance[iremain,:]
                    RemainderProfile.time = self.time[iremain]
                    
                if update:
                    self.profile = profNew
                    self.profile_variance = var_profNew
                    self.time = timeNew
                    self.mean_dt = delta_t
#                    self.Nprof = NumProfList
                    self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            elif remainder:
                RemainderProfile = self.copy()
                    
            if remainder:
                return RemainderProfile
            
#            i = np.int(np.round(delta_t/self.mean_dt))
        elif i > 1.0:
            i = np.int(i)
            profNew = np.zeros((np.floor(self.profile.shape[0]/i),self.profile.shape[1]))
            var_profNew = np.zeros(profNew.shape)
            timeNew = np.zeros(profNew.shape[0])
            
            # Calculate remainders on the end of the profile
            if remainder:
                RemainderProfile = self.copy();
                RemainderProfile.profile = self.profile[i*np.floor(self.profile.shape[0]/i):,:]
                RemainderProfile.profile_variance = self.profile_variance[i*np.floor(self.profile.shape[0]/i):,:]
                RemainderProfile.time = self.time[i*np.floor(self.profile.shape[0]/i):]
            
            
            
            for ai in range(i):
                if average:
                    NumProf = i
                else:
                    NumProf = 1
                profNew = profNew + self.profile[ai:(i*profNew.shape[0]):i,:]/NumProf
                var_profNew = var_profNew + self.profile_variance[ai:(i*profNew.shape[0]):i,:]/NumProf**2
                timeNew = timeNew + self.time[ai:(i*profNew.shape[0]):i]*1.0/i
        
#            for ai in range(profNew.shape[0]):
#                profNew[ai,:] = np.sum(self.profile[(ai*i):(i*(ai+1)),:],axis=0)
#                var_profNew[ai,:] = np.sum(self.profile_variance[(ai*i):(i*(ai+1)),:],axis=0)
            if update:
                self.profile = profNew
                self.profile_variance = var_profNew
                self.time = timeNew
                self.mean_dt = self.mean_dt*i
#                    self.Nprof = np.ones(time.shape)*i
                self.ProcessingStatus.extend(['Time Resampled to dt= %.1f s'%(self.mean_dt)])
            if remainder:
                return RemainderProfile
                
    def range_resample(self,R0=0,delta_R=0,update=False):
        # resample or integrate profile in range
        # R0 - center of first range bin
        # delta_R - range bin resolution
        # update - boolean to update the profile or return the profile
        #   True - don't return the profile, update the profile with the new range integrated one
        #   False - return the range integrated profile and don't update this one.
        print ('range_resample() initiated for %s but no processing code has been written for this.' %self.label)
#        if delta_R <= 0:
#            new_range_profile = np.nansum(self.profile,axis=1)
#            new_range_variance = np.nansum(self.profile_variance,axis=1)
#            new_range_binwidth_ns = self.binwidth_ns*np.shape(self.profile)[1]
        if delta_R > self.mean_dR:
            i = np.int(np.round(delta_R/self.mean_dR))
            if i > 1.0:
                profNew = np.zeros((self.profile.shape[0],np.floor(self.profile.shape[1]/i)))
                var_profNew = np.zeros(profNew.shape)
                rangeNew = np.zeros(profNew.shape[1])
                for ai in range(i):
                    profNew = profNew + self.profile[:,ai:(i*profNew.shape[1]):i]
                    var_profNew = var_profNew + self.profile_variance[:,ai:(i*profNew.shape[1]):i]
                    rangeNew = rangeNew + self.range_array[ai:(i*profNew.shape[1]):i]*1.0/i
            # Probably need to LP filter and resample
            
#            new_range_profile = self.profile
#            do other stuff
        
                if update:
                    self.profile = profNew
                    self.profile_variance = var_profNew
                    self.range_array = rangeNew
                    self.binwidth_ns = self.binwidth_ns*i
                    self.mean_dR = c*self.binwidth_ns*1e-9/2
#                    self.ProcessingStatus.extend(['Range Resample to dR = %.1f m'%(self.mean_dR)])
                    self.cat_ProcessingStatus('Range Resample to dR = %.1f m'%(self.mean_dR))
                else:
                    return profNew  # needs to be updated to return a LidarProfile type
            
        
        #update binwidth_ns
    def regrid_data(self,timeData,rangeData):
        print ('regrid_data() initiated for %s but no processing code has been written for this.' %self.label)
      
    def slice_time_index(self,time_lim=[0,1000]):
        if time_lim[1] >= np.size(self.time):
            time_lim[1] = np.size(self.time)
            print('Warning: requested upper time slice exceeds time dimensions of the profile:')
            print('Time dimension: %d'%np.size(self.time))
            print('Requested upper index: %d'%time_lim[1])
        if time_lim[0] >= np.size(self.time):
            time_lim[0] = np.size(self.time)
            print('Warning: requested lower time slice exceeds time dimensions of the profile:')
            print('Time dimension: %d'%np.size(self.time))
            print('Requested lower index: %d'%time_lim[1])
        keep_index = np.arange(time_lim[0],time_lim[1]+1)
        lower_remainder_index = np.arange(time_lim[0])
        upper_remainder_index = np.arange(time_lim[1]+1,np.size(self.time))
        
        lower_remainder = self.profile[lower_remainder_index,:]
        upper_remainder = self.profile[upper_remainder_index,:]

        self.profile = self.profile[keep_index,:]
        self.time = self.time[keep_index]
        self.profile_variance = self.profile_variance[keep_index,:]
        self.shot_count = self.shot_count[keep_index]
        self.NumProfList = self.NumProfList[keep_index]
        
        return lower_remainder,upper_remainder,lower_remainder_index,upper_remainder_index
        
        # Grab a slice of time in data and return the remainder(s)
        # Should be useful for processing multiple netcdf files to avoid discontinuities at file edges.
        print ('slice_time() initiated for %s but no processing code has been written for this.' %self.label)
        #return end_remainder,start_remainder      
        self.cat_ProcessingStatus(['Grab Time Slice'])
    def slice_time(self,time_range):
        itime1 = np.argmin(np.abs(time_range[0]-self.time))
        itime2 = np.argmin(np.abs(time_range[1]-self.time))
        time_slice = self.slice_time_index(time_lim=[itime1,itime2])
        self.cat_ProcessingStatus('Grab Time Slice %.1f - %.1f m'%(time_range[0],time_range[1]))
        return time_slice
    def slice_range(self,range_lim=[0,1e6]):
        # Slices profile to range_lim[start,stop] (set in m)
        # If range_lim is not supplied, negative ranges will be removed.
        
        keep_index = np.nonzero(np.logical_and(self.range_array >= range_lim[0], self.range_array <= range_lim[1]))[0]
        lower_remainder_index = np.nonzero(self.range_array < range_lim[0])
        upper_remainder_index = np.nonzero(self.range_array > range_lim[1])
        
        lower_remainder = self.profile[:,lower_remainder_index]
        range_lower_remainder = self.range_array[lower_remainder_index]
        upper_remainder = self.profile[:,upper_remainder_index]        
        range_upper_remainder = self.range_array[upper_remainder_index]        
        
        self.profile = self.profile[:,keep_index]
        self.profile_variance = self.profile_variance[:,keep_index]
        self.range_array = self.range_array[keep_index]
        
        # returned profiles should still by LidarProfile type - needs update
        return lower_remainder,upper_remainder,range_lower_remainder,range_upper_remainder
            
        self.cat_ProcessingStatus('Grab Range Slice %.1f - %.1f m'%(range_lim[0],range_lim[1]))
    def slice_range_index(self,range_lim=[0,1e6]):
        # Slices profile to range_lim[start,stop] (set in m)
        # If range_lim is not supplied, negative ranges will be removed.
        
        if range_lim[1] > self.range_array.size:
            range_lim[1] = self.range_array.size
        range_indicies = np.arange(self.range_array.size)
        keep_index = range_indicies[range_lim[0]:range_lim[1]]
#        lower_remainder_index = np.nonzero(self.range_array < range_lim[0])
#        upper_remainder_index = np.nonzero(self.range_array > range_lim[1])
#        
#        lower_remainder = self.profile[:,lower_remainder_index]
#        range_lower_remainder = self.range_array[lower_remainder_index]
#        upper_remainder = self.profile[:,upper_remainder_index]        
#        range_upper_remainder = self.range_array[upper_remainder_index]        
        
        self.profile = self.profile[:,keep_index]
        self.profile_variance = self.profile_variance[:,keep_index]
        self.range_array = self.range_array[keep_index]
            
        self.cat_ProcessingStatus('Grab Range Index Slice %.1f - %.1f indices'%(range_lim[0],range_lim[1]))    
    def copy(self,range_index=[0,0],time_index=[0,0],label='none',descript='none'):
        # needed to copy the profile and perform alternate manipulations
        # Code needs significant work!  Not sure how to best do this.
#        print ('copy() initiated for %s but no processing code has been written for this.' %self.label)
        if label == 'none':
            label = self.label
        if descript == 'none':
            descript = self.descript
        
        tmp_raw_profile = self.profile.copy()
        tmp_time = self.time.copy()
        tmp_range= self.range_array.copy()
        
        # Slice range according to requested indices range_index
        if range_index!= [0,0]:
            # check for legitimate indices before slicing
            if range_index[0] < -np.shape(tmp_raw_profile)[1]:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[0] = 0;
            if range_index[0] > np.shape(tmp_raw_profile)[1]-1:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[0] = np.shape(tmp_raw_profile)[1]-1;
            if range_index[1] < -np.shape(tmp_raw_profile)[1]:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[1] = 0;
            if range_index[1] > np.shape(tmp_raw_profile)[1]-1:
                print ('Warning: range_index out of bounds on LidarProfile.copy()')
                range_index[1] = np.shape(tmp_raw_profile)[1]-1;
            
            # slice in range
            tmp_raw_profile = tmp_raw_profile[:,range_index[0]:range_index[1]]
            tmp_range = tmp_range[range_index[0]:range_index[1]]
            if np.size(tmp_range) == 0:
                print ('Warning: range_index on LidarProfile.copy() produces an empty array')
        
        # Slice rime according to requested indices time_index  
        if time_index != [0,0]:
            # check to make sure the indices are in the array bounds
            if time_index[0] < -np.shape(tmp_raw_profile)[0]:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[0] = 0;
            if time_index[0] > np.shape(tmp_raw_profile)[0]-1:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[0] = np.shape(tmp_raw_profile)[0]-1;
            if time_index[1] < -np.shape(tmp_raw_profile)[0]:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[1] = 0;
            if time_index[1] > np.shape(tmp_raw_profile)[0]-1:
                print ('Warning: time_index out of bounds on LidarProfile.copy()')
                time_index[1] = np.shape(tmp_raw_profile)[0]-1;
            
            # slice time
            tmp_raw_profile = tmp_raw_profile[time_index[0]:time_index[1],:]
            tmp_time = tmp_time[time_index[0]:time_index[1]]
            if np.size(tmp_time) == 0:
                print ('Warning: time_index on LidarProfile.copy() produces an empty array')
        
        # Create the new profile
        NewProfile = LidarProfile(tmp_raw_profile,tmp_time,label=label,descript=descript)
        
        # Copy over everything else that was not transfered in initialization
        NewProfile.profile = self.profile.copy()        # highest processessed level of lidar profile - updates at each processing stage
        NewProfile.ProcessingStatus = list(self.ProcessingStatus)     # status of highest level of lidar profile - updates at each processing stage
        NewProfile.profile_variance = self.profile_variance.copy()   # variance of the highest level lidar profile      
        NewProfile.mean_dt = self.mean_dt                       # average profile integration time
        NewProfile.range_array = tmp_range.copy()                     # array containing the range bin data
        NewProfile.diff_geo_Refs = self.diff_geo_Refs           # list containing the differential geo overlap reference sources (answers: differential to what?)
        NewProfile.profile_type =  self.profile_type            # measurement type of the profile (either 'Photon Counts' or 'Photon Arrival Rate [Hz]')
        NewProfile.bin0 = self.bin0                             # bin corresponding to range = 0        
        NewProfile.lidar = self.lidar
        NewProfile.wavelength = self.wavelength
        
        NewProfile.bg = self.bg                 # profile background levels
        NewProfile.bg_var = self.bg_var
        NewProfile.mean_dR = self.mean_dR       # binwidth in range [m]
        
        NewProfile.time = self.time.copy()
        NewProfile.binwidth_ns = self.binwidth_ns
        NewProfile.NumProfList = self.NumProfList
        NewProfile.shot_count = self.shot_count
        
        return NewProfile
        
        
    def cat_time(self,NewProfile,front=True):
        # concatenate the added profile to the end of this profile and store it
#        print ('cat_time() initiated for %s but no processing code has been written for this.' %self.label)
        self.ProcessingStatus.extend(['Concatenate Time Data'])        
        
        ### Add checks for consistency - e.g. lidar type, photon counts vs arrival rate, etc
        if NewProfile.profile.size != 0:
            self.ProcessingStatus.extend(['Concatenate Time Data']) 
            if front:
                # Concatenate NewProfile in the front
                self.time = np.concatenate((NewProfile.time,self.time))
                self.profile = np.vstack((NewProfile.profile,self.profile))
                self.profile_variance = np.vstack((NewProfile.profile_variance,self.profile_variance))
                self.bg = np.concatenate((NewProfile.bg,self.bg))
                self.bg_var = np.concatenate((NewProfile.bg_var,self.bg_var))
                self.shot_count = np.concatenate((NewProfile.shot_count,self.shot_count))
                self.NumProfList = np.concatenate((NewProfile.NumProfList,self.NumProfList))
            else:
                # Concatenate NewProfile in the back
                self.time = np.concatenate((self.time,NewProfile.time))
                self.profile = np.vstack((self.profile,NewProfile.profile))
                self.profile_variance = np.vstack((self.profile_variance,NewProfile.profile_variance))
                self.bg = np.concatenate((self.bg,NewProfile.bg))
                self.bg_var = np.concatenate((self.bg_var,NewProfile.bg_var))
                self.shot_count = np.concatenate((self.shot_count,NewProfile.shot_count))
                self.NumProfList = np.concatenate((self.NumProfList,NewProfile.NumProfList))
        else:
            self.ProcessingStatus.extend(['Concatenate Time Data Skipped (Empty Profile Supplied)']) 
        
    def cat_range(self,NewProfile):
        # concatenate the added profile to the top of this profile and store it
        print ('cat_range() initiated for %s but no processing code has been written for this.' %self.label)
        self.ProcessingStatus.extend(['Concatenate Range Data']) 
        
    def cat_ProcessingStatus(self,ProcessingUpdate):
        self.ProcessingStatus.extend([ProcessingUpdate]) 
    
    def time_integrate(self,avg=False):
        if avg:
            num = np.shape(self.profile)[0]-np.sum(np.isnan(self.profile),axis=0)
            self.profile = np.nanmean(self.profile,axis=0)[np.newaxis,:]
            self.profile_variance = np.nansum(self.profile_variance,axis=0)[np.newaxis,:]/(num[np.newaxis,:]**2)
            self.bg = np.array([np.nanmean(self.bg)])
            self.shot_count = np.array([np.nanmean(self.shot_count)])
        else:
            self.profile = np.nansum(self.profile,axis=0)[np.newaxis,:]
            self.profile_variance = np.nansum(self.profile_variance,axis=0)[np.newaxis,:]
            self.bg = np.array([np.nansum(self.bg)])
            self.shot_count = np.array([np.nansum(self.shot_count)])
        self.mean_dt = self.mean_dt*self.time.size
        self.time = np.array([np.nanmean(self.time)])
        ## needs update to variance terms
        self.ProcessingStatus.extend(['Integrated In Time'])
    
    def gain_scale(self,gain):
        self.profile = self.profile*gain
        self.profile_variance = self.profile_variance*gain**2
        self.bg = self.bg*gain
        self.ProcessingStatus.extend(['Profile Rescaled by %f'%gain])
    
    def get_conv_kernel(self,sigt,sigz):
        t = np.arange(-np.round(4*sigt),np.round(4*sigt))      
        z = np.arange(-np.round(4*sigz),np.round(4*sigz))  
        zz,tt = np.meshgrid(z,t)
        
        kconv = np.exp(-tt**2*1.0/(sigt**2)-zz**2*1.0/(sigz**2))
        kconv = kconv/(1.0*np.sum(kconv))
        
        return zz,tt,kconv
        
    def conv(self,sigt,sigz):
        """
        Convolve a Gaussian with std sigt in the time dimension (in points)
        and sigz in the altitude dimension (also in points)
        """
        zz,tt,kconv = self.get_conv_kernel(sigt,sigz)
        
        # Replaced code with get_conv_kernel
#        t = np.arange(-np.round(4*sigt),np.round(4*sigt))      
#        z = np.arange(-np.round(4*sigz),np.round(4*sigz))  
#        zz,tt = np.meshgrid(z,t)
#        
#        kconv = np.exp(-tt**2*1.0/(sigt**2)-zz**2*1.0/(sigz**2))
#        kconv = kconv/(1.0*np.sum(kconv))
        
        if hasattr(self.profile,'mask'):
            self.profile = scipy.signal.convolve2d(self.profile.filled(0),kconv,mode='same')
        else:
            self.profile = scipy.signal.convolve2d(self.profile,kconv,mode='same')
        
        if hasattr(self.profile_variance,'mask'):
            self.profile_variance = scipy.signal.convolve2d(self.profile_variance.filled(0),kconv**2,mode='same')
        else:
            self.profile_variance = scipy.signal.convolve2d(self.profile_variance,kconv**2,mode='same')
        
        
        self.ProcessingStatus.extend(['Convolved Profile with Gaussian, sigma_t = %f, sigma_z = %f'%(sigt,sigz)])
        
    def SNR(self):
        """
        Calculate and return the SNR of the profile
        """
        return self.profile/np.sqrt(self.profile_variance)
    
    def p_thin(self,n=2):
        """
        Use Poisson thinning to creat n statistically independent copies of the
        profile.  This should only be used for photon count profiles where the
        counts are a poisson random number 
        (before background subtraction and overlap correction)
        The copied profiles are returned in a list
        """
        
        if any('Background Subtracted' in s for s in self.ProcessingStatus):
            print('Warning:  poisson thinning (self.p_thin) called on %s \n   %s has been background subtracted so it is \n   not strictly a poisson random number \n   applying anyway.' %(self.label,self.label))
        
        proflist = []
        p = 1.0/n
        for ai in range(n):
            copy = self.copy()
            copy = np.random.binomial(self.profile,p)
            copy.profile_variance = copy.profile_variance*p
            copy.label = copy.label + ' copy %d'%ai
            proflist.extend([copy])
            
        return proflist
        
    def divide_prof(self,denom_profile):
        """
        Divides the current profile by another lidar profile (denom_profile)
        Propagates the profile error from the operation.
        """
        SNRnum = self.SNR()
        SNRden = denom_profile.SNR()
        self.profile = self.profile/denom_profile.profile
        self.profile_variance = self.profile**2*(1/SNRnum**2+1/SNRden**2)
        
    def multiply_prof(self,profile2):
        """
        multiplies the current profile by another lidar profile (profile2)
        Propagates the profile error from the operation.
        """
        self.profile = self.profile*profile2.profile
        self.profile_variance = self.profile**2*profile2.profile_variance+self.profile_variance*profile2.profile**2
        
    def write2nc(self,ncfilename,tag='',name_override=False):
        """
        Writes the current profile out to a netcdf file named ncfilename
        adds the string tag the variable name if name_override==False
        if name_override=True, it names the variable according to the string
        tag.
        """
#        ncfilename = '/h/eol/mhayman/write_py_nc.nc'
#        fnc = netcdf.netcdf_file(ncfilename, 'w')
        nc_error = False
        if os.path.isfile(ncfilename):
            # if the file already exists set to modify it
            fnc = nc4.Dataset(ncfilename,'r+') #'w' stands for write, format='NETCDF4'
            file_exists = True
        else:
            fnc = nc4.Dataset(ncfilename,'w') #'w' stands for write, format='NETCDF4'
            file_exists = False
        if not any('time' in s for s in fnc.dimensions):        
            fnc.createDimension('time',self.time.size)
            timeNC = fnc.createVariable('time','f',('time',))
            timeNC[:] = self.time.copy()
            timeNC.units = 'seconds since 0000 UTC'
        elif fnc.dimensions['time'].size != self.time.size:
            print('Error in %s write2nc to %s ' %(self.label,ncfilename))
            print('  time dimension exists in %s but has size=%d'%(ncfilename,fnc.dimensions['time'].size))
            print('  time dimension in %s has size=%d'%(self.label,self.time.size))
            nc_error = True
        if not any('range' in s for s in fnc.dimensions):        
            fnc.createDimension('range',self.range_array.size)
            rangeNC = fnc.createVariable('range','f',('range',))
            rangeNC[:] = self.range_array.copy()
            rangeNC.units = 'meters'
        elif fnc.dimensions['range'].size != self.range_array.size:
            print('Error in %s write2nc to %s ' %(self.label,ncfilename))
            print('  range dimension exists in %s but has size=%d'%(ncfilename,fnc.dimensions['range'].size))
            print('  range dimension in %s has size=%d'%(self.label,self.range_array.size))
            nc_error = True
            
        if not any('wavelength' in s for s in fnc.variables):        
            wavelengthNC = fnc.createVariable('wavelength','f')
            wavelengthNC[:] = self.wavelength
            wavelengthNC.units = 'meters'
            
        ###
        # Additional Checks needed to make sure the time and range arrays are identical to what is in the data    
        ###
            
        if not nc_error:
            if name_override:
                varname = tag
            else:
                varname = self.label.replace(' ','_')+tag
                
            profileNC = fnc.createVariable(varname,'double',('time','range'))
            profileNC[:,:] = self.profile.copy()
            profileNC.units = self.profile_type
            var_profileNC = fnc.createVariable(varname+'_variance','double',('time','range'))
            var_profileNC[:,:] = self.profile_variance.copy()
            
            if file_exists:
                fnc.history = fnc.history + "\nModified " + datetime.datetime.today().strftime("%m/%d/%Y")     
            else:
                fnc.history = "Created " + datetime.datetime.today().strftime("%d/%m/%y")
        else:
            print('No netcdf written due to error')
        fnc.close()
        
        
        
   
        
class profile_netcdf():
    def __init__(self,netcdf_var):
        self.data = netcdf_var.data.copy()
        self.dimensions = netcdf_var.dimensions
        self.binwidth_ns = netcdf_var.binwidth_ns
    def copy(self):
        tmp = profile_netcdf(self)
        return tmp

def create_ncfilename(ncbase,Years,Months,Days,Hours):
    """
    filestring = create_ncfilename(ncbase,Years,Months,Days,Hours)
    Creates a netcdf filename based on the requested processing interval
    and the current date.  Returns a string with the netcdf filename
    
    ncbase = a string containing the base to which dates and times should 
        tagged
    Years, Months, Days, Hours - array outputs generated by 
        generate_WVDIAL_day_list()
    """
    
    runday = datetime.datetime.today().strftime("%Y%m%d")
    startstr = str(Years[0])
    if Months[0] < 10:
        startstr = startstr+'0'+str(Months[0])
    else:
        startstr = startstr+str(Months[0])
    if Days[0] < 10:
        startstr = startstr+'0'+str(Days[0])
    else:
        startstr = startstr+str(Days[0])
    if Hours[0,0] < 10:
        startstr = startstr+'T'+'0'+str(np.int(Hours[0,0]))
    else:
        startstr = startstr+'T'+str(np.int(Hours[0,0]))
    Minutes = np.int(60*np.remainder(Hours[0,0],1))
    if Minutes < 10:
        startstr = startstr+'0'+str(Minutes)
    else:
        startstr = startstr+str(Minutes)
    
    stopstr = str(Years[-1])
    if Months[-1] < 10:
        stopstr = stopstr+'0'+str(Months[-1])
    else:
        stopstr = stopstr+str(Months[-1])
    if Days[-1] < 10:
        stopstr = stopstr+'0'+str(Days[-1])
    else:
        stopstr = stopstr+str(Days[-1])
    if Hours[-1,-1] < 10:
        stopstr = stopstr+'T'+'0'+str(np.int(Hours[-1,-1]))
    else:
        stopstr = stopstr+'T'+str(np.int(Hours[-1,-1]))
    Minutes = np.int(60*np.remainder(Hours[-1,-1],1))
    if Minutes < 10:
        stopstr = stopstr+'0'+str(Minutes)
    else:
        stopstr = stopstr+str(Minutes)
    
    ncfilename = ncbase + '_' + startstr + '_' + stopstr + '_created_' + runday + '.nc'

    return ncfilename

def load_geofile(filename):
    geo = np.loadtxt(filename)
    return geo
    
def load_diff_geofile(filename,chan='backscatter'):
    diff_geo0 = np.loadtxt(filename)
    if chan == 'backscatter':
        diff_geo = {'bins':diff_geo0[:,0],'hi':1.0/diff_geo0[:,1],'lo':1.0/diff_geo0[:,2]}
    else:
        diff_geo = {'bins':diff_geo0[:,0],'cross':1.0/diff_geo0[:,1]}
    return diff_geo
    
def ncvar(ncID,varname):
    var = np.array(ncID.variables[varname].data.copy())
    return var
    
    
def plotprofiles(proflist,varplot=False):
    colorlist = ['b','g','r','c','m','y','k']
    plt.figure();
    for ai in range(len(proflist)):
        p1 = proflist[ai].copy()
        p1.time_integrate()
        plt.semilogx(p1.profile.flatten(),p1.range_array.flatten(),colorlist[np.mod(ai,len(colorlist))]+'-',label=p1.label)
        if varplot:
            plt.semilogx(np.sqrt(p1.profile_variance.flatten()),p1.range_array.flatten(),colorlist[np.mod(ai,len(colorlist))]+'--',label=p1.label+' std.')
        plt.grid(b=True);
        plt.legend()
        plt.ylabel('Range [m]')
        plt.xlabel(p1.profile_type)
        
def pcolor_profiles(proflist,ylimits=[0,np.nan],tlimits=[np.nan,np.nan],plotAsDays=False,scale=[]):
    """
    plot time and range resolved profiles as pcolors
    proflist - a list of lidar profiles
    ylimits - list containing upper and lower bounds of plots in km
    tlimits - list containing upper and lower bounds of plots in hr
    """
    
    Nprof = len(proflist)*1.0
    if plotAsDays:
        time_scale = 3600*24.0
        span_scale = 24.0
    else:
        time_scale = 3600.0
        span_scale = 1.0
    
    tmin = 1e9
    tmax = 0
    ymin = 1e9
    ymax = 0
    
    for ai in range(len(proflist)):
        tmin = np.min(np.array([tmin,proflist[ai].time[0]/time_scale]))
        tmax = np.max(np.array([tmax,proflist[ai].time[-1]/time_scale]))
        ymin = np.min(np.array([ymin,proflist[ai].range_array[0]*1e-3]))
        ymax = np.max(np.array([ymax,proflist[ai].range_array[-1]*1e-3]))
    if np.isnan(tlimits[0]):
        tlimits[0] = tmin
    if np.isnan(tlimits[1]):
        tlimits[1] = tmax
    if np.isnan(ylimits[0]):
        ylimits[0] = ymin
    if np.isnan(ylimits[1]):
        ylimits[1] = ymax
        
    # scale figure dimensions based on time and altitude dimensions
    time_span = tlimits[1]/span_scale-tlimits[0]/span_scale  # time domain of plotted data
    range_span = ylimits[1]-ylimits[0]  # range domain of plotted data
    
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

    
    ax_len = np.max(np.array([np.min(np.array([max_len,time_span*18.0/24.0])),min_len])) # axes length
    ax_h = np.max(np.array([np.min(np.array([max_h,range_span*2.1/12])),min_h]))  # axes height
    fig_len = x_left_edge+x_right_edge+ax_len  # figure length
    fig_h =y_bottom_edge+y_top_edge+ax_h  # figure height
    
    figL = []  # figure list
    axL = []   # axes list
    caxL = []  # color axes list
    imL = []   # image list
        
    for ai in range(len(proflist)): 
        axlim = [x_left_edge/fig_len,y_bottom_edge/fig_h/Nprof+(Nprof-ai-1)/Nprof,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/Nprof]
#        axlim3 = [[x_left_edge/fig_len,y_bottom_edge/fig_h/3.0+2.0/3.0,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/3.0], \
#        [x_left_edge/fig_len,y_bottom_edge/fig_h/3.0+1.0/3.0,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/3.0], \
#        [x_left_edge/fig_len,y_bottom_edge/fig_h/3.0+0.0/3.0,1-x_right_edge/fig_len,(1-y_top_edge/fig_h)/3.0]]
    
        fig = plt.figure(figsize=(fig_len,Nprof*fig_h)); # plt.figure(figsize=(15,5))
        ax = plt.axes(axlim) 
        im = plt.pcolor(proflist[ai].time/time_scale,proflist[ai].range_array*1e-3,np.log10(proflist[ai].profile).T)
        plt.clim([-8,-4])
        plt.ylim(ylimits)
        plt.xlim(tlimits/time_scale)
        plt.title(DateLabel + ', ' +proflist[ai].lidar + line_char + proflist[ai].profile_type,fontsize=title_font_size)
        plt.ylabel('Altitude AGL [km]')
        if plotAsDays:
            plt.xlabel('Days [UTC]')
        else:
            plt.xlabel('Time [UTC]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size=0.1,pad=0.2)
        plt.colorbar(im,cax=cax)
        
        figL.extend([fig])
        axL.extend([ax])
        caxL.extend([cax])
        imL.extend([imL])
        
        
def read_WVDIAL_binary(filename,MCSbins):    
    """
    Function for reading WV-DIAL (.dat) binary files
    Function for reading WV-DIAL (.dat) binary files
    filename - .dat file containing WV-DIAL data
    MCSbins - int containing the number of bins the MCS is writing out
    profileData - 2D array containing the time resolved profiles (time along the first dimension,
              range along the second dimension)
    varData - 2D array containing additional variables stuffed into the header
              each row contains the time resolved value.
    """
    f = open(filename,"rb")
    data = np.fromfile(f,dtype=np.double)
    f.close()

    extraVar = 6  # number of extra variables preceeding the profile
    
    data = data.reshape((MCSbins+extraVar,-1),order='F')
    data = data.newbyteorder()
    profileData = data[extraVar:,:]
    varData = data[0:extraVar+1,:]
    
    return profileData,varData
        
def generate_WVDIAL_day_list(startYr,startMo,startDay,startHr=0,duration=0.0,stopYr=0,stopMo=0,stopDay=0,stopHr=24):
    """
    Generates a list of processing Year,Day,Month,Hour to process an arbitrary
    amount chunk of time for WV-DIAL or DLB-HSRL
    The function requires the start year, month, day, (start hour is optional)
    The processing chunk must then either be specified by:
        duration in hours
        stop year, month, day and hour (stopYr,stopMo,stopDay,stopHr) 
            - stop hour does not have to be specified in which case it is the 
            end of the day
    """    
    startDate = datetime.datetime(startYr,startMo,startDay)+datetime.timedelta(hours=startHr)
    
    # start the day processing list based on start times
    YrList = np.array([startYr])    
    MoList = np.array([startMo])
    DayList = np.array([startDay])
    HrList = np.array([[startHr],[24]])
    
    if duration != 0:
        stopDate = startDate+datetime.timedelta(hours=duration)
    else:
        if stopYr == 0:
            stopYr = startDate.year
        if stopMo == 0:
            stopMo = startDate.month
        if stopDay == 0:
            stopDay = startDate.day
        stopDate = datetime.datetime(stopYr,stopMo,stopDay)+datetime.timedelta(hours=stopHr) 
        
    nextday = True
        
    while nextday:
        if stopDate.date() == startDate.date():
            HrList[1,-1] = stopDate.hour+stopDate.minute/60.0+stopDate.second/3600.0
            nextday = False
        else:
            startDate = startDate+datetime.timedelta(days=1)
            YrList = np.concatenate((YrList,np.array([startDate.year])))
            MoList = np.concatenate((MoList,np.array([startDate.month])))
            DayList = np.concatenate((DayList,np.array([startDate.day])))
            HrList = np.hstack((HrList,np.array([[0],[24]])))
        
    return YrList,MoList,DayList,HrList
    
def AerosolBackscatter(MolProf,CombProf,Sonde,negfilter=True):
    """
    Calculate the Aerosol Backscatter Coeffcient LidarProfiles: Molecular and Combined Channels
    Expects a 2d sonde profile that has the same dimensions as the Molecular and combined channels
    
    Set negfilter = False to avoid filtering out negative values
    """
    
    Beta_AerBS = MolProf.copy()

    # calculate backscatter ratio
    BSR = CombProf.profile/MolProf.profile
    
#    Beta_AerBS.profile = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
    Beta_AerBS.profile = (BSR.copy()-1)
    Beta_AerBS.profile_variance = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
    Beta_AerBS.multiply_prof(Sonde)    
    
    Beta_AerBS.descript = 'Calibrated Measurement of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    Beta_AerBS.label = 'Aerosol Backscatter Coefficient'
    Beta_AerBS.profile_type = 'Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]'
    
    if negfilter:
        Beta_AerBS.profile[np.nonzero(Beta_AerBS.profile <= 0)] = 1e-10;
    
    return Beta_AerBS
        
def Calc_AerosolBackscatter(MolProf,CombProf,Temp = np.array([np.nan]),Pres = np.array([np.nan]),negfilter=True,beta_sonde_scale=1.0):
    """
    Calculate the Aerosol Backscatter Coeffcient LidarProfiles: Molecular and Combined Channels
    Temperature and Pressure profiles can be specified.  Otherwise it uses a standard atmosphere.
    Pressure is expected in Pa ( 101325 Pa /atm )
    Temperature is expected in K
    If either is only one element long, the first element will be used to seed the base conditins
    in the standard atmosphere.
    
    Set negfilter = False to avoid filtering out negative values
    """
    
    Beta_AerBS = MolProf.copy()
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*Beta_AerBS.range_array*1e-3
        else:
            Temp = Temp[0]-5*Beta_AerBS.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-Beta_AerBS.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-Beta_AerBS.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/Beta_AerBS.wavelength)**4*1e-32*Pres/(Temp*kB)
   
    # calculate backscatter ratio
    BSR = CombProf.profile/MolProf.profile
    
#    Beta_AerBS.profile = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
    Beta_AerBS.profile = (BSR.copy()-1)
    Beta_AerBS.profile_variance = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
    Beta_AerBS.diff_geo_overlap_correct(beta_m_sonde,geo_reference='sonde')  # only aerosol backscatter
    
    Beta_AerBS.descript = 'Calibrated Measurement of Aerosol Backscatter Coefficient in m^-1 sr^-1'
    Beta_AerBS.label = 'Aerosol Backscatter Coefficient'
    Beta_AerBS.profile_type = 'Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]'
    
    if negfilter:
        Beta_AerBS.profile[np.nonzero(Beta_AerBS.profile <= 0)] = 1e-10;
    
    return Beta_AerBS
        
def Calc_Extinction(MolProf,MolConvFactor = 1.0,Temp = np.array([np.nan]),Pres = np.array([np.nan]),Cam=0.005,AerProf=np.array([0])):


    
    OptDepth = MolProf.copy()
    OptDepth.gain_scale(MolConvFactor)  # Optical depth requires scaling to avoid bias
    
    OptDepth.descript = 'Optical Depth of the altitude Profile starting at lidar base'
    OptDepth.label = 'Optical Depth'
    OptDepth.profile_type = 'Optical Depth'
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*OptDepth.range_array*1e-3
        else:
            Temp = Temp[0]-5*OptDepth.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-OptDepth.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-OptDepth.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/OptDepth.wavelength)**4*1e-32*Pres/(Temp*kB)
    
    OptDepth.diff_geo_overlap_correct(1.0/beta_m_sonde)   
    OptDepth.profile_variance = OptDepth.profile_variance*(1/OptDepth.profile)**2
#    Transmission = OptDepth.copy()
    OptDepth.profile = np.log(OptDepth.profile)
    OptDepth.gain_scale(-0.5)
    
    ODmol = np.cumsum(4*np.pi*beta_m_sonde*OptDepth.mean_dR)
    
    # if an aerosol backscatter profile is passed in, use it to help estimate uncertainty
    if hasattr(AerProf,'profile'):
        varCam = (Cam/2.0)**2
        ErrAerVar = 1.0/(beta_m_sonde+Cam*AerProf.profile)**2*(AerProf.profile**2*varCam+Cam**2*AerProf.profile_variance)
        OptDepth.profile_variance = 0.5**2*ErrAerVar
    
    Alpha = OptDepth.copy()   
    Alpha.profile[:,1:] = np.diff(Alpha.profile,axis=1)
    Alpha.profile_variance[:,1:] = Alpha.profile_variance[:,:-1] + Alpha.profile_variance[:,1:]
    Alpha.gain_scale(1.0/Alpha.mean_dR)
    
    Alpha.descript = 'Atmospheric Extinction Coefficient in m^-1'
    Alpha.label = 'Extinction Coefficient'
    Alpha.profile_type = 'Extinction Coefficient [$m^{-1}$]'
    
    return Alpha,OptDepth,ODmol    


def FilterCrossTalkCorrect(MolProf,CombProf,Cam,smart=False):
    """
    FilterCrossTalkCorrect(MolProf,CombProf,Cam)
    remove the aerosol coupling into the molecular channel.
    MolProf - LidarProfile type of Molecular channel
    CombProf - LidarProfile type of Combined channel
    Cam - normalized coupling coefficient of aerosol backscatter to the molecular
        channel.
        Typically < 0.01
    smart - if set to True, corrections will only be applied where the added 
            noise effect is less than that of the coupling error        
        
    Correcting this cross talk has some tradeoffs.
        It improves backscatter coefficient estimates by a percentage coparable to Cam but
        It couples additional noise from the aerosol channel into the molecular channel
        It can introduce more error in cases where one channel is driven into more nonlinearity
            than another
    """
    
    
    
    if smart:
#        DeltaBSR = Cam*(CombProf.profile**2/MolProf.profile**2+(CombProf.profile/MolProf.profile))
        DeltaBSR = Cam*CombProf.profile*(MolProf.profile+CombProf.profile)/(MolProf.profile*(MolProf.profile+Cam*CombProf.profile))
        BSR0 = MolProf.profile_variance*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
        BSRcor = (MolProf.profile_variance+Cam**2*CombProf.profile_variance)*(CombProf.profile)**2/(MolProf.profile)**4+CombProf.profile_variance*1/(MolProf.profile)**2
        CorrectMask = np.abs(DeltaBSR) > np.sqrt(BSRcor)-np.sqrt(BSR0)  # mask for locations to apply the correction
        MolProf.profile[CorrectMask] = 1.0/(1-Cam)*(MolProf.profile[CorrectMask]-CombProf.profile[CorrectMask]*Cam)
        MolProf.profile_variance[CorrectMask] = MolProf.profile_variance[CorrectMask]+Cam**2*CombProf.profile_variance[CorrectMask]
    else:    
        MolProf.profile = 1.0/(1-Cam)*(MolProf.profile-CombProf.profile*Cam)
        MolProf.profile_variance = MolProf.profile_variance+Cam**2*CombProf.profile_variance
        
def AerBackscatter_DynamicIntegration(MolProf,CombProf,Temp = np.array([np.nan]),Pres = np.array([np.nan]),num=3,snr_th=1.2,sigma = np.array([1.5,1.0]),beta_sonde_scale=1.0):
    beta0 = Calc_AerosolBackscatter(MolProf,CombProf,Temp=Temp,Pres=Pres,beta_sonde_scale=beta_sonde_scale)
    mask1 = np.log10(beta0.SNR()) > snr_th
    MolNew = MolProf.copy()
    CombNew = CombProf.copy()
    
    MolNew.profile = np.ma.array(MolNew.profile,mask=mask1)    
    CombNew.profile = np.ma.array(CombNew.profile,mask=mask1) 
    
    # avoid applying convolutions to the bottom rows of the profile
    zk,tk,kconv = MolNew.get_conv_kernel(sigma[0],sigma[1])
    mask1[:,:zk.shape[1]] = True
    
    MolNew.conv(sigma[0],sigma[1])
    CombNew.conv(sigma[0],sigma[1])
    
    if num == 0 or mask1.all():
        # merge top profile with new profile
        MolNew.profile[mask1] = MolProf.profile[mask1]
        MolNew.profile_variance[mask1] = MolProf.profile_variance[mask1]
        CombNew.profile[mask1] = CombProf.profile[mask1]
        CombNew.profile_variance[mask1] = CombProf.profile_variance[mask1]
        ProfResZ = np.ones(CombNew.profile.shape)*sigma[1]
        ProfResZ[mask1] = 0
        ProfResT = np.ones(CombNew.profile.shape)*sigma[0]
        ProfResT[mask1] = 0
        
        #return profiles
        return MolNew,CombNew,beta0,ProfResT,ProfResZ
    else:
        M1,C1,b1,Rt,Rz = AerBackscatter_DynamicIntegration(MolNew,CombNew,Temp=Temp,Pres=Pres,num=(num-1),snr_th=snr_th,beta_sonde_scale=beta_sonde_scale)
        # merge with next profile level up
        M1.profile[mask1] = MolProf.profile[mask1]
        M1.profile_variance[mask1] = MolProf.profile_variance[mask1]
        C1.profile[mask1] = CombProf.profile[mask1]
        C1.profile_variance[mask1] = CombProf.profile_variance[mask1]
        betaNew = Calc_AerosolBackscatter(M1,C1,Temp=Temp,Pres=Pres) 
        # update the resolution arrays
        Rt = Rt + sigma[0]
        Rt[mask1] = 0
        Rz = Rz + sigma[1]        
        Rz[mask1] = 0
        
        # return layered molecular profle, combined profile, backscatter profile, time resolution, altitude resolution
        return M1,C1,betaNew,Rt,Rz
        
def Retrieve_Ext_MLE(OptDepth,aer_beta,ODmol,blocksize=8,overlap=0,SNR_th=1.1,lam=np.array([10.0,3.0]),max_iterations=200): 
    if overlap == 0:
        overlap = blocksize/2
    
    Extinction = OptDepth.copy()
    Extinction.descript = 'Atmospheric Extinction Coefficient in m^-1 retrieved with MLE'
    Extinction.label = 'Extinction Coefficient'
    Extinction.profile_type = 'Extinction Coefficient [$m^{-1}$]'
    
    LidarRatio = OptDepth.copy()
    LidarRatio.descript = 'Atmospheric LidarRatio in sr retrieved with MLE'
    LidarRatio.label = 'LidarRatio'
    LidarRatio.profile_type = 'LidarRatio [$sr}$]'
    
    OptDepthMLE = OptDepth.copy()
    OptDepthMLE.descript = 'Optical Depth of the altitude Profile starting at lidar base, determined with MLE'
    
    
    # keep a record of which points were set to zero
    x_fit = np.ones(Extinction.profile.shape)    
    # keep a record of the Optical Depth Bias estimate
    ODbiasProf = np.zeros(Extinction.profile.shape[0])
    
    iprof = 0
    while iprof < Extinction.profile.shape[0]:    
        if iprof+blocksize < Extinction.profile.shape[0]:
            aerfit = aer_beta.profile[iprof:iprof+blocksize,:]
            aerfit_std = np.sqrt(aer_beta.profile_variance[iprof:iprof+blocksize,:])
            ODfit = OptDepth.profile[iprof:iprof+blocksize,:]-ODmol[np.newaxis,:]
            ODvar = OptDepth.profile_variance[iprof:iprof+blocksize,:]                
            aerfit[np.nonzero(np.isnan(aerfit))] = 0
            x_invalid = np.nonzero(aerfit < SNR_th*aerfit_std)
            x_fit[x_invalid[0]+iprof,x_invalid[1]] = 0
            sLR,ODbias = Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=blocksize+1,minLR=OptDepth.mean_dR,lam=lam,max_iterations=max_iterations)
            ODbiasProf[iprof:iprof+blocksize]=ODbias
            if iprof == 0:            
                LidarRatio.profile[iprof:iprof+blocksize,:] = sLR
            else:
                # Average the overlapping chunk
                LidarRatio.profile[iprof:iprof+overlap,:]=0.5*(LidarRatio.profile[iprof:iprof+overlap,:]+sLR[:overlap,:])
                # Fill in the new chunk
                LidarRatio.profile[iprof+overlap:iprof+blocksize,:] = sLR[overlap:,:]
        else:
            aerfit = aer_beta.profile[iprof:,:]
            aerfit_std = np.sqrt(aer_beta.profile_variance[iprof:,:])
            ODfit = OptDepth.profile[iprof:,:]-ODmol[np.newaxis,:]
            ODvar = OptDepth.profile_variance[iprof:,:]                
            aerfit[np.nonzero(np.isnan(aerfit))] = 0
            x_invalid = np.nonzero(aerfit < SNR_th*aerfit_std)
            x_fit[x_invalid[0]+iprof,x_invalid[1]] = 0
            sLR,ODbias = Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=blocksize+1,minLR=OptDepth.mean_dR,lam=lam,max_iterations=max_iterations)
            if iprof == 0:            
                LidarRatio.profile[iprof:iprof+blocksize,:] = sLR
            else:
                # Average the overlapping chunk
                LidarRatio.profile[iprof:iprof+overlap,:]=0.5*(LidarRatio.profile[iprof:iprof+overlap,:]+sLR[:overlap,:])
                # Fill in the new chunk
                LidarRatio.profile[iprof+overlap:,:] = sLR[overlap:,:]
        print('Completed %d of %d'%(iprof,Extinction.profile.shape[0]))        
        iprof=iprof+(blocksize-overlap)
    
    Extinction.profile = aer_beta.profile*x_fit*LidarRatio.profile
    Extinction.profile[np.nonzero(np.isnan(Extinction.profile))] = 0
    OptDepthMLE.profile= np.cumsum(Extinction.profile,axis=1)
    Extinction.profile = Extinction.profile/OptDepth.mean_dR
    LidarRatio.profile = LidarRatio.profile/OptDepth.mean_dR
    
    return Extinction,OptDepthMLE,LidarRatio,ODbiasProf,x_fit
    
    
def Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=10,maxLR=1e5,minLR=75.0,lam=np.array([10.0,3.0]),grad_gain=1e-1,max_iterations=200,optout=-1):
    """
    Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=10,SNR_th=3.3,maxLR=1e5)
    Uses optimization to estimate the extinction coefficent of aerosols and clouds
    Reduces the estimation problem to cases where aerosols are present (determined by SNR threshold-SNR_th)
    ODfit - array of Optical Depth profiles to be fit
    ODvar - uncertainty in those OD measurements in ODfit
    aerfit - array of aerosol backscatter coefficeints
    x_invalid - points in the array that are ignored due to lack of aerosol signal
    maxblock - number of profiles the function is allowed to operate on
    maxLR - maximum Lidar Ratio the optimizor is allowed to use
    minLR - minimum Lidar Ratio the optimizor is allowed to use (typically the range resolution)
    lam - two element array indicating the TV norm sensitivity in altitude and time respectively
    grad_gain - use smaller numbers (<1) to speed up convergence of the optimizer at the cost of accuracy
    max_iterations - maxium iterations allowed from the optimizor
    optout - optimizor output setting
        optout <= 0 : Silent operation
        optout == 1 : Print summary upon completion (default)
        optout >= 2 : Print status of each iterate and summary

    """
    
    rangeIndex = 20  # sets the minimum range where the OD will be used to retrieve extinction    
    
    if ODfit.shape[0] > maxblock:
        print('Max block size exceeded in Retrieve_Ext_Block_MLE')
        print('Current limit: maxblock=%d'%maxblock)
        print('Current number of profiles: %d'%ODfit.shape[0])
        print('Expect issues with x_invalid')
        ODfit = ODfit[:,maxblock,:]
        ODvar = ODvar[:,maxblock,:]
        aerfit = aerfit[:,maxblock,:]
        
#    timeLim = np.array([10.0,15])
#    NumProfs = 8
#    
#    b1 = np.argmin(np.abs(OptDepth.time/3600.0-timeLim[0]))  # 6.3, 15
#    #b2 = np.argmin(np.abs(OptDepth.time/3600-timeLim[1]))  # 6.3, 15
#    b2 = b1+NumProfs
#    
#    aerfit = aer_beta_dlb.profile[b1:b2,:]
#    #aerfit[np.nonzero(aerfit<0)] = 0
#    #aerfit[np.nonzero(np.isnan(aerfit))] = np.nan
#    aerfit_std = np.sqrt(aer_beta_dlb.profile_variance[b1:b2,:])
#    ODfit = OptDepth.profile[b1:b2,:]-ODmol[np.newaxis,:]
#    ODvar = OptDepth.profile_variance[b1:b2,:]
#    
#    sLR = np.zeros(aerfit.shape)
#    aerfit[np.nonzero(np.isnan(aerfit))] = 0
#    x_invalid = np.nonzero(aerfit < 3.3*aerfit_std)
    
    aerfit[x_invalid] = 0.0
    aerfit[:,:rangeIndex] = 0.0
    
#    x0 = np.ones(aerfit.size)*75*50*np.rand.random()
    x0 = 75*(25*np.random.rand(aerfit.size)+50)
    
#    x0 = (-np.log10(aerfit)-2)*1000.0
#    x0[np.nonzero(x0==0)] = 2000.0
#    x0[np.nonzero(x0==np.inf)] = 2000.0
    
    #x0[:] = 1200 #np.log10(1200)
    #x0[1:] = Soln[1:]
    bnds = np.zeros((x0.size,2))
    bnds[:,1] = maxLR
    bnds[:,0] = minLR
    
#    ODbias = -np.nanmin(ODfit[:,:2]) # -np.nanmin(ODfit[:,1:30],axis=1)[:,np.newaxis] # 19.05
    ODbias = -np.nanmean(ODfit[:,rangeIndex:rangeIndex+4])
    
    FitError = lambda x: Fit_LR_2D(x,aerfit,ODfit,ODbias,ODvar,lam=lam)  # substitute x[0] for ODbias to adaptively find bias
    gradFun0 = lambda x: Fit_LR_2D_prime(x,aerfit,ODfit,ODbias,ODvar,lam=lam)*grad_gain
    
    Soln = scipy.optimize.fmin_slsqp(FitError,x0,fprime=gradFun0,bounds=bnds,iter=max_iterations,iprint=optout) #fprime=gradFun0 acc=1e-14 fprime=gradFun0
    
    #sLR[xvalid] = 1200 #np.random.rand(xvalid.shape)*10
    
    #
    #ODsLR = np.cumsum(aerfit*sLR)+ODbias
    sLR = Soln.reshape(aerfit.shape)
#    extinction = aerfit*sLR
#    extinction[np.nonzero(np.isnan(extinction))] = 0
#    ODSoln = np.cumsum(extinction,axis=1)-ODbias
    
    return sLR, ODbias #,extinction,ODSoln,ODbias
#    extinction = extinction/OptDepth.mean_dR  # adjust for bin width    
        
def Fit_LR_2D(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0])):
    """
    Fit_LR_A_2D(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0]))
    Obtain an error estimate for the Optical Depth as a function of retrieved
    Lidar Ratio
    """    
    x2D = x.reshape(aerfit.shape)
    extinction = x2D*aerfit
    extinction[np.nonzero(np.isnan(extinction))] = 0
    ODsLR = np.cumsum(extinction,axis=1)-ODbias    
    ODfit = ODfit
    
    if any(lam > 0):
        deriv = lam[0]*np.nansum(np.abs(np.diff(x2D,axis=0)))+lam[1]*np.nansum(np.abs(np.diff(x2D,axis=1)))
        ErrorOut = np.nansum(0.5*(ODfit-ODsLR)**2/ODvar)+deriv
    else:
        ErrorOut = np.nansum(0.5*(ODfit-ODsLR)**2/ODvar)
    return ErrorOut

def Fit_LR_2D_prime(x,aerfit,ODfit,ODbias,ODvar,lam=np.array([0,0])):

    x2D = x = x.reshape(aerfit.shape)
    extinction = x2D*aerfit
    extinction[np.nonzero(np.isnan(extinction))] = 0
    ODsLR = np.cumsum(extinction,axis=1)-ODbias  


    gradErr = np.cumsum(aerfit,axis=1)*(-(ODfit-ODsLR)/ODvar)   
    gradErr[np.nonzero(np.isnan(gradErr))] = 0
    
    if any(lam >0):
        gradpen = lam[0]*np.sign(np.diff(x2D,axis=0))
        gradpen[np.nonzero(np.isnan(gradpen))] = 0
        gradErr[:-1,:] = gradErr[:-1,:]-gradpen
        gradErr[1:,:] = gradErr[1:,:]+gradpen
        
        gradpen = lam[1]*np.sign(np.diff(x2D,axis=1))
        gradpen[np.nonzero(np.isnan(gradpen))] = 0
        gradErr[:,:-1] = gradErr[:,:-1]-gradpen
        gradErr[:,1:] = gradErr[:,1:]+gradpen   
    
    return gradErr.flatten()  

def Klett_Inv(Comb,RefAlt,Temp = np.array([np.nan]),Pres = np.array([np.nan]),avgRef=False,geo_corr=np.array([]),BGIndex=-50,Nmean=0,kLR=1.0):
    """
    Klett_Inv(Comb,Temp,Pres,RefAlt,avgRef=False,geo_corr=np.array([]),BGIndex=-50)
    
    Accepts a raw photon count profile and estimates the aerosol backscatter
    using the Klett inversion (Klett, Appl. Opt. 1981)
    
    Comb - raw lidar profile.
    RefAlt - altitude in meters used a known reference point
    Temp - Temperature Profile in K - uses standard atmosphere if not provided
    Pres - Pressure Profile in Atm - uses standard atmosphere if not provided
    avgRef - if set to True, the signal value used for the reference altitude
            uses the time average of all data at that altitude
    geo_corr - input for a geometric overlap correction if known
    BGIndex - index for where to begin averaging to determine background levels
    Nmean - if avgRef=False, this sets the smoothing interval for estimating
            the signal at the reference altitude.  Noise at the reference
            altitude couples into the profiles.  
            If Nmean = 0, no smoothing is performed
    kLR - lidar ratio exponent to be used under 
            backscatter = const * extinction^kLR
            defaults to 1.0

    """
    CombK = Comb.copy()
    if not any('Background Subtracted over' in s for s in CombK.ProcessingStatus):
        CombK.bg_subtract(BGIndex)
    if not any('Applied R^2 Range Correction' in s for s in CombK.ProcessingStatus):
        CombK.range_correct()
    if geo_corr.size > 0 and not any('Geometric Overlap Correction' in s for s in CombK.ProcessingStatus):
        CombK.geo_overlap_correct(geo_corr)
    
    iref = np.argmin(np.abs(CombK.range_array-RefAlt))
    
#    kLR = 1.0
    
    if np.isnan(Temp).any() or np.isnan(Pres).any():
        if np.isnan(Temp[0]):
            Temp = 300-5*CombK.range_array*1e-3
        else:
            Temp = Temp[0]-5*CombK.range_array*1e-3
        if np.isnan(Pres[0]):
            Pres = 101325.0*np.exp(-CombK.range_array*1e-3/8)
        else:
            Pres = Pres[0]*np.exp(-CombK.range_array*1e-3/8)
    # Calculate expected molecular backscatter profile based on temperature and pressure profiles
    beta_m_sonde = 5.45*(550.0e-9/CombK.wavelength)**4*1e-32*Pres/(Temp*kB)    
    
    sigKref = (8*np.pi/3.0)*beta_m_sonde[iref]
    
    Sc = np.log(CombK.profile)
    Sc[np.nonzero(np.isnan(Sc))] = 0
    if avgRef:
        Scm = np.nanmean(Sc[:,iref])
    elif Nmean > 0:
        Nmean = 2*np.int(Nmean/2)  # force the conv kernel to be even (for simplicity)
        convKer = np.ones(Nmean)*1.0/Nmean
        GainMask = np.ones(Sc.shape[0])
        GainMask[:Nmean/2] = 1.0*Nmean/np.arange(Nmean/2,Nmean)
        GainMask[-Nmean/2+1:] = 1.0*Nmean/np.arange(Nmean-1,Nmean/2,-1)
        Scm = Sc[:,iref]
        izero = np.nonzero(Scm==0)[0]
        inonzero = np.nonzero(Scm)[0]
        ScmInterp = np.interp(izero,inonzero,Scm[inonzero])
        Scm[izero] = ScmInterp
        Scm = np.convolve(Scm,convKer,mode='same')[:,np.newaxis]
#        plt.figure();
#        plt.plot(Sc[:,iref]);
#        plt.plot(Scm.flatten());
        Scm = Scm*GainMask[:,np.newaxis]
#        plt.plot(Scm.flatten())
        
    else:
        Scm = (Sc[:,iref])[:,np.newaxis]
    
    Beta_AerBS = CombK.copy()
    Beta_AerBS.descript = 'Klett Estimate of Aerosol Backscatter Coefficient in m^-1 sr^-1\n using a %d m Reference Altitude\n '%RefAlt
    Beta_AerBS.label = 'Aerosol Backscatter Estimate'
    Beta_AerBS.profile_type = 'Klett estimated Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]' 
    Beta_AerBS.profile_variance = np.zeros(Beta_AerBS.profile_variance.shape)
    
    sigK = np.zeros(Sc.shape)
#    sigK[:,:iref+1] = np.exp((Sc-Scm)[:,:iref+1]/kLR)/(1.0/sigKref+2/kLR*np.cumsum((Sc-Scm)[:,iref::-1]/kLR,axis=1)[:,::-1]*CombK.mean_dR)
    sigK[:,:iref+1] = np.exp((Sc[:,:iref+1]-Scm)/kLR)/(1.0/sigKref+2/kLR*np.fliplr(np.cumsum(np.fliplr((Sc[:,:iref+1]-Scm)/kLR),axis=1))*CombK.mean_dR)
    Beta_AerBS.profile = sigK/(8*np.pi/3.0)-beta_m_sonde[np.newaxis,:]
    
    return Beta_AerBS
    

def Poisson_Thin(y,n=2):
    """
    Poisson_Thin(y,n=2)
    For y an array of poisson random numbers, this function
    builds n statistically independent copies of y
    returns a list of the copies as numpy arrays
    """
    p = 1.0/n
    
    copylist = [];
    for ai in range(n):
        copy = np.zeros(y.shape)
        copy = np.random.binomial(y,p)
        copylist.extend([copy])
    return copylist

def RB_Spectrum(T,P,lam,nu=np.array([])):
    """
    RB_Spectrum(T,P,lam,nu=np.array([]))
    Obtain the Rayleigh-Brillouin Spectrum of Earth atmosphere
    T - Temperature in K.  Accepts an array
    P - Pressure in Atm.  Accepts an array with size equal to size of T
    lam - wavelength in m
    nu - frequency basis.  If not supplied uses native frequency from the PCA
        analysis.
    
    
    """
    # Obtain the y parameters from inputs
    yR = RayleighBrillouin_Y(T,P,lam);

    #Load results from PCA analysis (RB_PCA.m)
    # Loads M, Mavg, x1d
    RBpca = np.load('/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/RB_PCA_Params.npz');
    M = RBpca['M']
    Mavg = RBpca['Mavg']
    x = RBpca['x']
    RBpca.close()
    
    # Calculate spectrum based from yR and PCA data
    Spca = Get_RB_PCA(M,Mavg,yR);

    

    if nu.size > 0:
        # if nu is provided, interpolate to obtain requrested frequency grid
        xR = RayleighBrillouin_X(T,lam,nu)
        SpcaI = np.zeros(xR.shape)
   
        for ai in range(T.size):
            SpcaI[:,ai] = np.interp(xR[:,ai],x.flatten(),Spca[:,ai],left=0,right=0)
            
        return SpcaI
#        S1 = interp1(xR,Spca(:,1),xR(:,1));
        
    else:
        # if nu is not provided, return the spectra and the native x axis
        return Spca,x

def Get_RB_PCA(M,Mavg,y):
    y = y.flatten()
    yvec = y[np.newaxis,:]**np.arange(M.shape[1])[:,np.newaxis]
    Spect = Mavg+np.dot(M,yvec)
    return Spect

def RayleighBrillouin_Y(T,P,lam):
    """
    y = RayleighBrillouin_Y(T,P,nu,lambda)
    Calculates the RB parameter y for a given 
    Temperature (T in K), Pressure (P in atm) and wavelength (lambda in m)
    """

#    kB = 1.3806504e-23;
#    Mair = 28.95*1.66053886e-27;
    
    k=np.sin(np.pi/2)*4*np.pi/lam;
    v0=np.sqrt(2*kB*T/Mair);
    
    viscosity=17.63e-6;
#    bulk_vis=viscosity*0.73;
#    thermal_cond=25.2e-3;
#    c_int=1.0;
    
    p_pa=P*1.01325e5;
    n0=p_pa/(T*kB);
    
    y=n0*kB*T/(k*v0*viscosity);
    
    return y

def RayleighBrillouin_X(T,lam,nu):
    """
    [x,y] = RayleighBrillouin_XY(T,P,nu,lambda)
    Calculates the RB parameters x and y for a given 
    Temperature (T in K) and wavelength (lambda in m)
    The parameter x is calculated for the supplied frequency grid nu (in Hz)
    If an array of P and T are passed in, x will be a matrix where each
    column is the x values corresponding to the P and T values
    """

#    kB = 1.3806504e-23;
#    Mair = 28.95*1.66053886e-27;
    
    if isinstance(T, np.ndarray):
        k=np.sin(np.pi/2)*4*np.pi/lam;
        v0=np.sqrt(2*kB*T[np.newaxis,:]/Mair);
        x = nu[:,np.newaxis]/(k*v0/(2*np.pi));
    else:
        k=np.sin(np.pi/2)*4*np.pi/lam;
        v0=np.sqrt(2*kB*T/Mair);
        x = nu/(k*v0/(2*np.pi));
        
    return x

def RayleighBrillouin_XY(T,P,lam,nu):
    """
    [x,y] = RayleighBrillouin_XY(T,P,nu,lambda)
    Calculates the RB parameters x and y for a given 
    Temperature (T in K), Pressure (P in atm) and wavelength (lambda in m)
    The parameter x is calculated for the supplied frequency grid nu (in Hz)
    """

    kB = 1.3806504e-23;
    Mair = 28.95*1.66053886e-27;
    
    k=np.sin(np.pi/2)*4*np.pi/lam;
    v0=np.sqrt(2*kB*T/Mair);
    x = nu/(k*v0/(2*np.pi));
    
    viscosity=17.63e-6;
#    bulk_vis=viscosity*0.73;
#    thermal_cond=25.2e-3;
#    c_int=1.0;
    
    p_pa=P*1.01325e5;
    n0=p_pa/(T*kB);
    
    y=n0*kB*T/(k*v0*viscosity);
    
    return x,y

def voigt(x,alpha,gamma):
    """
    voigt(x,alpha,gamma)
    Calculates a zero mean voigt profile for spectrum x 
    alpha - Gaussian HWHM
    gamma - lorentzian HWMM
    
    for instances where the profile is not zero mean substitute x-xmean for x
    
    see scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    """
    sigma = alpha / np.sqrt(2*np.log(2))
    return np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2)))

def WV_ExtinctionFromHITRAN(nu,TempProf,PresProf,filename='',freqnorm=False,nuLim=np.array([])):
    """
    WV_ExtinctionFromHITRAN(nu,TempProf,PresProf)
    returns a WV extinction profile in m^-1 for a given
    nu - frequency grid in Hz
    TempProf - Temperature array in K
    PresProf - Pressure array in Atm (must be same size as TempProf)
    
    Note that the height of the extinction profile will change based on the
    grid resolution of nu.  
    Set freqnorm=True
    To obtain a grid independent profile to obtain extinction in m^-1 Hz^-1
    
    This function requires access to the HITRAN ascii data:
    '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/WV_HITRAN2012_815_841.txt'
    The data file can be subtituted with something else by using the optional
    filename input.  This accepts a string with a path to the desired file.
    
    If a full spectrum is not needed (nu only represents an on and off line),
    use nuLim to define the frequency limits over which the spectral lines 
    should be included.
    nuLim should be a two element numpy.array
    nuLim[0] = minimum frequency
    nuLim[1] = maximum frequency
    
    """
    nuL = np.mean(nu);
    
    if not filename:
#        print('Using Default HITRAN file')
        filename = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/WV_HITRAN2012_815_841.txt';
    
    Mh2o = (mH2O*1e-3)/N_A; # mass of a single water molecule, kg/mol
    
    # read HITRAN data
    data = np.loadtxt(filename,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9),skiprows=13)
    
    if nuLim.size == 0:
        nuLim = np.array([np.min(nu),np.max(nu)])
    
    #Voigt profile calculation
    wn_nu  = nu/c*1e-2; # convert to wave number in cm^-1
    wn_nuL  = nuL/c*1e-2; # convert laser frequency to wave number in cm^-1
    wn_nuLim = nuLim/c*1e-2  # convert frequency span of included lines to wave number in cm^-1
    #Find lines from WNmin to WNmax to calculate voigt profile
    hitran_line_indices = np.nonzero(np.logical_and(data[:,2] > wn_nuLim[0],data[:,2] < wn_nuLim[1]))[0];
    print('%d'%hitran_line_indices.size)
    
    hitran_T00 = 296;              # HITRAN reference temperature [K]
    hitran_P00 = 1;                # HITRAN reference pressure [atm]
    hitran_nu0_0 = data[hitran_line_indices,2];      # absorption line center wavenumber from HITRAN [cm^-1]
    hitran_S0 = data[hitran_line_indices,3];         # initial linestrength from HITRAN [cm^-1/(mol*cm^-2)]   
    hitran_gammal0 = data[hitran_line_indices,5];    # air-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
#    hitran_gamma_s = data[hitran_line_indices,6];    # self-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
    hitran_E = data[hitran_line_indices,7];          # ground state transition energy from HITRAN [cm^-1]  
    hitran_alpha = data[hitran_line_indices,8];      # linewidth temperature dependence factor from HITRAN
    hitran_delta = data[hitran_line_indices,9];     # pressure shift from HiTRAN [cm^-1 atm^-1]
    
    
    voigt_sigmav_f = np.zeros((np.size(TempProf),np.size(wn_nu)));
    
    
    # calculate the absorption cross section at each range
    for ai in range(np.size(TempProf)): 
        #    %calculate the pressure shifts for selected lines as function of range
        hitran_nu0 = hitran_nu0_0+hitran_delta*(PresProf[ai]/hitran_P00); # unclear if it should be Pi/P00
        hitran_gammal = hitran_gammal0*(PresProf[ai]/hitran_P00)*((hitran_T00/TempProf[ai])**hitran_alpha);    # Calculate Lorentz lineweidth at P(i) and T(i)
        hitran_gammad = (hitran_nu0)*((2.0*kB*TempProf[ai]*np.log(2.0))/(Mh2o*c**2))**(0.5);  # Calculate HWHM Doppler linewidth at T(i)                                        ^
        
        # term 1 in the Voigt profile
#        voigt_y = (hitran_gammal/hitran_gammad)*((np.log(2.0))**(0.5));
        voigt_x_on = ((wn_nuL-hitran_nu0)/hitran_gammad)*(np.log(2.0))**(0.5);
    
        # setting up Voigt convolution
#        voigt_t = np.arange(-np.shape(hitran_line_indices)[0]/2.0,np.shape(hitran_line_indices)[0]/2); # set up the integration spectral step size
    
        voigt_f_t = np.zeros((np.size(hitran_line_indices),np.size(wn_nu)));
        for bi in range(voigt_x_on.size):
            voigt_f_t[bi,:] = voigt(wn_nu-hitran_nu0[bi],hitran_gammad[bi],hitran_gammal[bi]); 
            if freqnorm:
                voigt_f_t[bi,:] = voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:],x=nu);
            else:
                voigt_f_t[bi,:] = voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:]);  # add x=wn_nu to add frequency normalization
    
        # Calculate linestrength at temperature T
        hitran_S = hitran_S0*((hitran_T00/TempProf[ai])**(1.5))*np.exp(1.439*hitran_E*((1.0/hitran_T00)-(1.0/TempProf[ai])));
      
        # Cross section is normalized for spectral integration (no dnu multiplier required)
        voigt_sigmav_f[ai,:] = np.sum(hitran_S[:,np.newaxis]*voigt_f_t,axis=0);  
    
    
    ExtinctionProf = (1e-2)*voigt_sigmav_f;  # convert to m^2
    return ExtinctionProf
    
def get_beta_m_sonde(Profile,Years,Months,Days,sonde_basepath,interp=False):
    """
    Returns a 2D array containing the expected molecular backscatter component
    
    StartDateTime - initial data set time (where time=0 starts)
        set by datetime.datetime(Years[0],Months[0],Days[0],0)
    
    StopDateTime - only needed to make sure and get all the right sonde files
        set by datetime.datetime(Years[-1],Months[-1],Days[-1],0)  
    
    """
    
    # brute force step through and load each month's data
    if np.unique(Years).size==1 and np.unique(Months).size == 1:
        # if we know for sure the data is confined to one month
        Num_Sonde_Iterations = 1
    else:
        Num_Sonde_Iterations = Years.size
    
    for ai in range(Num_Sonde_Iterations):
        if ai == 0:
            YearStr = str(Years[ai])
            if Months[ai] < 10:
                MonthStr = '0'+str(Months[ai])
            else:
                MonthStr = str(Months[ai])
            ### Grab Sonde Data
            sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
            
            print('Accessing %s' %sondefilename)
            #(Man or SigT)
#            f = netcdf.netcdf_file(sondefilename, 'r')
            f = nc4.Dataset(sondefilename,'r')
            TempDat = f.variables['tpSigT'][:].copy()  # Kelvin
            PresDat = f.variables['prSigT'][:].copy()*100.0  # hPa - convert to Pa (or Man or SigT)
            SondeTime = f.variables['relTime'][:].copy() # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
            SondeAlt = f.variables['htSigT'][:].copy()  # geopotential altitude in m
            StatElev = f.variables['staElev'][:].copy()  # launch elevation in m
            f.close()
            
        elif Months[ai-1] != Months[ai]:
            YearStr = str(Years[ai])
            if Months[ai] < 10:
                MonthStr = '0'+str(Months[ai])
            else:
                MonthStr = str(Months[ai])
                
            ### Grab Sonde Data
            sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
            #(Man or SigT)
#            f = netcdf.netcdf_file(sondefilename, 'r')
            f = nc4.Dataset(sondefilename,'r')
            TempDat = np.hstack((TempDat,f.variables['tpSigT'][:].copy()))  # Kelvin
            PresDat = np.hstack((f.variables['prSigT'][:].copy()*100.0))  # hPa - convert to Pa (or Man or SigT)
            SondeTime = np.concatenate((SondeTime,f.variables['relTime'][:].copy())) # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
            SondeAlt = np.hstack((SondeAlt,f.variables['htSigT'][:].copy()))  # geopotential altitude in m
            StatElev = np.concatenate((StatElev,f.variables['staElev'][:].copy()))  # launch elevation in m
            f.close()
        
        
    # set unrealistic sonde data to nans    
    TempDat[np.nonzero(np.logical_or(TempDat < 173.0, TempDat > 373.0))] = np.nan;
    PresDat[np.nonzero(np.logical_or(PresDat < 1.0*100, PresDat > 1500.0*100))] = np.nan;
   
   
    # get sonde time format into the profile time reference    
    StartDateTime = datetime.datetime(Years[0],Months[0],Days[0],0)
    sonde_datetime0 = datetime.datetime(1970,1,1,0,0)
    sonde_datetime = []
    tref = np.zeros(SondeTime.size)
    for ai in range(SondeTime.size):
        # obtain sonde date/time in datetime format
        sonde_datetime.extend([sonde_datetime0+datetime.timedelta(SondeTime[ai]/(3600*24))])
        # calculate the sonde launch time in profile time
        tref[ai] =  (sonde_datetime[ai]-StartDateTime).total_seconds()
        
    # find the sonde index that best matches the time of the profile
    sonde_index_prof = np.argmin(np.abs(Profile.time[np.newaxis,:]-tref[:,np.newaxis]),axis=1)  # profile index for a given sonde launch
    sonde_index = np.argmin(np.abs(Profile.time[np.newaxis,:]-tref[:,np.newaxis]),axis=0)  # sonde index for a given profile
    sonde_index_u = np.unique(sonde_index)  # unique list of sonde launches used to build the profle
    
    
    beta_m_sonde = np.zeros(Profile.profile.shape)   
    
    if interp:
        # if possible, add additional endpoints to ensure interpolation
        if np.min(sonde_index_u) > 0:
            sonde_index_u = np.concatenate((np.array([np.min(sonde_index_u)-1]),sonde_index_u))
        if np.max(sonde_index_u) < sonde_index.size:
            sonde_index_u = np.concatenate((sonde_index_u,np.array([np.max(sonde_index_u)+1])))
            
        Tsonde = np.zeros((sonde_index_u.size,Profile.range_array.size))
        Psonde = np.zeros((sonde_index_u.size,Profile.range_array.size))
        
        for ai in range(sonde_index_u.size):
            Tsonde[ai,:] = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],TempDat[sonde_index_u[ai],:])
            Psonde[ai,:] = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],PresDat[sonde_index_u[ai],:])
            
        for ai in range(Profile.range_array.size):
            TsondeR = np.interp(Profile.time,tref[sonde_index_u],Tsonde[:,ai])
            PsondeR = np.interp(Profile.time,tref[sonde_index_u],Psonde[:,ai])
            beta_m_sonde[:,ai] = 5.45*(550.0e-9/Profile.wavelength)**4*1e-32*PsondeR/(TsondeR*kB)
    else:
    
        for ai in range(sonde_index_u.size):
            Tsonde = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],TempDat[sonde_index_u[ai],:])
            Psonde = np.interp(Profile.range_array,SondeAlt[sonde_index_u[ai],:]-StatElev[sonde_index_u[ai]],PresDat[sonde_index_u[ai],:])
            beta_m_sondes0 = 5.45*(550.0e-9/Profile.wavelength)**4*1e-32*Psonde/(Tsonde*kB)
            ifill = np.nonzero(sonde_index==sonde_index_u[ai])[0]
            beta_m_sonde[ifill,:] = beta_m_sondes0[np.newaxis,:]*np.ones((ifill.size,beta_m_sondes0.size))
    beta_mol = Profile.copy()
    beta_mol.profile = beta_m_sonde
    beta_mol.profile_variance = (beta_mol.profile*0.01)**2  # force SNR of 100 in sonde profile.
    beta_mol.ProcessingStatus = []     # status of highest level of lidar profile - updates at each processing stage

    beta_mol.diff_geo_Refs = ['none']           # list containing the differential geo overlap reference sources (answers: differential to what?)
    beta_mol.profile_type =  'Sonde Estimated Molecular Backscatter Coefficient [$m^{-1}sr^{-1}$]'
    
    beta_mol.bg = np.zeros(beta_mol.bg.shape) # profile background levels
    
    beta_mol.descript = 'Sonde Estimated Molecular Backscatter Coefficient in m^-1 sr^-1'
    beta_mol.label = 'Molecular Backscatter Coefficient'

    return beta_mol,tref,sonde_index_u
    
    #beta_m_sondes0 = 5.45*(550.0/Molecular.wavelength)**4*1e-32*PresDat[sonde_index_u,:]/(TempDat[sonde_index_u,:]*lp.kB)
    
#    pres_func = scipy.interpolate.interp2d(tref[sonde_index_u,np.newaxis]*np.ones(SondeAlt[sonde_index_u,:].shape),SondeAlt[sonde_index_u,:]-StatElev[sonde_index_u,np.newaxis],np.log10(PresDat[sonde_index_u,:]))    
#    temp_func = scipy.interpolate.interp2d(tref[sonde_index_u,np.newaxis]*np.ones(SondeAlt[sonde_index_u,:].shape),SondeAlt[sonde_index_u,:]-StatElev[sonde_index_u,np.newaxis],TempDat[sonde_index_u,:])  
#    #beta_m_func = scipy.interpolate.interp2d(tref[sonde_index_u,np.newaxis]*np.ones(SondeAlt[sonde_index_u,:].shape),SondeAlt[sonde_index_u,:]-StatElev[sonde_index_u,np.newaxis],beta_m_sondes0)    
#    
#    rr,tt = np.meshgrid(Profile.range_array,Profile.time)
#    beta_m_sonde_func = 5.45*(550.0/Molecular.wavelength)**4*1e-32*10**(pres_func(tt.flatten(),rr.flatten()))/(temp_func(tt.flatten(),rr.flatten())*lp.kB)
    
    
    
##    sonde_index = np.min([np.shape(SondeAlt)[0]-1,sonde_index])
#    Tsonde = np.interp(CombHi.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
#    Psonde = np.unique(isonde)
#    # Obtain sonde data for backscatter coefficient estimation
#    Tsonde = np.interp(CombHi.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
#    Psonde = np.interp(CombHi.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],PresDat[sonde_index,:])
    
    
    
#    # note the operating wavelength of the lidar is 532 nm
#    beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*lp.kB)