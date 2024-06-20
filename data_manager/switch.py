"""
See PAX [manual](https://www.manualslib.com/manual/1634678/Thorlabs-Pax1000.html?page=39#manual) for details
- *TimeElapsed*: time elapsed [seconds]
- *s1, s2, s3*: Stokes params normalized to S0
- *S0, S1, S2, S3*: raw Stokes params [mW]
- *Azumith, Ellipticity*: Poincare sphere angles [degrees]
- *DOP*: degree of polarization [%]
- *DOCP*: degree of circular polarization [%]
- *DOLP*: degree of linear polarization [%]
- *Power*: total power [mW]
- *Power_pol*: polarized power [mW]
- *Power_unpol*: unpolarized power [mW]
- *Power_split_ratio*: see manual
- *Phase_difference*: see manual
- *Timestamp*: DateTime Timestamp (hh:mm:ss.µss)
"""

BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
from .dataset import Dataset
import data_manager.utils.math_utils as math_utils
import data_manager.utils.dataframe_management_utils as dataframe_management_utils
import data_manager.utils.switch_detection_utils as switch_detection_utils
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import allantools

class SwitchSet(Dataset):
    def __init__(self, filename, set_range=None, time_offset=0.0, skip_default_signal_baseline=True):
        super().__init__(filename, set_range, time_offset, skip_default_signal_baseline)
        
        self.change_point_params = []
        self.switch_offset = None
        self.switch_time = None
        self.signal_1_df = None
        self.signal_2_df = None
        self.stokes_ptf_df = None
        self.input_polarization = None
        self.transfer_func_df = None
        self.assign_by = None
        self.angle_threshold_deg = None
        self.reset_times = None
        self.reset_by_rolling = None
        self.signal_1_input_stokes = None
        self.signal_2_input_stokes = None
        #self.metrics=['dist','angle']

    ### Print metainfo obtained from the csv file
    def print_info(self):
        print("=== PAX Configuration parameters ===")
        print("Device ID: {:s}".format(self.device_id))
        print("Serial number: {:s}".format(self.serial_num))
        print("Wavelength (nm): {:.2f}".format(self.wavelength))
        print("Basic Sample Rate (Hz): {:.1f}".format(self.basic_sample_rate))
        print("Operating Mode Period (# of rotations): {:.1f}".format(self.op_mode_period))
        print("Operating Mode FFT Points: {:d}".format(self.op_mode_FFT_num))
        print("--> Nominal Sample Rate (Hz): {:.2f}".format(self.nominal_sample_rate))
        print("\n=== Dataset information ===")
        print("Filename: {:s}".format(self.filename))
        print("Time range read (seconds): min={:.2f}, max={:.2f}".format(self.mintime, self.maxtime))
        print("Average Sample Rate (Hz): {:.2f}".format(self.avg_sample_rate))
    
    ### Plots specified parameter over time (from the raw data)
    # birds_eye: plot the entirety of the avaliable data?
    # plot_param: choose from s1,s2,s3,S0,S1,S2,S3,Azimuth,Ellipticity,Power,DOP,...
    # sample_range: used to zoom in on a particular time range, e.g. (2000,2050) seconds
    # time_offset: used by SetPair (allows offsetting of plot by this constant; purely for plotting, no functional purpose)
    def plot_raw(self,birds_eye=True,plot_param='s1',sample_range=None,plot_switch=False,plot_avg=False,time_offset=0.0):
        # sample_range should be of the form (sample_start, sample_end) if a smaller range is desired
        # if sample_start or sample_end are None themselves, they will be filled in
        if sample_range is not None:
            sample_start, sample_end = dataframe_management_utils.fill_in_range(sample_range, self.df)
            sample_range = (sample_start+time_offset, sample_end+time_offset)   # Make sure we include the offset
        
        # Plot entire dataset if requested
        if birds_eye:
            BE_fig, BE_ax = plt.subplots(figsize=(12,3))
            BE_ax.plot(self.df['TimeElapsed'], dataframe_management_utils.transfer_value_units(self.df[plot_param]), label=plot_param, linewidth=0.5, marker='o', markersize=0.8, color='red', alpha=0.5)
            BE_ax.set_xlabel('Time [s]')
            BE_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, dataframe_management_utils.gen_time_str(self.df)), fontsize=14, fontweight='bold')
            BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,dataframe_management_utils.get_param_units(plot_param,'TODO')))
            BE_ax.grid(True)
            BE_ax.legend(loc='upper right')
            # We add green lines to the birds eye plot to denote where sample_range is located
            if sample_range is not None:
                for val in sample_range:
                    BE_ax.axvline(val, color = 'green', linewidth=2)
            display(BE_fig); plt.close(BE_fig)   # Show entire dataset
        
        # Plot smaller sample range if requested
        if sample_range is not None:
            ZI_fig, ZI_ax = plt.subplots(figsize=(12,3))
            ZI_ax.set_xlim(sample_range[0],sample_range[1])
            ZI_ax.plot(self.df['TimeElapsed'], dataframe_management_utils.transfer_value_units(self.df[plot_param]), label=plot_param, linewidth=1, marker='o', markersize=1.5, color='red')
            ZI_ax.set_xlabel('Time [s]')
            ZI_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, dataframe_management_utils.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
            ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,dataframe_management_utils.get_param_units(plot_param,'TODO')))
            ZI_ax.grid(True)
            ZI_ax.legend(loc='upper right')
            # If change times have been detected, we can plot that extra information
            if self.change_point_params != []:
                # If requested, plot the estimated switches on the time series data (plot switches on entire dataset)
                if plot_switch:
                    num_switch_to_start_on = np.floor((sample_range[0] - self.switch_offset)/self.switch_time)
                    plot_time = num_switch_to_start_on * self.switch_time + self.switch_offset
                    while plot_time < sample_range[1]:
                        ZI_ax.axvline(plot_time, color = 'purple', linewidth=0.8)
                        plot_time += self.switch_time

                # If requested, plot the detected jumps that were used to determine the switch times
                if plot_switch:   # Replaced plot_jumps with plot_switch
                    for i in range(len(self.df)):
                        if self.df.at[i,'IsJump']:
                            X = self.df.loc[i-1:i,'TimeElapsed']
                            Y = self.df.loc[i-1:i,plot_param]
                            ZI_ax.plot(X,Y, linewidth=1, marker='o', markersize=1.5, color='orange')

                # TODO: doing this by groupby may be more efficient/cleaner
                # If requested, overwrite the included points with the color blue
                # Assumes all valid sections start with an IsStartPoint=True and end with an IsEndPoint=True
                if plot_switch:   # Replaced plot_valid with plot_switch
                    start_index = self.df.index[self.df['TimeElapsed'] >= sample_range[0]].min()   # Start point where sample range starts
                    end_index = self.df.index[self.df['TimeElapsed'] <= sample_range[1]].max()   # End point where sample range ends
                    start_index -= 20   # Give some wiggle room for start of a valid section
                    end_index += 20
                    start_index = max(start_index,0)   # In case i is negative, start at beginning
                    end_index = min(end_index,len(self.df)-1)
                    X = []; Y = []
                    i = start_index
                    while i <= end_index:
                        if self.df.at[i,'IsStartPoint']:   # We reach the start of a valid section
                            while True:   # Iterate over the valid section, adding all valid points
                                X.append(self.df.at[i,'TimeElapsed'])
                                Y.append(self.df.at[i,plot_param])
                                i+=1
                                if self.df.at[i-1,'IsEndPoint']:   # If the point just added was an end point, break
                                    break
                            ZI_ax.plot(X,Y, linewidth=1, marker='o', markersize=1.5, color='blue')   # Plot the valid section
                            X = []; Y = []
                        else:
                            i+=1

                # If requested and if the data has been averaged, we will plot that overtop the raw data
                if (self.signal_1_df is not None) & plot_avg:
                    #TODO: plot interpolated points in a different color
                    plot_avg_std=False
                    plot_param_avg_str = plot_param+"Avg"
                    if plot_avg_std:
                        plot_param_std_str = plot_param+"Std"
                        ZI_ax.errorbar(self.signal_1_df['EstTime'],self.signal_1_df[plot_param_avg_str],yerr=self.signal_1_df[plot_param_std_str],label=plot_param+" 1",linewidth=1,marker='o',markersize=3,color='orange')
                        ZI_ax.errorbar(self.signal_2_df['EstTime'],self.signal_2_df[plot_param_avg_str],yerr=self.signal_2_df[plot_param_std_str],label=plot_param+" 2",linewidth=1,marker='o',markersize=3,color='green')
                    else:
                        ZI_ax.plot(self.signal_1_df['EstTime'],dataframe_management_utils.transfer_value_units(self.signal_1_df[plot_param_avg_str]),label=plot_param+" 1",linewidth=1,marker='o',markersize=3,color='orange')
                        ZI_ax.plot(self.signal_2_df['EstTime'],dataframe_management_utils.transfer_value_units(self.signal_2_df[plot_param_avg_str]),label=plot_param+" 2",linewidth=1,marker='o',markersize=3,color='green')

            display(ZI_fig); plt.close(ZI_fig)
        #return BE_fig, ZI_fig
        #return ZI_fig
        return
    
    ### Plots specified parameter(s) over time (from the separated and averaged data) for *both* separated signals
    # birds_eye: plot the entirety of the avaliable data?
    # plot_param: choose from s1,s2,s3,S0,S1,S2,S3,Azimuth,Ellipticity,Power,DOP,... (but add 'Avg' to the end)
    # plot_param_2: allows plotting of two parameters
    # sample_range: used to zoom in on a particular time range, e.g. (2000,2050) seconds
    # plot_signal: we have two separated signals; which do we want to show? 1, 2, None (both)
    # time_offset: used by SetPair (allows offsetting of plot by this constant; purely for plotting, no functional purpose)
    # plot_rolling: if plot_param=='rotAngle', this will determine whether to plot the rolling average of rotAngle; default is True
    # enforce_ylim: sometimes for noisy signals the ylim can get messed up; when set to True, this will automatically make the upper ylim 20% above the reset threshold
    def plot_separated(self,birds_eye=True,plot_param='s1Avg',plot_param_2=None,sample_range=None,plot_signal=None,time_offset=0.0,plot_rolling=True,enforce_ylim=False):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return
        for p in (plot_param, plot_param_2):
            if p is not None and p not in self.signal_1_df.columns:
                if self.stokes_ptf_df is not None and p not in self.stokes_ptf_df.columns:
                    raise KeyError('Parameter \'{:s}\' does not exist'.format(p))
                if self.stokes_ptf_df is None:
                    raise KeyError('ptf not yet calculated! (or parameter \'{:s}\' doesn exist)'.format(p))
        
        # sample_range should be of the form (sample_start, sample_end) if a smaller range is desired
        # if sample_start or sample_end are None themselves, they will be filled in
        if sample_range is not None:
            min_est_time = min(self.signal_1_df.loc[0,'EstTime'],self.signal_2_df.loc[0,'EstTime'])   # There are two signal arrays
            sample_start = min_est_time if sample_range[0] is None else sample_range[0]
            max_est_time = max(self.signal_1_df.loc[self.signal_1_df.shape[0]-1,'EstTime'],\
                               self.signal_2_df.loc[self.signal_2_df.shape[0]-1,'EstTime'])
            sample_end = max_est_time if sample_range[1] is None else sample_range[1]
            sample_range = (sample_start+time_offset, sample_end+time_offset)   # Make sure we include the offset
        
        # Plot entire dataset if specified
        alpha=0.5 # Temporary, for getting good looking plots for 407 report; can be switched back to 1.0
        markersize=0.5 # was 0.8
        linestyle_1='-'
        linestyle_2='--'
        linewidth=0.6
        marker=''
        if birds_eye:
            # Birds Eye plot
            BE_fig, BE_ax = plt.subplots(figsize=(12,3))
            custom_palette = sns.color_palette("dark")
            # Only plot specified signal(s)
            if plot_param in self.signal_1_df.columns:   # Plot from separated signals
                if plot_signal is None or plot_signal == 1:
                    BE_ax.plot(self.signal_1_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_1_df[plot_param]), label=plot_param+" 1", \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[0])
                if plot_signal is None or plot_signal == 2:
                    BE_ax.plot(self.signal_2_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_2_df[plot_param]), label=plot_param+" 2", \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[1])
            else:   # Plot from ptf
                BE_ax.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df[plot_param]), label=plot_param, \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[0])
                if plot_param=='rotAngle' and plot_rolling:   # If plotting rotAngle, also plot rolling average
                    BE_ax.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[2])
                if self.angle_threshold_deg is not None:
                    BE_ax.axhline(y=dataframe_management_utils.transfer_value_units(self.angle_threshold_deg), color='red', linewidth=1)
                    if enforce_ylim: BE_ax.set_ylim(-dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*0.2, dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*1.2)
                if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            BE_ax.axvline(time, color = 'red', alpha=0.5, linewidth=0.5)
            BE_ax.set_xlabel('Time [s]')
            BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,dataframe_management_utils.get_param_units(plot_param,'TODO')))
            BE_ax.grid(True)
            # If requested, plot secondary parameter
            if plot_param_2 is not None:
                BE_ax2 = BE_ax.twinx()
                if plot_param_2 in self.signal_1_df.columns:
                    if plot_signal is None or plot_signal == 1:
                        BE_ax2.plot(self.signal_1_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_1_df[plot_param_2]), label=plot_param_2+" 1", \
                                    alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[9])
                    if plot_signal is None or plot_signal == 2:
                        BE_ax2.plot(self.signal_2_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_2_df[plot_param_2]), label=plot_param_2+" 2", \
                                    alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[8])
                else: # Plot from ptf
                    BE_ax2.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df[plot_param_2]), label=plot_param_2, \
                                alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[9])
                    if plot_param_2=='rotAngle' and plot_rolling:   # If plotting rotAngle, also plot rolling average
                        BE_ax2.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                                   alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[7])
                    if self.angle_threshold_deg is not None:
                        BE_ax2.axhline(y=dataframe_management_utils.transfer_value_units(self.angle_threshold_deg), color='red', linewidth=1)
                        if enforce_ylim: BE_ax2.set_ylim(-dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*0.2, dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*1.2)
                    if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            BE_ax2.axvline(time, color = 'red', alpha=0.5, linewidth=0.5)
                BE_ax2.set_ylabel('{:s} [{:s}]'.format(plot_param_2,dataframe_management_utils.get_param_units(plot_param_2,'TODO')))
                # lines code is just to get legend working (ChatGPT generated)
                lines1, labels1 = BE_ax.get_legend_handles_labels()
                lines2, labels2 = BE_ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                BE_ax.set_title('{:s} & {:s} (separated) | {:s} | {:s}'.format(plot_param, plot_param_2, self.title, dataframe_management_utils.gen_time_str(self.df)), fontsize=14, fontweight='bold')
                BE_ax.legend(lines, labels, loc='upper left')
            else:
                BE_ax.set_title('{:s} (separated) | {:s} | {:s}'.format(plot_param, self.title, dataframe_management_utils.gen_time_str(self.df)), fontsize=14, fontweight='bold')
                BE_ax.legend(loc='upper left')
            # We add green lines to the birds eye plot to denote where sample_range is located
            if sample_range is not None:
                for val in sample_range:
                    BE_ax.axvline(val, color = 'green', linewidth=1)
            display(BE_fig); plt.close(BE_fig)   # Show entire dataset
        
        # If requested, we'll also plot the smaller sample range
        alpha=0.5 # Temporary, for getting good looking plots for 407 report; can be switched back to 1.0
        markersize=0.5 # was 0.8
        linestyle_1='-'
        linestyle_2='--'
        linewidth=0.6
        marker=''
        if sample_range is not None:
            # Zoomed In plot 
            ZI_fig, ZI_ax = plt.subplots(figsize=(12,3))
            # Only plot specified signal(s)
            if plot_param in self.signal_1_df.columns:   # Plot from separated signals
                if plot_signal is None or plot_signal == 1:
                    ZI_ax.plot(self.signal_1_df['EstTime'],dataframe_management_utils.transfer_value_units(self.signal_1_df[plot_param]),label=plot_param+" 1", \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[0])
                if plot_signal is None or plot_signal == 2:
                    ZI_ax.plot(self.signal_2_df['EstTime'],dataframe_management_utils.transfer_value_units(self.signal_2_df[plot_param]),label=plot_param+" 2", \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[1])
            else:   # Plot from ptf
                ZI_ax.plot(self.stokes_ptf_df['EstTime'],dataframe_management_utils.transfer_value_units(self.stokes_ptf_df[plot_param]),label=plot_param, \
                           alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[0])
                if plot_param=='rotAngle' and plot_rolling:   # If plotting rotAngle, also plot rolling average
                    ZI_ax.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               alpha=alpha, linestyle=linestyle_1, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[2])
                if self.angle_threshold_deg is not None:
                    ZI_ax.axhline(y=dataframe_management_utils.transfer_value_units(self.angle_threshold_deg), color='red', linewidth=1)
                    if enforce_ylim: ZI_ax.set_ylim(-dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*0.2, dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*1.2)
                if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            if sample_range[0] <= time <= sample_range[1]:
                                ZI_ax.axvline(time, color = 'red', alpha=0.5, linewidth=0.5)
            ZI_ax.set_xlabel('Time [s]')
            ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,dataframe_management_utils.get_param_units(plot_param,'TODO')))
            ZI_ax.grid(True)
            # If requested, plot secondary parameter
            if plot_param_2 is not None:
                ZI_ax2 = ZI_ax.twinx()
                if plot_param_2 in self.signal_1_df.columns:   # Plot from separated signals
                    if plot_signal is None or plot_signal == 1:
                        ZI_ax2.plot(self.signal_1_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_1_df[plot_param_2]), label=plot_param_2+" 1", \
                                    alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[9])
                    if plot_signal is None or plot_signal == 2:
                        ZI_ax2.plot(self.signal_2_df['EstTime'], dataframe_management_utils.transfer_value_units(self.signal_2_df[plot_param_2]), label=plot_param_2+" 2", \
                                    alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[8])
                else:   # Plot from ptf
                    ZI_ax2.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df[plot_param_2]), label=plot_param_2, \
                                alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[9])
                    if plot_param_2=='rotAngle' and plot_rolling:   # If plotting rotAngle, also plot rolling average
                        ZI_ax2.plot(self.stokes_ptf_df['EstTime'], dataframe_management_utils.transfer_value_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               alpha=alpha, linestyle=linestyle_2, linewidth=linewidth, marker=marker, markersize=markersize, color=custom_palette[7])
                    if self.angle_threshold_deg is not None:
                        ZI_ax2.axhline(y=dataframe_management_utils.transfer_value_units(self.angle_threshold_deg), color='red', linewidth=1)
                        if enforce_ylim: ZI_ax2.set_ylim(-dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*0.2, dataframe_management_utils.transfer_value_units(self.angle_threshold_deg)*1.2)
                    if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            if sample_range[0] <= time <= sample_range[1]:
                                ZI_ax2.axvline(time, color = 'red', alpha=0.5, linewidth=0.5)
                ZI_ax2.set_ylabel('{:s} [{:s}]'.format(plot_param_2,dataframe_management_utils.get_param_units(plot_param_2,'TODO')))
                # lines code is just to get legend working (ChatGPT generated)
                lines1, labels1 = ZI_ax.get_legend_handles_labels()
                lines2, labels2 = ZI_ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                ZI_ax.set_title('{:s} & {:s} (separated) | {:s} | {:s}'.format(plot_param, plot_param_2, self.title, dataframe_management_utils.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
                ZI_ax.legend(lines, labels, loc='upper left')
            else:
                ZI_ax.set_title('{:s} (separated) | {:s} | {:s}'.format(plot_param, self.title, dataframe_management_utils.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
                ZI_ax.legend(loc='upper left')
            #TODO: standard deviation values
            #TODO: ZI_ax.errorbar(self.signal_1_df['EstTime'],self.signal_1_df[plot_param],yerr=self.signal_1_df[plot_param_std_str],label=plot_param+" 1",linewidth=1,marker='o',markersize=1.5,color='orange')
            #TODO: ZI_ax.errorbar(self.signal_2_df['EstTime'],self.signal_2_df[plot_param],yerr=self.signal_2_df[plot_param_std_str],label=plot_param+" 2",linewidth=1,marker='o',markersize=1.5,color='purple')
            ZI_ax.set_xlim(sample_range[0],sample_range[1])
            display(ZI_fig); plt.close(ZI_fig)
        
        #return BE_fig, ZI_fig
        return
    
    
    ### Given the two change point ranges spaced >3000s apart (two to account for recording drift over 1000's of seconds),
    ### this function computes the time offset of the switches and marks down such information
    ### so it can be used by average_data, plotting functions, and future functions
    ### Later results will be more precise if the first change point range is close to the front of the dataset
    def find_switches(self, nominal_switch_rate=2, change_point_range_1=None, change_point_range_2=None, time_offset=0.0, n_exclude=3, print_process=False):
        # Logic: Ensure change point hasn't already been calculated
        if self.change_point_params!=[]:
            # TODO: allow to recalculate change points based on new change point range
            print(BOLD_ON + 'Error: change times already detected!' + BOLD_OFF)
            return
        self.nominal_switch_time = 1/(2*nominal_switch_rate)   # Time between switches
        self.df.insert(self.df.columns.get_loc('TimeDiff')+1, 'IsJump', [False]*len(self.df))
        
        # Logic: If desired, determine cp ranges automatically
        # Should not be used alongside time_offset
        if change_point_range_1 is None:
            if print_process: print('Determining change point ranges automatically...')
            change_point_range_1 = (self.mintime, self.mintime+100)
            if self.maxtime < self.mintime + 200:
                change_point_range_2=None
            else:
                if self.maxtime > self.mintime + 3100:
                    change_point_range_2 = (self.mintime+3000, self.mintime+3100)
                else:
                    change_point_range_2 = (self.maxtime-100, self.maxtime)

        ### Perform change point detection at the beginning of the dataset to find the offset of the switches
        ### Uses util function switch_detection_utils.change_point(...); most of the math is located here
        if print_process: print('Using change_point_range_1={}\nUsing change_point_range_2={}'.format(change_point_range_1,change_point_range_2))
        change_point_range_1 = (change_point_range_1[0]+time_offset, change_point_range_1[1]+time_offset)   # Include time_offset
        change_point_range_1, change_point_df_1, points_skipped_1, switch_param_1, mean_offset_1, left_edge_1, right_edge_1, jump_fig_1 = switch_detection_utils.change_point(self.df,self.nominal_switch_time,change_point_range_1,n_exclude,print_process)
        self.change_point_params.append((change_point_range_1, change_point_df_1, points_skipped_1, switch_param_1, mean_offset_1, left_edge_1, right_edge_1, jump_fig_1))
        #self.switch_offset = mean_offset_1
        
        # If we're dealing with an extra long dataset, the switch time is liable to drift, so we
        # perform change point detection a second time ~3000 seconds (~an hour) later and correct
        # the switch time
        if change_point_range_2 is not None:
            change_point_range_2 = (change_point_range_2[0]+time_offset, change_point_range_2[1]+time_offset)   # Include time_offse
            change_point_range_2, change_point_df_2, points_skipped_2, switch_param_2, mean_offset_2, left_edge_2, right_edge_2, jump_fig_2 = switch_detection_utils.change_point(self.df,self.nominal_switch_time,change_point_range_2,n_exclude,print_process)
            self.change_point_params.append((change_point_range_2, change_point_df_2, points_skipped_2, switch_param_2, mean_offset_2, left_edge_2, right_edge_2, jump_fig_2))

            ### Takes the calculated offset at *the beginning of* and *1 hour into* the dataset, and identifies the *actual* time between switches, which is likely
            ### off from the nominal value by some amount on the order of a fraction of a percent (only has to be accounted for for multi-hour datasets)
            actual_switch_time, actual_switch_offset = switch_detection_utils.calc_actual_switch_time(mean_offset_1, change_point_range_1, mean_offset_2, change_point_range_2, self.nominal_switch_time, print_process)
            self.switch_time = actual_switch_time
            self.switch_offset = actual_switch_offset
        else:
            self.switch_time = self.nominal_switch_time   # If second range not provided, don't correct switch time
            self.switch_offset = mean_offset_1   # There is nothing in the offset to correct for
        
        
        ### Now we can confidently place the time of each switch between the polarization signals
        ### We find which dataset samples correspond to those switches, and mark them for later use
        t_min = self.df.at[0,'TimeElapsed']; t_max = self.df.at[self.df.shape[0]-1,'TimeElapsed']
        num_switch_to_start_on = np.ceil((t_min - self.switch_offset)/self.switch_time)
        time_to_start_on = num_switch_to_start_on * self.switch_time + self.switch_offset
        switch_array = np.arange(time_to_start_on, t_max, self.switch_time)   # Array of estimated switch times along dataset
        switch_indices = np.searchsorted(self.df['TimeElapsed'].tolist(), switch_array)   # Samples in dataset which occur right after a switch
        #value = [idx in switch_indices for idx in self.df.index]
        self.df.insert(self.df.columns.get_loc('IsJump')+1, 'IsSwitch', self.df.index.isin(switch_indices))   # These samples are marked as switch points
        
        ### We exclude samples which are too close to a switch, as they could be outliers or limbo samples
        ### (the PAX has a tendency to produce these around the switches)
        ### The number of excluded samples controlled by parameter n (default is n=4, usually more than enough)
        # IsStartPoint and IsEndPoint mark the start and end of 'valid' (non-excluded) sections of data samples
        # (later separated into the two signals)
        self.df.insert(self.df.columns.get_loc('IsSwitch')+1, 'IsValidPoint', [True]*len(self.df))   # Assume all samples valid; exclude as we go
        self.df.insert(self.df.columns.get_loc('IsValidPoint')+1, 'IsStartPoint', [False]*len(self.df))
        self.df.insert(self.df.columns.get_loc('IsStartPoint')+1, 'IsEndPoint', [False]*len(self.df))
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'Azimuth']].head(10))
        
        # Iterate over (most of) dataset; when we encounter a switch point, we'll exclude n samples to the left&right
        # Valid segments may not go off the edge of the dataset (we won't include them)
        for i in range(n_exclude+1, len(self.df)-n_exclude-1):
            #print('i={:d}: IsSwitch={:s}'.format(i, str(self.df.at[i, 'IsSwitch'])))
            if self.df.at[i, 'IsSwitch']:
                # We exclude n samples to the left of i and n-1 samples to the right of i
                # since the switch point is the sample after the estimated switch time
                exclude_left_edge = i-n_exclude; exclude_right_edge = i+n_exclude-1   # Left and right edges of the exclude range
                self.df.loc[exclude_left_edge:exclude_right_edge, 'IsValidPoint'] = False   # Exclude samples in range
                self.df.loc[exclude_left_edge-1, 'IsEndPoint'] = True   # Mark the end of the previous valid range
                self.df.loc[exclude_right_edge+1, 'IsStartPoint'] = True   # Mark the start of the next valid range
        
        # Be sure to exclude the first and last few samples which weren't checked over in the previous code block
        # We also ensure each valid segment starts and ends with an IsStartPoint and an IsEndPoint, respectively
        first_start_ind = self.df.index[self.df['IsStartPoint']].min()
        last_start_ind = self.df.index[self.df['IsStartPoint']].max()
        first_end_ind = self.df.index[self.df['IsEndPoint']].min()
        last_end_ind = self.df.index[self.df['IsEndPoint']].max()
        if first_end_ind < first_start_ind:   # If first end point preceeds first start point
            self.df.at[first_end_ind,'IsEndPoint'] = False   # Get rid of first end point
            self.df.loc[0:first_end_ind, 'IsValidPoint'] = False   # Get rid of first valid points
        if last_start_ind > last_end_ind:   # If last start point succeeds last end point
            self.df.at[last_start_ind,'IsStartPoint'] = False   # Get rid of last start point
            self.df.loc[last_start_ind:len(self.df)-1, 'IsValidPoint'] = False   # Get rid of last valid samples
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'IsStartPoint','IsEndPoint']].head(20))
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'IsStartPoint','IsEndPoint']].tail(20))
        
        #return jump_fig_1, jump_fig_2
        return
    
    ### This is a user function used to take the valid sections found within find_switches,
    ### and separate the data into the two orthogonal polarization signals, averaging the data within each valid section
    def average_data(self, assign_by=None, print_process=False, print_assignment=False):
        if self.change_point_params == []:
            print(BOLD_ON + 'Error: change times not yet detected' + BOLD_OFF)
            return
        if self.signal_1_df is not None:
            # TODO: if change point has changed, recalculate averages
            print(BOLD_ON + 'Error: averaging already performed!' + BOLD_OFF)
            return
        
        if print_process: print(BOLD_ON+'=== starting average_data ==='+BOLD_OFF)
        # Create a mask for identifying contiguous sections of 'True' or 'False' values
        # 'mask' is the cumulative sum of changes in 'IsValid' values; if 'IsValid' changes value, mask will increment
        # This means every point in a 'True' or a 'False' section will have the same mask value, but mask values
        # will change between sections
        self.df['dif'] = self.df['IsValidPoint'] != self.df['IsValidPoint'].shift()
        self.df['mask'] = (self.df['dif']).cumsum()
        self.df['mask'] = self.df['mask'] - self.df.at[self.df.index[self.df['IsValidPoint']].min(),'mask']   # Ensures first True section has mask value 0
        #display(self.df[['TimeElapsed','IsValidPoint','Azimuth','dif','mask']].head(30))   # Use this line to see dif and mask in df
        
        # df[df['IsValid']] excludes all False values
        # We then group by the mask value to separate 'True' values into different sections (one group for each section)
        # groups.agg() performs aggregation functions, creating a row in avg_df for each group
        # NOTE: this copies the data; changes made to avg_df will not affect self.df
        if print_process: print(BOLD_ON+'averaging data...'+BOLD_OFF,end=' ')
        avg_df = self.df[self.df['IsValidPoint']].groupby(self.df['mask']).agg(
            StartTime=('TimeElapsed', 'first'),
            EndTime=('TimeElapsed', 'last'),
            AvgTime=('TimeElapsed', 'mean'),
            NumPoints=('TimeElapsed', 'size'),
            AzimuthAvg=('Azimuth', 'mean'),
            AzimuthStd=('Azimuth', 'std'),
            EllipticityAvg=('Ellipticity', 'mean'),
            EllipticityStd=('Ellipticity', 'std'),
            s1Avg=('s1', 'mean'),
            s1Std=('s1', 'std'),
            s2Avg=('s2', 'mean'),
            s2Std=('s2', 'std'),
            s3Avg=('s3', 'mean'),
            s3Std=('s3', 'std'),
            S0Avg=('S0', 'mean'),
            S0Std=('S0', 'std'),
            S1Avg=('S1', 'mean'),
            S1Std=('S1', 'std'),
            S2Avg=('S2', 'mean'),
            S2Std=('S2', 'std'),
            S3Avg=('S3', 'mean'),
            S3Std=('S3', 'std'),
            DOPAvg=('DOP', 'mean'),
            DOPStd=('DOP', 'std'),
            PowerAvg=('Power', 'mean'),
            PowerStd=('Power', 'std'))
        if print_process: print(BOLD_ON+'Done'+BOLD_OFF)
        # Now we have a dataframe with one observation per valid interval, with the data averaged within this interval
        # Add some other information:
        avg_df_dropped = avg_df[avg_df['NumPoints'] >= 4]   # Drop an observation if it contains fewer than 4 points
        if print_process: print(BOLD_ON+'Dropped {} observations with <4 points (out of total {} observations)'.format(len(avg_df)-len(avg_df_dropped),len(avg_df))+BOLD_OFF)
        avg_df = avg_df_dropped
        avg_df.insert(avg_df.columns.get_loc('AvgTime')+1, 'Length', np.zeros(len(avg_df),dtype=float))
        avg_df['Length'] = avg_df['EndTime'] - avg_df['StartTime']   # Length (in s) of valid interval
        avg_df.insert(avg_df.columns.get_loc('NumPoints')+1, 'AvgSampleRate', np.zeros(len(avg_df),dtype=float))
        avg_df['AvgSampleRate'] = avg_df['NumPoints'] / avg_df['Length']   # Average sample rate in valid interval
        
        
        ### We need to check for missing observations; if the PAX lags and the delay between two points exceeds the
        ### time between signal switches, we need to fill in that valid section that wasn't recorded
        ### or else the program will get mixed up which signal is which.
        ### To do this, and assign signal signalMembership at the same time, we designate 'odd' and 'even' bands between
        ### switch times. WLOG, signal 1 should be in the odd bands, and signal 2 should be in the even ones
        ### If we find an unnocupied band, we fill it in
        avg_df.reset_index(drop=True,inplace=True)   # Replace mask indices with regular ones to make indexing easier
        
        # Ensure t=0 occurs at the time of the first switch
        # TODO: in change_point(), make this shift and keep it permanent; it makes things easier to think about
        avg_df['AvgTime'] = avg_df['AvgTime'] - self.switch_offset
        
        # This is taking the interval time 'mod' 2*switch time
        # Half these values should be around the switch time, half should be around twice that
        avg_df['modTwiceSwitchTime'] = avg_df['AvgTime'] % (2*self.switch_time)
        avg_df['signalMembership'] = avg_df['modTwiceSwitchTime'] > self.switch_time
        # Uncomment the histogram for a visual
        # Signal 1 is the first group, signal 2 is the second
        # avg_df['modTwiceSwitchTime'].plot.hist(range=(0,2*self.switch_time), log=True, bins=30); plt.show()
        avg_df.drop(columns=['modTwiceSwitchTime'], inplace=True)   # We don't need this anymore, and it isn't really useful information outside of here
        
        # If modTwiceSwitchTime above was the mod, groupNumber is the remainder (but over a 1*switch time length, not 2*)
        # This should basically increase by 1 for each observation (valid interval) and skip over the missing sections
        # (Honestly this is better used to determine signalMembership, where odd/even values are signal 1/2, but I've
        # already coded it and the upper method used to calculate signalMembership is used to show the histogram)
        avg_df['groupNumber'] = np.floor(avg_df['AvgTime'] / (self.switch_time))
        
        # Undo the time shift
        avg_df['AvgTime'] = avg_df['AvgTime'] + self.switch_offset
        
        # We use the group number to calculate the "theoretical" time based on which band the observation falls under
        # (this is so all observations are evenly spaced in time)
        avg_df.insert(avg_df.columns.get_loc('AvgTime')+1, 'EstTime', np.zeros(len(avg_df),dtype=float))
        avg_df['EstTime'] = (avg_df['groupNumber'] * self.switch_time) + self.switch_time/2 + self.switch_offset
        # We should now be using EstTime for everything, instead of AvgTime
        avg_df.drop(columns=['groupNumber'],inplace=True)
        
        ### Finally, we can split the data into two signals and store
        # Keep in mind avg_df contains data from both signals; we split the data based on 'signalMembership',
        # which is True/False for signal 1/2
        grouped_by_signalMembership = avg_df.groupby('signalMembership')
        # We take deep copies of the dataframes so they aren't affected by changes in the original dataset
        signal_dataframes = {signal_number: signal_df.copy() for signal_number, signal_df in grouped_by_signalMembership}
        
        ### Now we find the missing sections and impute (linearly interpolate) the values
        ### For each gap we create and store the necessary rows; at the end, we add them all
        ### to the final dataframe and sort by time to get everything back in order
        new_dataframes = {}
        for signal_number, signal_df in signal_dataframes.items():   # There are only two dfs to iterate over
            signal_df.reset_index(drop=True, inplace=True)
            signal_df.drop(columns='signalMembership', inplace=True)   # Don't need this column anymore
            if print_process: print('')
            new_df = switch_detection_utils.interpolate(signal_df, 2*self.switch_time, print_process)
            new_dataframes[signal_number]=new_df  

        ### Now we have the two separated signal dataframes.
        ### What follows is logic to handle their management
        self.signal_1_df = new_dataframes[True]
        self.signal_2_df = new_dataframes[False]
        # We choose a convention where "signal 1" is the signal whose first s1 quantity is lowest
        # By default, assign_by='s1Avg', but if s1 turns out to be close for both signals, you can assign by
        # the stokes parameter with the greatest variation between the two signals
        assign_by_arr = np.array(['s1Avg','s2Avg','s3Avg'])
        # If unspecified, we'll choose the assign_by parameter automatically, based on the largest spread between signals
        if assign_by is None:
            if print_assignment: print('\tChoosing assign_by parameter automatically; ',end='')
            first_stokes_difs = [(self.signal_2_df[assign_by][0]-self.signal_1_df[assign_by][0]) \
                                 for assign_by in assign_by_arr]
            max_dif_index = np.argmax([abs(n) for n in first_stokes_difs])
            assign_by = assign_by_arr[max_dif_index]
            assign_dif = first_stokes_difs[max_dif_index]
            if print_assignment: print('\tUsing assign_by = \"{:s}\" where dif = {:.2f}'.format(assign_by,assign_dif))
        else:
            assign_dif = self.signal_2_df[assign_by][0]-self.signal_1_df[assign_by][0]
        self.assign_by = assign_by
        
        # Switch (or don't) the signals and warn if the difference is too small to confidently distinguish
        if abs(assign_dif) < 0.3:
            print(BOLD_ON+'Warning: {:s} of signals are close together (dif = {:.2f}); difficult to assign \"signal 1\"'.format(assign_by,assign_dif)+BOLD_OFF)
        if assign_dif < 0:
            if print_assignment: print('\tDif = {:.2f} for {:s} is negative; switching signals'.format(assign_dif,assign_by))
            temp = self.signal_1_df
            self.signal_1_df = self.signal_2_df
            self.signal_2_df = temp
        else:
            if print_assignment: print('\tDif = {:.2f} for {:s} is positive; not switching signals'.format(assign_dif,assign_by))
        
        if print_process:
            print('Signal 1 size={}\tSignal 1 range: t=({:.2f},{:.2f})'.format(len(self.signal_1_df),self.signal_1_df.at[0,'EstTime'],self.signal_1_df.at[len(self.signal_1_df)-1,'EstTime']))
            print('Signal 2 size={}\tSignal 2 range: t=({:.2f},{:.2f})'.format(len(self.signal_2_df),self.signal_2_df.at[0,'EstTime'],self.signal_2_df.at[len(self.signal_2_df)-1,'EstTime']))
        if print_assignment:
            stokes_1_first, stokes_2_first = math_utils.first_stokes(self.signal_1_df, self.signal_2_df)
            print('\tFirst measurement of Signal 1: ', stokes_1_first)
            print('\tFirst measurement of Signal 2: ', stokes_2_first)
        
        self.df.drop(columns=['dif', 'mask'], inplace=True)   # At the end, drop these columns from the original dataframe
        if print_process: print(BOLD_ON+'=== end average_data ==='+BOLD_OFF)
        #return avg_df
        return
    
    
    ### This method calculates the stokes rotation matrix for each observed pairs of SOPs, given various reference criteria.
    ### If given an pair of SOPs (in the form reference = ([S1, S2, S3], [S1, S2, S3]) ), the calculated rotation matrix
    ### for each observation will be based off that given reference. The user can also choose to use the first recorded
    ### pair of SOPs as the reference for all observations by doing reference="first".
    ### If reference=None, the method will take on the "resetting reference based on threshold" appraoch; it will begin by
    ### calculating the rotation matrix with respect to the first recorded pair of SOPs (like when reference="first"), but
    ### when the calculated rotation angle of the rotation matrix surpasses angle_threshold_deg, it will reset the reference
    ### SOPs to be the recorded SOPs at that exact moment (meaning the rotation angle will jump back down, close to zero, on
    ### the next calculation. This option should be used for the witness beam wavelength, which (in theory) we can continuously
    ### monitor for drift of the PTF.
    ### For the other channel (the one we want to disturb as infrequently as possible), use the option reference=[t1, t2, ...]
    ### where reference is the list of reset times recorded from the witness beam channel. The method will reset the reference
    ### SOPs at those times (or as we pass them), no matter what the rotation angle is doing.
    def calc_stokes_ptf(self, reference=None, switch_inputs=False, angle_threshold_deg=10, reset_delay=0, reset_by_rolling=True, rolling_pts=5, print_process=False):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return

        self.reset_by_rolling = None
        self.angle_threshold_deg = None
        self.signal_1_input_stokes = None
        self.signal_2_input_stokes = None
        self.stokes_ptf_df = None
        self.reset_times = None

        # Logic: Determine what criteria for the reference SOPs we're using
        resetting_approach = False   # Witness beam channel
        reset_times_given = False   # Other channel
        if (reference is None):
            print("Using \"resetting reference SOPs\" approach with threshold = {:.1f} degrees".format(angle_threshold_deg))
            resetting_approach = True
            self.reset_by_rolling = reset_by_rolling
            self.angle_threshold_deg = angle_threshold_deg
            signal_1_input_stokes, signal_2_input_stokes = math_utils.first_stokes(self.signal_1_df, self.signal_2_df)
            self.signal_1_input_stokes = signal_1_input_stokes
            self.signal_2_input_stokes = signal_2_input_stokes
        elif (isinstance(reference, str)):
            print("Using first recorded SOPs as reference matrix")
            signal_1_input_stokes, signal_2_input_stokes = math_utils.first_stokes(self.signal_1_df, self.signal_2_df)
            self.signal_1_input_stokes = signal_1_input_stokes
            self.signal_2_input_stokes = signal_2_input_stokes
        elif isinstance(reference, tuple):
            print("Using provided SOPs as reference matrix")
            signal_1_input_stokes, signal_2_input_stokes = reference
            if switch_inputs:   # Told to switch the inputs of signal 1 and signal 2
                temp = signal_1_input_stokes
                signal_1_input_stokes = signal_2_input_stokes
                signal_2_input_stokes = temp
        elif isinstance(reference, list):
            print("Using provided reset times to reset reference matrix")
            reset_times_given = True
            signal_1_input_stokes, signal_2_input_stokes = math_utils.first_stokes(self.signal_1_df, self.signal_2_df)
            self.signal_1_input_stokes = signal_1_input_stokes
            self.signal_2_input_stokes = signal_2_input_stokes
        else:
            print("No approach found; provide proper \"reference\" parameter")
        print("Averaging rotAngle using " + BOLD_ON + "{:d} rolling points".format(rolling_pts) + BOLD_OFF)
        
        ### Initial reference SOPs determined by above logic
        reference_matrix = np.column_stack(math_utils.construct_orthonormal_bases(signal_1_input_stokes, signal_2_input_stokes))

        ### Iterate over the dataset, calculating the stokes rotation matrix between the observed pair of SOPs and the reference.
        ### reference_matrix may reset depending on the criteria above
        rows = list(zip(self.signal_1_df.iterrows(), self.signal_2_df.iterrows()))
        stokes_ptf_rows = []  # List to store the Series objects
        reset_times = []   # Times at which we reset the reference
        current_segment_values = []   # Initialize a list to temporarily store values of the current rolling average segment
        for (i1, row1), (i2, row2) in rows:
            result_row = math_utils.calc_stokes_rotation(reference_matrix, row1, row2)  # Calculate rotation matrix
            current_segment_values.append(result_row["rotAngle"]) # Add this value to the current rolling average segment
            # Caclulate rolling average for this point
            if len(current_segment_values) <= rolling_pts:
                result_row["rotAngleRolling"] = sum(current_segment_values) / len(current_segment_values)
            else:
                result_row["rotAngleRolling"] = sum(current_segment_values[-rolling_pts:]) / rolling_pts

            # Check these conditions for whether its time to reset
            time_to_reset=False
            if resetting_approach:
                # If resetting by raw rotAngle values, check threshold
                if not reset_by_rolling and result_row['rotAngle'] > angle_threshold_deg:
                    time_to_reset=True
                # If resettign by rolling rotAngle average, check threshold
                if reset_by_rolling and result_row['rotAngleRolling'] > angle_threshold_deg:
                    time_to_reset=True
                # TODO: also reset at the top of every hour, since we'll be cutting the data into hour long sections
            # If resetting by provided reset times, check if we've reached (exceeded) the next reset time
            elif reset_times_given and len(reset_times) < len(reference) and result_row["EstTime"] > reference[len(reset_times)]+reset_delay:
                time_to_reset=True

            if time_to_reset:
                # This row marks a reset; reset the reference SOPs
                result_row["WasReset"] = True   # Mark reset as True; default is False
                current_segment_values = []
                reset_times.append(result_row["EstTime"])
                signal_1_reference_stokes = result_row['sig1Stokes']
                signal_2_reference_stokes = result_row['sig2Stokes']
                reference_matrix = np.column_stack(math_utils.construct_orthonormal_bases(signal_1_reference_stokes, signal_2_reference_stokes))
            
            stokes_ptf_rows.append(result_row)
        
        # Print some debugging stuff
        if resetting_approach or reset_times_given: print(BOLD_ON+"Reference matrix was reset {:d} times".format(len(reset_times))+BOLD_OFF)
        if (resetting_approach or reset_times_given) and print_process:
            for i in range(0,min(10,len(reset_times))):
                print("Reset at time t={:.2f}".format(reset_times[i]))
            if len(reset_times) > 10:
                print("...")

        # We recorded the output rotation matrices as a list of pandas Series (each Series containing the relevant information)
        # Convert this list to a DataFrame
        self.stokes_ptf_df = pd.DataFrame(stokes_ptf_rows)
        self.stokes_ptf_df['EstTime'] = self.stokes_ptf_df['EstTime'].astype(float)

        # Calculate the difference between each rotAngle value
        # This may be a random walk, and may be useful if plotted on a histogram or plugged into ADev
        self.stokes_ptf_df['rotAngleDif'] = self.stokes_ptf_df['rotAngle'].diff()
        for i in range(len(self.stokes_ptf_df) - 1):  # -1 because we'll look at the next row inside the loop
            if self.stokes_ptf_df.loc[i, 'WasReset']:
                self.stokes_ptf_df.loc[i + 1, 'rotAngleDif'] = np.nan

        self.reset_times = reset_times
        return reset_times

    def adev(self, plot_param='s1', num_taus=5000, plot_adev=False, plot_psd=False):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return
        
        twice = True if plot_param in self.signal_1_df else False
        
        meas_rate = 1/(2*self.switch_time)
        if twice:
            signal_1 = self.signal_1_df[plot_param].values
            signal_2 = self.signal_2_df[plot_param].values
        else:
            signal = self.stokes_ptf_df[plot_param].values
        
        if plot_psd:
            # Plot PSD
            PSD_fig,PSD_ax = plt.subplots(figsize=(12,3))
            if twice:
                PSD_ax.psd(signal_1, NFFT=1024, Fs=meas_rate, label='Signal 1')
                PSD_ax.psd(signal_2, NFFT=1024, Fs=meas_rate, label='Signal 2')
            else:
                PSD_ax.psd(signal, NFFT=1024, Fs=meas_rate, label='Signal')
            PSD_ax.set_title('Power Spectral Density (PSD) of Signal')
            PSD_ax.set_xlabel('Frequency [Hz]')
            PSD_ax.set_ylabel('Power')
            PSD_ax.grid(True)
            PSD_ax.legend(loc='upper right')
            PSD_fig.tight_layout()
            display(PSD_fig); plt.close(PSD_fig)

        # Calc taus
        #num_taus = 5000
        base = 10
        #power = int(np.log(signal_1.size/2)/np.log(base))
        #pnts = np.logspace(0, power, num=num_taus, base=base, dtype=int) # exact integer number of points
        #taus = np.unique(pnts/meas_rate)
        min_value = 1/meas_rate
        if twice: max_value = min_value * (signal_1.size-1)
        else: max_value = min_value * (signal.size-1)
        taus = np.logspace(np.log(min_value) / np.log(base), np.log(max_value) / np.log(base), num=num_taus, base=base)
        #taus = 'all'

        
        if twice:
            taus2_1, ad_1, ade_1, ns_1 = allantools.oadev(signal_1, rate = meas_rate, taus=taus, data_type="freq")
            taus2_2, ad_2, ade_2, ns_2 = allantools.oadev(signal_2, rate = meas_rate, taus=taus, data_type="freq")
        else:
            taus2, ad, ade, ns = allantools.oadev(signal, rate = meas_rate, taus=taus, data_type="freq")
        #print('\t\trate={:.2f} | min tau = {:.5f} | max tau = {:.5f}'.format(meas_rate, taus[0], taus[-1]))
        if twice:
            print('Number of taus used for signal 1: {:d}'.format(taus2_1.size))
            print('Number of taus used for signal 2: {:d}'.format(taus2_2.size))
        else:
            print('Number of taus used for signal: {:d}'.format(taus2.size))
        
        label = '{:s} (ADev) | {:s} | {:s}'.format(plot_param, self.title, dataframe_management_utils.gen_time_str(self.df))
        label_1 = '{:s} (ADev) | {:s}, Signal {:d} | {:s}'.format(plot_param, self.title, 1, dataframe_management_utils.gen_time_str(self.df))
        label_2 = '{:s} (ADev) | {:s}, Signal {:d} | {:s}'.format(plot_param, self.title, 2, dataframe_management_utils.gen_time_str(self.df))
        
        adev_arr = []
        if twice:
            adev_arr.append((taus2_1, ad_1, ade_1, ns_1, label_1))
            adev_arr.append((taus2_2, ad_2, ade_2, ns_2, label_2))
        else:
            adev_arr.append((taus2, ad, ade, ns, label))
        
        ADev_fig = None
        if plot_adev:
            ADev_fig = StationarySet.plot_adev(adev_arr, plot_param=plot_param)
#             ADev_fig,ADev_ax = plt.subplots(figsize=(12,4))
#             ADev_ax.errorbar(taus2_1, ad_1, yerr=ade_1, label=label_1)
#             ADev_ax.errorbar(taus2_2, ad_2, yerr=ade_2, label=label_2)
#             ADev_ax.set_xscale("log")
#             ADev_ax.set_yscale("log")
#             ADev_ax.set_xlabel('Tau [s]')
#             ADev_ax.set_ylabel('Allan Deviation')
#             ADev_ax.set_title('Signal Stability')
#             ADev_ax.grid(True)
#             ADev_ax.legend(loc='upper left')
#             ADev_fig.tight_layout()
#             display(ADev_fig); plt.close(ADev_fig)
        
        return adev_arr, ADev_fig