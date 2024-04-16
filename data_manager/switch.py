BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
from .dataset import Dataset
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import allantools

class SwitchSet(Dataset):
    STOKES_PTF_DF_COLUMNS = ['EstTime', 'basesMatrix', 'sig1Stokes', 'sig2Stokes', 'axis', 'rotAngle', 'angleDif', 'rotAngleRolling', 'WasReset']
    
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
        #self.metrics=['dist','angle']
    
    ### Plots specified parameter over time (from the raw data)
    # birds_eye: plot the entirety of the avaliable data?
    # plot_param: choose from s1,s2,s3,S0,S1,S2,S3,Azimuth,Ellipticity,Power,DOP,...
    # sample_range: used to zoom in on a particular time range, e.g. (2000,2050) seconds
    # time_offset: used by SetPair (allows offsetting of plot by this constant; purely for plotting, no functional purpose)
    def plot_raw(self,birds_eye=True,plot_param='s1',sample_range=None,plot_switch=False,plot_jumps=False,plot_valid=False,plot_avg=False,time_offset=0.0):
        # sample_range should be of the form (sample_start, sample_end) if a smaller range is desired
        # if sample_start or sample_end are None themselves, they will be filled in
        if sample_range is not None:
            sample_start, sample_end = Dataset.fill_in_range(sample_range, self.df)
            sample_range = (sample_start+time_offset, sample_end+time_offset)   # Make sure we include the offset
        
        # Plot entire dataset if requested
        if birds_eye:
            BE_fig, BE_ax = plt.subplots(figsize=(12,3))
            BE_ax.plot(self.df['TimeElapsed'], Dataset.get_correct_units(self.df[plot_param]), label=plot_param, linewidth=0.5, marker='o', markersize=0.8, color='red', alpha=0.5)
            BE_ax.set_xlabel('Time [s]')
            BE_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df)), fontsize=14, fontweight='bold')
            BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
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
            ZI_ax.plot(self.df['TimeElapsed'], Dataset.get_correct_units(self.df[plot_param]), label=plot_param, linewidth=1, marker='o', markersize=1.5, color='red')
            ZI_ax.set_xlabel('Time [s]')
            ZI_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
            ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
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
                if plot_jumps:
                    for i in range(len(self.df)):
                        if self.df.at[i,'IsJump']:
                            X = self.df.loc[i-1:i,'TimeElapsed']
                            Y = self.df.loc[i-1:i,plot_param]
                            ZI_ax.plot(X,Y, linewidth=1, marker='o', markersize=1.5, color='orange')

                # TODO: doing this by groupby may be more efficient/cleaner
                # If requested, overwrite the included points with the color blue
                # Assumes all valid sections start with an IsStartPoint=True and end with an IsEndPoint=True
                if plot_valid:
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
                        ZI_ax.plot(self.signal_1_df['EstTime'],Dataset.get_correct_units(self.signal_1_df[plot_param_avg_str]),label=plot_param+" 1",linewidth=1,marker='o',markersize=3,color='orange')
                        ZI_ax.plot(self.signal_2_df['EstTime'],Dataset.get_correct_units(self.signal_2_df[plot_param_avg_str]),label=plot_param+" 2",linewidth=1,marker='o',markersize=3,color='green')

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
    def plot_separated(self,birds_eye=True,plot_param='s1Avg',plot_param_2=None,sample_range=None,plot_signal=None,time_offset=0.0):
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
        if birds_eye:
            # Birds Eye plot
            BE_fig, BE_ax = plt.subplots(figsize=(12,3))
            custom_palette = sns.color_palette("dark")
            # Only plot specified signal(s)
            if plot_param in self.signal_1_df.columns:   # Plot from separated signals
                if plot_signal is None or plot_signal == 1:
                    BE_ax.plot(self.signal_1_df['EstTime'], Dataset.get_correct_units(self.signal_1_df[plot_param]), label=plot_param+" 1", \
                               alpha=0.5, linestyle='-', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[0])
                if plot_signal is None or plot_signal == 2:
                    BE_ax.plot(self.signal_2_df['EstTime'], Dataset.get_correct_units(self.signal_2_df[plot_param]), label=plot_param+" 2", \
                               alpha=0.5, linestyle='-', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[1])
            else:   # Plot from ptf
                BE_ax.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df[plot_param]), label=plot_param, \
                               alpha=0.5, linestyle='-', linewidth=0.4, marker='', markersize=0.5, color=custom_palette[0])
                if plot_param=='rotAngle':   # If plotting rotAngle, also plot rolling average
                    BE_ax.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               linestyle='-', linewidth=0.8, marker='', markersize=0.5, color=custom_palette[2])
                if self.angle_threshold_deg is not None: BE_ax.axhline(y=Dataset.get_correct_units(self.angle_threshold_deg), color='red', linewidth=1)
                if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            BE_ax.axvline(time, color = 'red', linewidth=0.5)
            BE_ax.set_xlabel('Time [s]')
            BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
            BE_ax.grid(True)
            # If requested, plot secondary parameter
            if plot_param_2 is not None:
                BE_ax2 = BE_ax.twinx()
                if plot_param_2 in self.signal_1_df.columns:
                    if plot_signal is None or plot_signal == 1:
                        BE_ax2.plot(self.signal_1_df['EstTime'], Dataset.get_correct_units(self.signal_1_df[plot_param_2]), label=plot_param_2+" 1", \
                                    alpha=0.5, linestyle='--', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[9])
                    if plot_signal is None or plot_signal == 2:
                        BE_ax2.plot(self.signal_2_df['EstTime'], Dataset.get_correct_units(self.signal_2_df[plot_param_2]), label=plot_param_2+" 2", \
                                    alpha=0.5, linestyle='--', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[8])
                else: # Plot from ptf
                    BE_ax2.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df[plot_param_2]), label=plot_param_2, \
                                alpha=0.5, linestyle='--', linewidth=0.4, marker='', markersize=0.5, color=custom_palette[9])
                    if plot_param_2=='rotAngle':   # If plotting rotAngle, also plot rolling average
                        BE_ax2.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                                   linestyle='--', linewidth=0.8, marker='', markersize=0.5, color=custom_palette[7])
                    if self.angle_threshold_deg is not None: BE_ax2.axhline(y=Dataset.get_correct_units(self.angle_threshold_deg), color='red', linewidth=1)
                    if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            BE_ax2.axvline(time, color = 'red', linewidth=0.5)
                BE_ax2.set_ylabel('{:s} [{:s}]'.format(plot_param_2,SwitchSet.UNITS.get(plot_param_2,'TODO')))
                # lines code is just to get legend working (ChatGPT generated)
                lines1, labels1 = BE_ax.get_legend_handles_labels()
                lines2, labels2 = BE_ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                BE_ax.set_title('{:s} & {:s} (separated) | {:s} | {:s}'.format(plot_param, plot_param_2, self.title, Dataset.gen_time_str(self.df)), fontsize=14, fontweight='bold')
                BE_ax.legend(lines, labels, loc='upper left')
            else:
                BE_ax.set_title('{:s} (separated) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df)), fontsize=14, fontweight='bold')
                BE_ax.legend(loc='upper left')
            # We add green lines to the birds eye plot to denote where sample_range is located
            if sample_range is not None:
                for val in sample_range:
                    BE_ax.axvline(val, color = 'green', linewidth=1)
            display(BE_fig); plt.close(BE_fig)   # Show entire dataset
        
        # If requested, we'll also plot the smaller sample range
        if sample_range is not None:
            # Zoomed In plot 
            ZI_fig, ZI_ax = plt.subplots(figsize=(12,3))
            # Only plot specified signal(s)
            if plot_param in self.signal_1_df.columns:   # Plot from separated signals
                if plot_signal is None or plot_signal == 1:
                    ZI_ax.plot(self.signal_1_df['EstTime'],Dataset.get_correct_units(self.signal_1_df[plot_param]),label=plot_param+" 1", \
                               linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[0])
                if plot_signal is None or plot_signal == 2:
                    ZI_ax.plot(self.signal_2_df['EstTime'],Dataset.get_correct_units(self.signal_2_df[plot_param]),label=plot_param+" 2", \
                               linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[1])
            else:   # Plot from ptf
                ZI_ax.plot(self.stokes_ptf_df['EstTime'],Dataset.get_correct_units(self.stokes_ptf_df[plot_param]),label=plot_param, \
                           linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[0])
                if plot_param=='rotAngle':   # If plotting rotAngle, also plot rolling average
                    ZI_ax.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[2])
                if self.angle_threshold_deg is not None: ZI_ax.axhline(y=Dataset.get_correct_units(self.angle_threshold_deg), color='red', linewidth=1)
                if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            if sample_range[0] <= time <= sample_range[1]:
                                ZI_ax.axvline(time, color = 'red', linewidth=0.5)
            ZI_ax.set_xlabel('Time [s]')
            ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
            ZI_ax.grid(True)
            # If requested, plot secondary parameter
            if plot_param_2 is not None:
                ZI_ax2 = ZI_ax.twinx()
                if plot_param_2 in self.signal_1_df.columns:   # Plot from separated signals
                    if plot_signal is None or plot_signal == 1:
                        ZI_ax2.plot(self.signal_1_df['EstTime'], Dataset.get_correct_units(self.signal_1_df[plot_param_2]), label=plot_param_2+" 1", \
                                    linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[9])
                    if plot_signal is None or plot_signal == 2:
                        ZI_ax2.plot(self.signal_2_df['EstTime'], Dataset.get_correct_units(self.signal_2_df[plot_param_2]), label=plot_param_2+" 2", \
                                    linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[8])
                else:   # Plot from ptf
                    ZI_ax2.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df[plot_param_2]), label=plot_param_2, \
                                linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[9])
                    if plot_param_2=='rotAngle':   # If plotting rotAngle, also plot rolling average
                        ZI_ax2.plot(self.stokes_ptf_df['EstTime'], Dataset.get_correct_units(self.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling', \
                               linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[7])
                    if self.angle_threshold_deg is not None: ZI_ax2.axhline(y=Dataset.get_correct_units(self.angle_threshold_deg), color='red', linewidth=1)
                    if self.reset_times is not None and len(self.reset_times) > 0:
                        for time in self.reset_times:
                            if sample_range[0] <= time <= sample_range[1]:
                                ZI_ax2.axvline(time, color = 'red', linewidth=0.5)
                ZI_ax2.set_ylabel('{:s} [{:s}]'.format(plot_param_2,SwitchSet.UNITS.get(plot_param_2,'TODO')))
                # lines code is just to get legend working (ChatGPT generated)
                lines1, labels1 = ZI_ax.get_legend_handles_labels()
                lines2, labels2 = ZI_ax2.get_legend_handles_labels()
                lines = lines1 + lines2
                labels = labels1 + labels2
                ZI_ax.set_title('{:s} & {:s} (separated) | {:s} | {:s}'.format(plot_param, plot_param_2, self.title, Dataset.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
                ZI_ax.legend(lines, labels, loc='upper left')
            else:
                ZI_ax.set_title('{:s} (separated) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
                ZI_ax.legend(loc='upper left')
            #TODO: standard deviation values
            #TODO: ZI_ax.errorbar(self.signal_1_df['EstTime'],self.signal_1_df[plot_param],yerr=self.signal_1_df[plot_param_std_str],label=plot_param+" 1",linewidth=1,marker='o',markersize=1.5,color='orange')
            #TODO: ZI_ax.errorbar(self.signal_2_df['EstTime'],self.signal_2_df[plot_param],yerr=self.signal_2_df[plot_param_std_str],label=plot_param+" 2",linewidth=1,marker='o',markersize=1.5,color='purple')
            ZI_ax.set_xlim(sample_range[0],sample_range[1])
            display(ZI_fig); plt.close(ZI_fig)
        
        #return BE_fig, ZI_fig
        return
    
    @staticmethod
    def compute_range(hist_df, first_tolerance, second_tolerance, patches, print_process=False):
        if print_process: print('computing range...',end=' ')
        bin_start = hist_df.loc[hist_df['index'] == 0, 'bins'].iloc[0]
        bin_end = hist_df.loc[hist_df['index'] == 1, 'bins'].iloc[0]
        width = bin_end - bin_start   # width of bins
        first_tolerance_error_message = 'Likely cause: VERY high variability in dataset (how did you do this)'
        second_tolerance_error_message = 'Likely cause: FPCs moved during setup OR high variability in dataset'
        # Based on the frequencies of jump differences, approximate the range in which the peak in differences is located
        # hist_df.at[1,'log_values_diffs'] = counts of the ~0 jump differences (most adjacent points are of the same polarization)
        # hist_df.at[2,'log_values_diffs'] = counts of the next most frequent jump differences; should be the jump between polarizations
        # hist_df.at[3,'log_values_diffs'] = counts of the thirs most frequent jump differences; assuming each polarization remains constant, this should only be large if the second and third bins divide the peak in jump differences
        
        # TODO: the parameter might not have enough difference in the jumps to detect the change point
        # Temporary solution: if this is the case, throw an error to notify user. We can catch it up above
        #display(hist_df.head(10))
        if len(hist_df)<4:
            raise Exception("Not enough differences in the jumps to detect change points")
        
        # TODO: if peak divided between two bins, simply shift number of bins and redo process to put peak into one
        if hist_df.at[3,'log_values_diffs'] < hist_df.at[2,'log_values_diffs']:   # If the O.M. diff between 2 and 3 is less than than that between 1 and 2
            #print('expected')
            # Peak of differences spans only 1 bin (expected)
#             if print_process:
#                 if hist_df.at[1,'log_values_diffs'] < first_tolerance:
#                     print(BOLD_ON + 'Error in First Tolerance: difference={:.2f} < tolerance={:.2f}'.format(hist_df.at[1,'log_values_diffs'], first_tolerance) + BOLD_OFF)
#                     print(BOLD_ON + first_tolerance_error_message + BOLD_OFF)
#                 if hist_df.at[2,'log_values_diffs'] < second_tolerance:
#                     print(BOLD_ON + 'Error in Second Tolerance: difference={:.2f} < tolerance={:.2f}'.format(hist_df.at[2,'log_values_diffs'], second_tolerance) + BOLD_OFF)
#                     print(BOLD_ON + second_tolerance_error_message + BOLD_OFF)
            left_edge = hist_df.at[1,'bins']
            right_edge = left_edge + width
            patches[hist_df.at[1,'index']].set_fc('red')   # Add highlights to jump_fig hist
        else:
            #print('rare')
            # Peak of differences spans 2 bins (rare)
            # (Assuming the data is steady so the two bins will be adjacent)
            # We increase the first tolerance and decrease the second tolerance since
            # the third bin is much larger than expected and the second bin is much smaller
            # Assume the split bins are of equal height (bin edge divides peak symmetrically)
            # Then we subtract log10(2) = 0.3 from the second tolerance and add it to the first
#             if print_process:
#                 if hist_df.at[1,'log_values_diffs'] < first_tolerance+0.3:
#                     print(BOLD_ON + 'Error in (adjusted) First Tolerance: difference={:.2f} < tolerance={:.2f}'.format(hist_df.at[1,'log_values_diffs'], first_tolerance+0.3) + BOLD_OFF)
#                     print(BOLD_ON + first_tolerance_error_message + BOLD_OFF)
#                 if hist_df.at[3,'log_values_diffs'] < second_tolerance-0.3:
#                     print(BOLD_ON + 'Error in (adjusted) Second Tolerance: difference={:.2f} < tolerance={:.2f}'.format(hist_df.at[3,'log_values_diffs'], second_tolerance-0.3) + BOLD_OFF)
#                     print(BOLD_ON + second_tolerance_error_message + BOLD_OFF)
            # Assuming the bin edge divides peak symmetrically,
            # we find that bin edge (center) which is the max of the left edges of the two split bins
            # and we take +/- half of the bin width
            center = np.maximum(hist_df.at[1,'bins'], hist_df.at[2,'bins'])
            left_edge = center - width/2
            right_edge = center + width/2
            patches[hist_df.at[1,'index']].set_fc('red')   # Add highlights to jump_fig hist
            patches[hist_df.at[2,'index']].set_fc('red')
        if print_process: print('---> range = ({:.2f},{:.2f})'.format(left_edge,right_edge))
        return left_edge, right_edge

    @staticmethod
    def compute_offset(midtime_array, nominal_time, print_process=False):
        if print_process: print('computing offset...')
        ### We want to find the phase shift (offset time) of the detected jumps ###
        
        # We get offsets of each jump. We imagine a series of ghost switches which are spaced by exactly nominal_time
        # and begin exactly at time = 0. If the midtimes were perfectly spaced, each midtime would come a consistent
        # offset time (OT) after the previous ghost switch. The offset time (OT) could be anywhere between
        # OT=0 (if the switch comes right after the previous ghost switch) and
        # OT=nominal_time (if the switch comes right before the next ghost switch).
        # By taking midtime mod nominal_time, we get this offset
        initial_offset_array = np.array([midtime % nominal_time for midtime in midtime_array])
        
        # However, the midtimes will deviate forward and backward from this offset, forming a (roughly) gaussian
        # distribution around the "mean offset" time we are trying to find. We will average all the midtimes to
        # find this mean offset, but we must consider that if the tail of the gaussian extends before OT=0 or
        # after OT=nominal_time, it will wrap around which will severely skew the data.
        # For poorly behaved data (with either a non-gaussian distribution or a distribution who's range is larger
        # that nominal_time), it may be impossible to tell which part of the distribution represents the actual
        # offset time and which part is the tail wrapping around. We assume well behaved data, however, and thus
        # we might identify an initial guess for the mean by binning the data and using the mode.
        # Using this, and assuming a symmetric distribution, we can give the histogram the best chance of fitting
        # in the [0,nominal_time] range by unwrapping in a nominal_time/2 range around the mode location
        
        # Make the initial histogram
        offset_fig,(initial_offset_ax,final_offset_ax) = plt.subplots(1,2,figsize=(12,4))
        initial_offset_bins = 20   # Seemed like a good number
        initial_offset_values, initial_offset_bins, _ = initial_offset_ax.hist(initial_offset_array,bins=initial_offset_bins,label='offsets',alpha=0.8)
        initial_offset_ax.grid(True)
        initial_offset_ax.set_title('Initial histogram of offsets')
        initial_offset_ax.set_xlabel('Offset magnitude')
        
        # Find the location of the mode
        # Put bin locations and heights together so when we sort it the locations stay with the values
        initial_offset_hist_df = pd.DataFrame({'bins':initial_offset_bins[:-1],'values':initial_offset_values})
        width = initial_offset_hist_df.at[1,'bins'] - initial_offset_hist_df.at[0,'bins']   # width of bins
        # Sort by height of bin
        initial_offset_hist_df = initial_offset_hist_df.sort_values('values', ascending=False).reset_index(drop=True)
        mode_loc = initial_offset_hist_df.at[0,'bins'] + width/2   # Left edge of the tallest bin plus half the bin width
        # Plot the mode on the initial offset histogram
        initial_offset_ax.axvline(mode_loc, label='mode guess', color='red', linestyle='--')
        initial_offset_ax.legend(loc='upper right')
        
        wrap_loc = (mode_loc + nominal_time/2) % nominal_time
        # compare_func determines if offset needs to be wrapped around
        # wrap_func wraps around the offset
        # if (wrap_loc>nominal_time/2), offsets greater than wrap_loc need to be brought before 0
        # else, offsets less than or equal to wrap_loc need to be brought after nominal time
        compare_func = (lambda offset: offset>wrap_loc) if (wrap_loc>nominal_time/2) else (lambda offset: offset<=wrap_loc)
        wrap_func = (lambda offset: offset-nominal_time) if (wrap_loc>nominal_time/2) else (lambda offset: offset+nominal_time)
        # Perform the wrapping
        final_offset_array = np.array([wrap_func(offset) if compare_func(offset) else offset for offset in initial_offset_array])
        # Now we can finally get the average of the offsets
        mean_offset = final_offset_array.mean()
        # Histogram of wrapped offsets
        final_offset_ax.hist(final_offset_array,bins=initial_offset_bins,label='offsets',alpha=0.8)
        final_offset_ax.axvline(mean_offset, label='mean offset', color='red', linestyle='--')
        final_offset_ax.grid(True)
        final_offset_ax.set_title('Wrapped histogram of offsets')
        final_offset_ax.set_xlabel('Offset magnitude')
        final_offset_ax.legend(loc='upper right')
        offset_fig.tight_layout()
        if print_process:
            # If desired, we'll show these histograms
            display(offset_fig); plt.close(offset_fig)
            print('\n')
        else:
            plt.close(offset_fig)
        return mean_offset
    
    @staticmethod
    def change_point(main_df, nominal_switch_time, change_point_range, n_exclude, print_process=False):
        if print_process: print(BOLD_ON+'=== performing change point detection ==='+BOLD_OFF)
        # change_point_range should be of the form (change_point_start, change_point_end) if a smaller range is desired
        # if either start or end are None themselves, they will be filled in automatically
        change_point_start, change_point_end = Dataset.fill_in_range(change_point_range, main_df)
        change_point_range = (change_point_start, change_point_end)
        if print_process: print('cp range: {}'.format(change_point_range))
        
        # Make dataframe we'll use for change point calculations
        points_skipped = main_df.index[main_df['TimeElapsed'] >= change_point_range[0]].min()
        change_point_df = main_df[(main_df['TimeElapsed'] >= change_point_range[0]) & (main_df['TimeElapsed'] <= change_point_range[1])]
        change_point_df = change_point_df.copy().reset_index(drop=True)
        if print_process: print('using {} data entries'.format(len(change_point_df)))
        if print_process: print('skipping {} points'.format(points_skipped))
        
        # noise_tolerance: Maximum amount we expect switch_param could change per second(?) in a steady signal
        use_angles = False
        noise_tolerance=3 if use_angles else 0.03   # TODO: temporary solution for stokes params tolerance
        # first_tolerance: Order of magnitude difference between first and second bin
        # second_tolerance: Order of magnitude difference between second and third bin
        first_tolerance=1; second_tolerance=0.5
#         if print_process:
#             # Display params we're using
#             print('\tnoise_tolerance = {:.1f}'.format(noise_tolerance))
#             print('\tfirst_tolerance = {:.1f}'.format(first_tolerance))
#             print('\tsecond_tolerance = {:.1f}\n'.format(second_tolerance))

        ###
        ###
        ### Perform the change point process for all parameters and use the one with the largest jumps
        max_jump_width = -1; switch_param = 'None'
        switch_param_list = ['Azimuth', 'Ellipticity'] if use_angles else ['s1','s2','s3']
        if print_process: print('considering params: {}'.format(switch_param_list))
        error_messages = []
        for switch_param_iter in switch_param_list:
            if print_process: print(BOLD_ON+'\n--- switch param: {:s} ---'.format(switch_param_iter)+BOLD_OFF)
            # Preprocessing
            y_iter = np.array(change_point_df[switch_param_iter].tolist())   # data we'll be using
            if print_process: print('size of data we are using = {}'.format(y_iter.shape))
            
            diffs_iter = np.abs(np.array(y_iter[1:]) - np.array(y_iter[:-1]))   # Differences between data points spaced k indices apart
            #print('diffs_iter first few values'); display(diffs_iter[:3])
            #change_point_df[] = diffs_iter_temp = np.abs(change_point_df[switch_param_iter].diff())   # Differences between data points spaced k indices apart
            #display(diffs_iter.shape)
            #display(diffs_iter_temp.shape)
            
            # Visualizing the differences we're getting
            jump_fig_iter, (jump_ax1_iter, jump_ax2_iter) = plt.subplots(1,2,figsize=(12,2))
            jump_ax1_iter.plot(abs(diffs_iter))   # Plot how differences appear over time
            jump_ax1_iter.set_title('Differences between adjacent points')
            jump_ax1_iter.set_xlim(0,len(diffs_iter))
            xmax = np.max(diffs_iter)
            num_bins = int(np.ceil(xmax/(2*noise_tolerance)))   # 2*noise_tolerance = width of bins gives number of bins
            #num_bins = 100
            values_iter, bins_iter, patches_iter = jump_ax2_iter.hist(diffs_iter, bins=num_bins, range=(0, xmax))   # Plot histogram of differences
            jump_ax2_iter.set_title('Histogram of jump differences')
            jump_ax2_iter.set_yscale('log')
            jump_ax2_iter.set_xlim(0,xmax)
            jump_fig_iter.tight_layout()

            # Put the histogram data into a DataFrame
            hist_df_iter = pd.DataFrame({'bins':bins_iter[:-1],'values':values_iter})   # Return DataFrame of bin edges and values
            hist_df_iter['log_values'] = hist_df_iter['values'].replace(0,1)   # So that log doesn't throw errors. The 0/1 values won't have any effects
            hist_df_iter['log_values'] = np.log10(hist_df_iter['log_values'])   # Log scale since we're examining differences in orders of magnitude
            hist_df_iter = hist_df_iter.sort_values('log_values', ascending=False).reset_index()   # Move top bins up front
            hist_df_iter['log_values_diffs'] = np.abs(hist_df_iter['log_values'].diff())   # Take differences between order of magnitude (O.M.) measurements (these differences not equal to jump differences!)
            #if print_process: print('hist df head:'); display(hist_df_iter.head(3))

            try:
                # Computing the range which we think the proper jumps are within
                left_edge_iter, right_edge_iter = SwitchSet.compute_range(hist_df_iter, first_tolerance, second_tolerance, patches_iter, print_process)
                #if print_process: print('| range iter = ({:.2f},{:.2f})'.format(left_edge_iter,right_edge_iter))
                # Add range markers to jump plot
                jump_ax1_iter.hlines(left_edge_iter, 0, len(diffs_iter), color='red', linewidth=1)
                jump_ax1_iter.hlines(right_edge_iter, 0, len(diffs_iter), color='red', linewidth=1)
                width_iter = right_edge_iter-left_edge_iter   # Caculate width of the detected ZI jump
                if print_process:
                    # If desired, we display jump_fig
                    print('displaying jump_fig for switch parameter = {:s}\n'.format(switch_param_iter))
                    display(jump_fig_iter); plt.close(jump_fig_iter)
                    print('\n')
                else:
                    plt.close(jump_fig_iter)
            except Exception as e:
                # If this switch_param_iter didn't work to detect ZI jumps, we'll add it to the error messages
                # One parameter not working is expected;
                # If both do not work, then we have a problem: we'll print the errors then, if that happens
                err_str = "Error calculating ZI jump widths in {:s}:\n\t{:s}".format(switch_param_iter,e.message)
                error_messages.append(err_str)
                width_iter = -1
                if print_process:
                    # If desired, we'll print the error message here too, though
                    print(err_str+'\n')
            
            if width_iter > max_jump_width:
                if print_process: print('found width of '+BOLD_ON+'{:.3f} for {:s}'.format(width_iter,switch_param_iter)+BOLD_OFF)
                if print_process: print('greater than prev width of {:.3f} for {:s}'.format(max_jump_width,switch_param))
                switch_param = switch_param_iter
                diffs = diffs_iter
                left_edge = left_edge_iter
                right_edge = right_edge_iter
                max_jump_width = width_iter
                jump_fig = jump_fig_iter # TODO: jump_fig_iter was closed, so jump_fig might not work?
            if print_process: print(BOLD_ON+'--- end of switch param: {:s} ---'.format(switch_param_iter)+BOLD_OFF)
        
        # If neither parameter worked to detect the ZI jumps, we raise an error
        if max_jump_width<0:
            err_str = ""
            for err in error_messages:
                err_str = err_str + err + '\n'
            raise Exception(err_str + "Not enough difference in any given switch parameter to detect ZI jumps/times")
        
        
        
        ### This is the parameter we're using to detect changes; the one with the largest jumps
        if print_process: print(BOLD_ON+'\n---> switch parameter found: {:s} <---'.format(switch_param)+BOLD_OFF)
        
        # Now we have the range in which we expect the peak of differences to be (left_edge,right_edge),
        # so we search for all points where these differences developed (the jumps) and mark them as such
        if print_process: print('finding differences within range = ({:.2f},{:.2f})'.format(left_edge,right_edge))
        midtime_array = []
        for i in range(len(diffs)):
            dif = diffs[i]
            if (dif >= left_edge) & (dif <= right_edge):
                x1 = change_point_df.at[i,'TimeElapsed']
                x2 = change_point_df.at[i+1,'TimeElapsed']
                midtime_array.append((x1+x2)/2)
                # index needs shifting for main_df since using ZI range excludes some beginning points
                # only needs shifting here because we're going off of diffs (which doesn't account for the change point range shift)
                main_df.loc[points_skipped+i+1, 'IsJump'] = True
        if print_process:
            # If desired, print the number of jump points we detect
            ZI_range_time = change_point_df.at[len(change_point_df)-1,'TimeElapsed'] - change_point_df.at[0,'TimeElapsed']
            print("Jump points detected: {:d}\nout of total: ~{:.1f}\n".format(len(midtime_array), ZI_range_time/nominal_switch_time))

        ### We want to find the phase shift (offset time) of the detected jumps ###
        mean_offset = SwitchSet.compute_offset(midtime_array, nominal_switch_time, print_process)
        if print_process: print('offset = {:.3f} seconds'.format(mean_offset))
        if print_process: print(BOLD_ON+'=== end change point detection ==='+BOLD_OFF)
        return change_point_range, change_point_df, points_skipped, switch_param, mean_offset, left_edge, right_edge, jump_fig
    
    ### Function used by user
    ### Given the two change point ranges spaced >2000s apart (two to account for recording drift over 1000's of seconds),
    ### this function computes the time offset of the switches and marks down such information
    ### so it can be used by average_data, plotting functions, and future functions
    ### Later results will be more precise if the first change point range is close to the front of the dataset
    def find_switches(self, nominal_switch_rate=2, change_point_range_1=None, change_point_range_2=None, time_offset=0.0, n_exclude=3, print_process=False):
        # Ensure change point hasn't already been calculated
        if self.change_point_params!=[]:
            # TODO: allow to recalculate change points based on new change point range
            print(BOLD_ON + 'Error: change times already detected!' + BOLD_OFF)
            return
        self.nominal_switch_time = 1/(2*nominal_switch_rate)   # Time between switches
        self.df.insert(self.df.columns.get_loc('TimeDiff')+1, 'IsJump', [False]*len(self.df))
        
        
        ### Perform change point detection at the beginning of the dataset to find the offset of the switches
        # If desired, determine cp ranges automatically
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
        if print_process: print('Using change_point_range_1={}\nUsing change_point_range_2={}'.format(change_point_range_1,change_point_range_2))
        change_point_range_1 = (change_point_range_1[0]+time_offset, change_point_range_1[1]+time_offset)   # Include time_offset
        change_point_range_1, change_point_df_1, points_skipped_1, switch_param_1, mean_offset_1, left_edge_1, right_edge_1, jump_fig_1 = SwitchSet.change_point(self.df,self.nominal_switch_time,change_point_range_1,n_exclude,print_process)
        self.change_point_params.append((change_point_range_1, change_point_df_1, points_skipped_1, switch_param_1, mean_offset_1, left_edge_1, right_edge_1, jump_fig_1))
        #self.switch_offset = mean_offset_1
        # If we're dealing with an extra long dataset, the switch time is liable to drift, so we
        # perform change point detection a second time ~3000 seconds (~an hour) later and correct
        # the switch time
        if change_point_range_2 is not None:
            change_point_range_2 = (change_point_range_2[0]+time_offset, change_point_range_2[1]+time_offset)   # Include time_offse
            change_point_range_2, change_point_df_2, points_skipped_2, switch_param_2, mean_offset_2, left_edge_2, right_edge_2, jump_fig_2 = SwitchSet.change_point(self.df,self.nominal_switch_time,change_point_range_2,n_exclude,print_process)
            self.change_point_params.append((change_point_range_2, change_point_df_2, points_skipped_2, switch_param_2, mean_offset_2, left_edge_2, right_edge_2, jump_fig_2))
            # Difference in the two offsets; this is the total drift between the two ranges
            offset_drift = mean_offset_2 - mean_offset_1
            # Occasionally, stuff will happen with the mod of the offset drift, and the phase range will have to be adjusted
            if abs(offset_drift) > (self.nominal_switch_time/2):
                if offset_drift > 0:
                    offset_drift -= self.nominal_switch_time
                else:
                    offset_drift += self.nominal_switch_time
            # Now, offset_drift should be a little above 0 or a little below 0
            # A positive offset_drift means the actual switch time is a little more than the nominal switch time
            # Vice versa for a negative offset drift
            # How many switches it took for the offset to drift by offset_drift:
            cp_range_1_midpoint = (change_point_range_1[1]+change_point_range_1[0])/2; cp_range_2_midpoint = (change_point_range_2[1]+change_point_range_2[0])/2
            num_switches_between_cp_ranges = (cp_range_2_midpoint - cp_range_1_midpoint) / self.nominal_switch_time
            # Correction to the nominal_switch_time to account for limited resolution in the waveform generator
            # How much drift is contained in each switch between the two ranges
            switch_time_correction = offset_drift / num_switches_between_cp_ranges
            # Add the delta; this is the actual time between switches
            self.switch_time = self.nominal_switch_time + switch_time_correction
            # If the offset is referenced from 0, we need to adjust it now that the switch time has changed
            # The offset needs to be the remainder of the detection location divided by the switch time
            # mean_offset_1_location is the detection location
            mean_offset_1_location = mean_offset_1 + self.nominal_switch_time * np.floor(cp_range_1_midpoint/self.nominal_switch_time)
            # We calculate the new offset based on the newly corrected switch time
            self.switch_offset = mean_offset_1_location % self.switch_time
            if print_process: print(BOLD_ON+'Nominal Switch Time = {:.7f}\nOffset change of {:.3f} seconds over {} switches\nCorrected Switch Time = {:.7f}'.format(self.nominal_switch_time,offset_drift,num_switches_between_cp_ranges,self.switch_time)+BOLD_OFF)
        else:
            self.switch_time = self.nominal_switch_time   # If second range not provided, don't correct switch time
            self.switch_offset = mean_offset_1   # There is nothing in the offset to correct for
        
        
        ### Now we can confidently place the time of each switch between the polarization signals
        ### We find which points correspond to those switches, and mark them for later use
        t_min = self.df.at[0,'TimeElapsed']; t_max = self.df.at[self.df.shape[0]-1,'TimeElapsed']
        num_switch_to_start_on = np.ceil((t_min - self.switch_offset)/self.switch_time)
        time_to_start_on = num_switch_to_start_on * self.switch_time + self.switch_offset
        switch_array = np.arange(time_to_start_on, t_max, self.switch_time)   # Array of estimated switch times along dataset
        switch_indices = np.searchsorted(self.df['TimeElapsed'].tolist(), switch_array)   # Time points in dataset which occur right after a switch
        #value = [idx in switch_indices for idx in self.df.index]
        self.df.insert(self.df.columns.get_loc('IsJump')+1, 'IsSwitch', self.df.index.isin(switch_indices))   # These points are marked as switch points
        
        ### Now we exclude points too close to a switch from being included in the separated data,
        ### as they could be outliers or limbo points
        # IsStartPoint and IsEndPoint mark the start and end of valid sections of data, which will later be separated
        # out into two signals
        self.df.insert(self.df.columns.get_loc('IsSwitch')+1, 'IsValidPoint', [True]*len(self.df))   # Assume all points valid; exclude as we go
        self.df.insert(self.df.columns.get_loc('IsValidPoint')+1, 'IsStartPoint', [False]*len(self.df))
        self.df.insert(self.df.columns.get_loc('IsStartPoint')+1, 'IsEndPoint', [False]*len(self.df))
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'Azimuth']].head(10))
        
        # Iterate over (most of) dataset; when we encounter a switch point, we'll exclude n points to the left&right
        # Valid segments may not go off the edge of the dataset (we won't include them)
        for i in range(n_exclude+1, len(self.df)-n_exclude-1):
            #print('i={:d}: IsSwitch={:s}'.format(i, str(self.df.at[i, 'IsSwitch'])))
            if self.df.at[i, 'IsSwitch']:
                # We exclude n points to the left of i and n-1 points to the right of i
                # since the switch point is the point after the estimated switch time
                exclude_left_edge = i-n_exclude; exclude_right_edge = i+n_exclude-1   # Left and right edges of the exclude range
                self.df.loc[exclude_left_edge:exclude_right_edge, 'IsValidPoint'] = False   # Exclude points in range
                self.df.loc[exclude_left_edge-1, 'IsEndPoint'] = True   # Mark the end of the previous valid range
                self.df.loc[exclude_right_edge+1, 'IsStartPoint'] = True   # Mark the start of the next valid range
        
        # Be sure to exclude the first and last few points which weren't checked over in the previous code block
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
            self.df.loc[last_start_ind:len(self.df)-1, 'IsValidPoint'] = False   # Get rid of last valid points
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'IsStartPoint','IsEndPoint']].head(20))
        #display(self.df[['TimeElapsed','IsJump', 'IsSwitch', 'IsValidPoint', 'IsStartPoint','IsEndPoint']].tail(20))
        
        #return jump_fig_1, jump_fig_2
        return
    
    ### This is a user function used to take the valid sections found within find_switches,
    ### and separate the data into the two polarization signals, averaging the data within each valid section
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
        # For each gap we create and store the necessary rows; at the end, we add them all
        # to the final dataframe and sort by time to get everything back in order
        def interpolate(df, time_gap, print_process):
            if print_process: print('--- interpolating data ---')
            #print_interpolate = print_process
            print_interpolate = False
            # Create a new DataFrame to store the interpolated values
            columns = df.columns
            interpolated_df = pd.DataFrame(columns=columns)

            df['TimeDiff'] = df['EstTime'].diff()
            # Iterate through the rows to interpolate missing values for each column
            interpolated_data_list = []
            for i in range(len(df) - 1):
                row1 = df.iloc[i]
                row2 = df.iloc[i + 1]

                # Calculate the number of missing observations between the two rows
                # Normally TimeDiff / time_gap should be 1.0
                # If it's between 1.5 and 2.5, we say there is a single gap
                # between 2.5 and 3.5, a double gap
                # Subtract 0.5 and then floor to get the number of missing points
                num_missing = int(np.floor(row2['TimeDiff'] / time_gap - 0.5))

                if num_missing > 0:
                    if print_interpolate: print('Imputing '+BOLD_ON+'{} points'.format(num_missing)+BOLD_OFF+' at '+BOLD_ON+'t={:.1f}'.format(row1['EstTime'])+BOLD_OFF+' of length '+BOLD_ON+'{:.3f} seconds'.format(row2['TimeDiff'])+BOLD_OFF+' (vs nominal switch time of '+BOLD_ON+'{:.3f} seconds'.format(time_gap)+BOLD_OFF+')')
                    # Generate and append missing rows with interpolated timestamps and data for each column
                    # For each gap, we insert an observation containing linearly interpolated (averaged) data between the
                    # recorded observations immediately preeceeding and following the missing one
                    # Some attributes are imputed to show this data was interpolated (e.g. NumPoints = -1)
                    for j in range(1, num_missing + 1):
                        interpolated_data = {}
                        for column in columns:
                            if df[column].dtype in [float]:
                                interpolated_data[column] = row1[column] + (j * (row2[column] - row1[column])) / (num_missing + 1)
                            else:
                                interpolated_data[column] = row1[column]
                        interpolated_data['NumPoints'] = -1
                        interpolated_data['TimeDiff'] = interpolated_data['AvgTime'] = 'NaN'
                        interpolated_data['StartTime'] = interpolated_data['EndTime'] = 'NaN'
                        interpolated_data['Length'] = interpolated_data['AvgSampleRate'] = 'NaN'
                        # Add interpolated observation to list; later we will turn into dataframe
                        interpolated_data_list.append(interpolated_data)

            # After the loop, convert the list of interpolated observations to a DataFrame
            interpolated_df = pd.DataFrame(interpolated_data_list)
            # Concatenate the original DataFrame and the interpolated DataFrame
            result_df = pd.concat([df, interpolated_df]).sort_values(by='EstTime').reset_index(drop=True)

            if print_process: print('length before: {}'.format(len(df)))
            if print_process: print('length after: {}'.format(len(result_df)))
            if print_process: print('{} total added'.format(len(result_df)-len(df)))

            result_df.drop(columns=['TimeDiff'], inplace=True)

            if print_process: print('--- end interpolating data ---')
            return result_df
        
        new_dataframes = {}
        for signal_number, signal_df in signal_dataframes.items():
            signal_df.reset_index(drop=True, inplace=True)
            signal_df.drop(columns='signalMembership', inplace=True)   # Don't need this column anymore
            if print_process: print('')
            new_df = interpolate(signal_df, 2*self.switch_time, print_process)
            new_dataframes[signal_number]=new_df  
        
        # Store our two signal outputs
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
        
        # Switch (or don't) the signals and warn if the difference is too hard to tell
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
            stokes_1_first, stokes_2_first = self.first_stokes()
            print('\tFirst measurement of Signal 1: ', stokes_1_first)
            print('\tFirst measurement of Signal 2: ', stokes_2_first)
        
        self.df.drop(columns=['dif', 'mask'], inplace=True)   # At the end, drop these columns from the original dataframe
        if print_process: print(BOLD_ON+'=== end average_data ==='+BOLD_OFF)
        #return avg_df
        return
    
    ### Returns the average polarization state of both signals
    ### Meant for short datasets, like those used for the input measurements
    def average_stokes(self):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return
        
        # Base z scores off of change_point_param
        switch_param = self.change_point_params[0][3]
        z_score_param = 'S' + switch_param[1] + 'Avg'   # E.g. "S2Avg"
        
        # Get z scores of the averages for the switch parameter used
        signal_1_z_scores = np.abs((self.signal_1_df[z_score_param] - self.signal_1_df[z_score_param].mean()) / self.signal_1_df[z_score_param].std())
        signal_2_z_scores = np.abs((self.signal_2_df[z_score_param] - self.signal_2_df[z_score_param].mean()) / self.signal_2_df[z_score_param].std())
        # Set a Z-score threshold for outliers (e.g., 2)
        z_score_threshold = 2
        # Filter out values above the Z-score threshold
        signal_1_df_filtered = self.signal_1_df[signal_1_z_scores <= z_score_threshold]
        signal_2_df_filtered = self.signal_2_df[signal_2_z_scores <= z_score_threshold]
        
        # For the non-outlier rows, get the averages of S0, S1, etc
        signal_1_stokes = [signal_1_df_filtered[param].mean() for param in ['S1Avg','S2Avg','S3Avg']]
        signal_2_stokes = [signal_2_df_filtered[param].mean() for param in ['S1Avg','S2Avg','S3Avg']]
        
        return signal_1_stokes, signal_2_stokes
    
    ### Returns the first observed polarization state of both signals
    def first_stokes(self):
        signal_1_stokes = [self.signal_1_df[param][0] for param in ['S1Avg','S2Avg','S3Avg']]
        signal_2_stokes = [self.signal_2_df[param][0] for param in ['S1Avg','S2Avg','S3Avg']]
        return signal_1_stokes, signal_2_stokes
    
    ### Takes a reference polarization (input_stokes = ([S0, S1, S2, S3], [S0, S1, S2, S3]))
    ### and calculates the similarity metric between each observation and the reference polarization (e.g. dot product)
    def calc_similarity(self, input_stokes='first', angle_threshold_deg=None):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return
        
        # Determine the reference polarization we're using
        # if input_stokes is "average" or "first", we calculate it here
        # otherwise we expect the input stokes arrays to be given
        if isinstance(input_stokes, str):
            if input_stokes=='average':
                signal_1_stokes, signal_2_stokes = self.average_stokes()
                if angle_threshold_deg is not None:
                    raise Exception('Cannot use ~average~ for input stokes if defining a threshold')
            if input_stokes=='first':
                signal_1_stokes, signal_2_stokes = self.first_stokes()
        if isinstance(input_stokes, tuple):
            signal_1_stokes, signal_2_stokes = input_stokes
            if angle_threshold_deg is not None:
                raise Exception('Cannot use custom input stokes if defining a threshold')
        self.signal_1_stokes = signal_1_stokes
        self.signal_2_stokes = signal_2_stokes
        
        ### We calculate the similarity metric between each observation and a reference polarization
        # We apply the metric to each signal dataframe, using the un-normalized stokes parameters in the signal
        # We use the un-normalized stokes parameters because they contain many more significant digits
        if angle_threshold_deg is None:
            for input_stokes, df in [(signal_1_stokes, self.signal_1_df), (signal_2_stokes, self.signal_2_df)]:
                df['distSimMetric'] = df.apply(lambda row: Dataset.dist_metric(input_stokes, [row['S1Avg'], row['S2Avg'], row['S3Avg']]), axis=1)
                df['angleSimMetric'] = df.apply(lambda row: Dataset.angle_metric(input_stokes, [row['S1Avg'], row['S2Avg'], row['S3Avg']]), axis=1)
        else:
            thing=1
            # iterate over signals, resetting input stokes when metric crosses threshold
        
        return

    ### Just calculates angle (in radians) between the two vectors (to make code below more readable)
    ### Assumes unnormalized vectors
    @staticmethod
    def angle_between_SOPs(stokes_vector_1, stokes_vector_2):
        vector_dot = np.dot(stokes_vector_1, stokes_vector_2)
        n1 = np.linalg.norm(stokes_vector_1)
        n2 = np.linalg.norm(stokes_vector_2)
        dot_normalized = vector_dot / (n1 * n2)
        angle = np.rad2deg(np.arccos(np.clip(dot_normalized, -1.0, 1.0))) # Clip to [-1,1] for numerical stability
        return angle
    
    ### Used by calc_stokes_ptf()
    ### Func to reverse-engineer the STOKES matrix from the input polarization and time-dependent output polarizations
    ### signal_1_current_row is one observation in the averaged signal 1 (same for signal_2_current_row)
    ### These rows should already be aligned in time
    ### Assume unnormalized stokes vectors
    @staticmethod
    def calc_stokes_rotation(reference_matrix, signal_1_current_row, signal_2_current_row):
        try:
            time = (signal_1_current_row['EstTime'] + signal_2_current_row['EstTime']) / 2
            if abs(signal_1_current_row['EstTime'] - signal_2_current_row['EstTime']) > 1:
                print('Check alignment of signals; if >1 second between adjacent averaging points, let em know')
        except TypeError as e:   # debugging
            print(e)
            print('signal_1_current_row')
            display(signal_1_current_row)
            print('signal_2_current_row')
            display(signal_2_current_row)
            raise
        
        ### Get the stokes parameters of the current row
        signal_1_current_stokes = np.array((signal_1_current_row['S1Avg'], signal_1_current_row['S2Avg'], \
                                                signal_1_current_row['S3Avg']))
        signal_2_current_stokes = np.array((signal_2_current_row['S1Avg'], signal_2_current_row['S2Avg'], \
                                                signal_2_current_row['S3Avg']))
        current_matrix = np.column_stack(SwitchSet.construct_orthonormal_bases(signal_1_current_stokes, \
                                                                              signal_2_current_stokes))
        
        ### Here is the meat of the function; we calculate the rotation matrix and deduce the axis/angle representation
        rotation_matrix = np.dot(current_matrix, reference_matrix.T)
        axis = np.array([rotation_matrix[2,1] - rotation_matrix[1,2],
                         rotation_matrix[0,2] - rotation_matrix[2,0],
                         rotation_matrix[1,0] - rotation_matrix[0,1]])
        angle = np.rad2deg(np.arccos((np.trace(rotation_matrix)-1)/2))

        # Get angle between SOPs of the two output signals; should be similar to angle between SOPs of input signals
        # TODO: calculate angle between SOPs of input signals
        angle_between_current_stokes = SwitchSet.angle_between_SOPs(signal_1_current_stokes, signal_2_current_stokes)

        return pd.Series([time, current_matrix, signal_1_current_stokes, signal_2_current_stokes, \
                          axis, angle, angle_between_current_stokes, 0.0, False],
                         index=SwitchSet.STOKES_PTF_DF_COLUMNS)
    
    ### Constructs 3 orthonormal bases given two linearly independent vectors
    ### See notebook for details
    ### Assumes unnormalized input vectors
    # TODO: see below
    @staticmethod
    def construct_orthonormal_bases(p, q):
        p = np.array(p).astype(float)
        a = p
        a /= np.linalg.norm(a)

        q = np.array(q).astype(float)
        b = q - np.dot(q,p) * p
        # TODO: do we need to normalize in above equation after dot?
        b /= np.linalg.norm(b)

        c = np.cross(a,b)
        c /= np.linalg.norm(c)

        return a, b, c
    
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

        # Determine what criteria for the reference SOPs we're using
        resetting_approach = False   # Witness beam channel
        reset_times_given = False   # Other channel
        if (reference is None):
            print("Using \"resetting reference SOPs\" approach with threshold = {:.1f} degrees".format(angle_threshold_deg))
            resetting_approach = True
            self.reset_by_rolling = reset_by_rolling
            self.angle_threshold_deg = angle_threshold_deg
            signal_1_input_stokes, signal_2_input_stokes = self.first_stokes()
            self.signal_1_input_stokes = signal_1_input_stokes
            self.signal_2_input_stokes = signal_2_input_stokes
        elif (isinstance(reference, str)):
            print("Using first recorded SOPs as reference matrix")
            signal_1_input_stokes, signal_2_input_stokes = self.first_stokes()
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
            signal_1_input_stokes, signal_2_input_stokes = self.first_stokes()
            self.signal_1_input_stokes = signal_1_input_stokes
            self.signal_2_input_stokes = signal_2_input_stokes
        else:
            print("No approach found; provide proper \"reference\" parameter")
        print("Averaging rotAngle using " + BOLD_ON + "{:d} rolling points".format(rolling_pts) + BOLD_OFF)
        
        # First reference SOPs
        reference_matrix = np.column_stack(SwitchSet.construct_orthonormal_bases(signal_1_input_stokes, signal_2_input_stokes))

        ### Iterate over the dataset, calculating the stokes rotation matrix between the observed pair of SOPs and the reference
        ### reference_matrix may reset depending on the criteria above
        rows = list(zip(self.signal_1_df.iterrows(), self.signal_2_df.iterrows()))
        stokes_ptf_rows = []  # List to store the Series objects
        reset_times = []   # Times at which we reset the reference
        current_segment_values = []   # Initialize a list to temporarily store values of the current rolling average segment
        for (i1, row1), (i2, row2) in rows:
            result_row = SwitchSet.calc_stokes_rotation(reference_matrix, row1, row2)  # Calculate rotation matrix
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
                reference_matrix = np.column_stack(SwitchSet.construct_orthonormal_bases(signal_1_reference_stokes, signal_2_reference_stokes))
            
            stokes_ptf_rows.append(result_row)
        
        # Print some stuff
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

    def compute_hourly_segments(self):
        """
        Computes hourly segments from the main DataFrame (df) by determining the ranges 
        of 'TimeElapsed' for each hour, and then maps these to corresponding 'EstTime' ranges 
        and indices in another DataFrame (stokes_ptf_df).
    
        This method initializes and populates a dictionary (`segment_dict`) where each key is a tuple
        indicating the day and hour (starting from the initial timestamp in `df`), and the value is
        another dictionary containing ranges for 'TimeElapsed', 'EstTime', and indices in `stokes_ptf_df`.
    
        The dictionary provides a structured way to access time segments which encompass full-hour intervals
        based on the data present in `df`.
    
        Returns:
            dict: A dictionary with keys as (day, hour) tuples and values containing dictionaries with:
                  - 'TimeElapsedRange': Tuple indicating the start and end of the TimeElapsed range.
                  - 'EstTimeRange': Tuple indicating the start and end of the EstTime range in `stokes_ptf_df`.
                  - 'IdxRange': Tuple indicating the start and end indices in `stokes_ptf_df`.
    
        Raises:
            ValueError: If the initial Timestamp data in `df` is not properly formatted or missing.
        """
        # Initialize the dictionary to store the results
        segment_dict = {}

        # Get the initial timestamp from df
        initial_timestamp = self.df['Timestamp'].iloc[0]
        initial_day = 0  # Day starts from 0
        initial_hour = initial_timestamp.hour

        # Calculate the number of seconds from the beginning of the dataset to the start of the first full hour
        seconds_to_first_full_hour = (60 - initial_timestamp.minute) * 60 - initial_timestamp.second
        
        # Iterate over each full hour after the initial time
        start_time = initial_timestamp + pd.Timedelta(seconds=seconds_to_first_full_hour)
        end_time = start_time + pd.Timedelta(hours=1)
        idx = 0

        while end_time <= self.df['Timestamp'].iloc[-1] + pd.Timedelta(hours=1):
            current_day = (start_time - initial_timestamp).days
            current_hour = start_time.hour
            
            # Find indices in df that are within the current hour segment
            mask_df = (self.df['Timestamp'] >= start_time) & (self.df['Timestamp'] < end_time)
            relevant_df = self.df.loc[mask_df]
            
            if not relevant_df.empty:
                # TimeElapsed range
                timestamp_range = (relevant_df['Timestamp'].min(), relevant_df['Timestamp'].max())
                time_elapsed_range = (relevant_df['TimeElapsed'].min(), relevant_df['TimeElapsed'].max())
                df_idx_range = (relevant_df.index.min(), relevant_df.index.max())
                
                # Corresponding indices in stokes_ptf_df
                mask_stokes = (self.stokes_ptf_df['EstTime'] >= time_elapsed_range[0]) & \
                              (self.stokes_ptf_df['EstTime'] <= time_elapsed_range[1])
                relevant_stokes = self.stokes_ptf_df.loc[mask_stokes]
                
                # EstTime range and indices range
                est_time_range = (relevant_stokes['EstTime'].min(), relevant_stokes['EstTime'].max())
                stokes_idx_range = (relevant_stokes.index.min(), relevant_stokes.index.max())

                # Add to dictionary
                segment_dict[(current_day, current_hour)] = {
                    'TimestampRange': timestamp_range,
                    'TimeElapsedRange': time_elapsed_range,
                    'DfIdxRange': df_idx_range,
                    'EstTimeRange': est_time_range,
                    'StokesIdxRange': stokes_idx_range
                }

            # Move to the next hour
            start_time += pd.Timedelta(hours=1)
            end_time += pd.Timedelta(hours=1)
            idx += 1

        self.segment_dict = segment_dict
        return segment_dict

    def print_segment_dict(self):
        """
        prints keys and values of dataset segments
        """
        for (key, value) in self.segment_dict.items():
            print(key)

    def get_segment_adev(self, key):
        """
        Processes a specific segment of `stokes_ptf_df` based on the given key which corresponds
        to a tuple of (day, hour). This method retrieves the indexed range from `segment_dict`
        and performs data analysis on the `rotAngleDif` values within this segment.
    
        The function drops any NaN values from `rotAngleDif`, calculates the cumulative sum,
        and then performs Allan deviation analysis on this data. The results, along with other
        metadata like a generated title and time string, are returned as a tuple.
    
        Parameters:
            key (tuple): A tuple of (day, hour) that identifies the specific hour segment
                         for which the data should be processed.
    
        Returns:
            tuple: Contains the results of the Allan deviation analysis, the set title,
                   and a generated time string, structured as:
                   - Allan deviation results
                   - Title (str)
                   - Time string (str)
    
        Raises:
            KeyError: If the provided key does not exist in `segment_dict`.
            IndexError: If the indexed range is out of bounds for `stokes_ptf_df`.
        """
        # Retrieve the index range from the segment_dict for the given key
        value_dict = self.segment_dict[key]
        stokes_idx_range = value_dict['StokesIdxRange']
        
        # Slice the stokes_ptf_df dataframe using the retrieved index range
        stokes_ptf_df_slice = self.stokes_ptf_df.iloc[stokes_idx_range[0]:stokes_idx_range[1]+1]

        # Perform the specified operations
        dif_data = stokes_ptf_df_slice['rotAngleDif'].dropna()
        walk_data = np.cumsum(dif_data)
        meas_rate = 1 / (2 * self.switch_time)
        elm = allantools.oadev(walk_data.values, rate=meas_rate, taus='all', data_type="freq")
        timestamp_range = value_dict['TimestampRange']
        time_str_start = timestamp_range[0].strftime('%H:%M')
        time_str_end = timestamp_range[1].strftime('%H:%M')
        time = '{:s} - {:s}'.format(time_str_start,time_str_end)
        set_title = self.title
        
        # Combine all parameters into a tuple
        params = (*elm, set_title, time)
        
        return params

    def calc_adev_divided(self):
        """
        Divides the dataset into hourly segments using "compute_hourly_segments()", and
        calculates adev for each segment. params_arr can be plotted using plotting.plot_adev
        """
        self.compute_hourly_segments()
        adev_params_arr = []
        for key in self.segment_dict.keys():
            adev_params_arr.append(self.get_segment_adev(key))
        self.adev_params_arr = adev_params_arr
        return adev_params_arr
    
    ### Used by calc_jones_ptf()
    ### Func to reverse-engineer the jones matrix from the input polarization and time-dependent output polarizations
    ### row1 is one observation in the averaged signal 1
    @staticmethod
    def calc_jones_rotation(I, row1, row2):
        try:
            time = (row1['EstTime'] + row2['EstTime']) / 2
            if abs(row1['EstTime'] - row2['EstTime']) > 1:
                print('Check alignment of signals; >1 second between adjacent averaging points')
        except TypeError as e:
            print(e)
            print('row1')
            display(row1)
            print('row2')
            display(row2)
            raise
        O = np.array([[row1['JonesUpper'],row2['JonesUpper']],
                      [row1['JonesLower'],row2['JonesLower']]])
        M = np.dot(O,la.inv(I))
        A1,A2,B1,B2 = M[0][0], M[1][1].conjugate(), M[0][1], -M[1][0].conjugate()
        return pd.Series([time,row1['JonesUpper'],row1['JonesLower'],row2['JonesUpper'],row2['JonesLower'],A1,A2,B1,B2],
                         index=['Time','signal_1_jv_upper','signal_1_jv_lower','signal_2_jv_upper','signal_2_jv_lower','A1','A2','B1','B2'])

    ### Takes a reference polarization (input_stokes = ([S1, S2, S3], [S1, S2, S3]))
    ### and calculates the rotation matrix (in JONES form) between each observation and the reference polarization
    def calc_jones_ptf(self, input_stokes='first', angle_threshold_deg=None):
        if self.signal_1_df is None:
            print('Error: averages not yet calculated')
            return
        
        # Determine the reference polarization we're using
        # if input_stokes is "average" or "first", we calculate it here
        # otherwise we expect the input stokes arrays to be given
        if isinstance(input_stokes, str):
            if input_stokes=='average':
                signal_1_stokes, signal_2_stokes = self.average_stokes()
                if angle_threshold_deg is not None:
                    raise Exception('Cannot use ~average~ for input stokes if defining a threshold')
            if input_stokes=='first':
                signal_1_stokes, signal_2_stokes = self.first_stokes()
        if isinstance(input_stokes, tuple):
            signal_1_stokes, signal_2_stokes = input_stokes
            if angle_threshold_deg is not None:
                raise Exception('Cannot use custom input stokes if defining a threshold')
        self.signal_1_stokes = signal_1_stokes
        self.signal_2_stokes = signal_2_stokes
        
        ### We calculate the JONES rotation matrix between the observation and reference polarization
        # Input array of polarizations
        signal_1_jv_upper, signal_1_jv_lower = mueller.stokes_to_jones(np.array([np.linalg.norm(signal_1_stokes),*signal_1_stokes]))
        signal_2_jv_upper, signal_2_jv_lower = mueller.stokes_to_jones(np.array([np.linalg.norm(signal_2_stokes),*signal_2_stokes]))
        I = np.array([[signal_1_jv_upper,signal_2_jv_upper],
                      [signal_1_jv_lower,signal_2_jv_lower]])
        #self.input_polarization = I

        # Calculate the jones states for each signal, then the jones rotation matrix for each observation
        for df in (self.signal_1_df, self.signal_2_df):
            # Calculate the polarization from the stokes vectors for all the averages
            df[['JonesUpper','JonesLower']] = df.apply(lambda row: mueller.stokes_to_jones(np.array([row['S0Avg'],row['S1Avg'],row['S2Avg'],row['S3Avg']])),axis=1,result_type='expand')
        rows = list(zip(self.signal_1_df.iterrows(), self.signal_2_df.iterrows()))
        self.transfer_func_df = pd.DataFrame([SwitchSet.calc_jones_rotation(I, row1, row2) for (i1, row1), (i2, row2) in rows])
        self.transfer_func_df['Time'] = self.transfer_func_df['Time'].astype(float)
        
        return
    
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
        
        label = '{:s} (ADev) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df))
        label_1 = '{:s} (ADev) | {:s}, Signal {:d} | {:s}'.format(plot_param, self.title, 1, Dataset.gen_time_str(self.df))
        label_2 = '{:s} (ADev) | {:s}, Signal {:d} | {:s}'.format(plot_param, self.title, 2, Dataset.gen_time_str(self.df))
        
        adev_arr = []
        if twice:
            adev_arr.append((taus2_1, ad_1, ade_1, ns_1, label_1))
            adev_arr.append((taus2_2, ad_2, ade_2, ns_2, label_2))
        else:
            adev_arr.append((taus2, ad, ade, ns, label))
        
        ADev_fig = None
        if plot_adev:
            ADev_fig = StationarySet.plot_adev(adev_arr)
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