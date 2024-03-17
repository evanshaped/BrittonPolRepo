"""
=== Paradigm used for finding transition times ===


For any arbitrary signal we intend to measure, there are a few types of deviations we expect

Deviation 1: Outliers
Data points measured well above the expected maximum polarization or well below the expected minimum. So far as I've seen, these only occur at the transition times and rarely deviate more than 1 times the length of our typical jumps. These may be the result of confusion in the polarimeter as during one measurement cycle, it receives data partly from the first polarization and partly from the second. Maybe a weird behavior in the FFT?

Deviation 2: 'Limbo Points'
These are data measured in between the two expected polarizations. They separate the jump between the two, dividing the entire length of the jump into two adjacent differences. Unsure whether they occur for the same reason as outliers (confused PAX), or if maybe the PAX averages the two polarizations. Important to note that both Outliers and Limbo Points frequently appear at the same time across multiple paramters (expected).

Deviation 3: Polarization Drift
An expected deviation resulting from instability in the setup or the PTF of the fiber under test. We encode the maximum expected rate of drift in the 'degree_tolerance' parameter.

Methedology:
Since transitions occur very suddenly, we take the differences between all adjacent points and an arbitrary difference is likely to represent a transition time if it's magnitude is very large compared to the magnitude of the differences between adjacent points taken from the same signal (near 0).

Considerations:

Outliers (assuming they only occur at transition times) will increase the magnitude of the difference jump at transitions, and add another difference directly after the sharply increased first difference. We will account for this by... TODO

Limbo Points: (decreases magnitude of jumps; we will take differences between every 2 points to ensure each jump is captured. Side effect of roughly doubling number of jumps)
TODO

Polarization Drift:
TODO
"""



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



"""
**Dataset(filename, set_range=None)**
<br>The dataset class is designed to read, store, and perform operations on a CSV dataset from the PAX1000 digital polarimeter.
Cleaning of the CSV file happens automatically upon initialization of a Dataset instance, and assumes the CSV was output from the PAX1000 software (e.g., there are 7 header rows containing the dataset parameters, 1 row of whitespace, and 1 row of the column names, below which are the data (observations)). See the PAX [manual](https://www.manualslib.com/manual/1634678/Thorlabs-Pax1000.html?page=39#manual) for details on the recorded quantities.

**Parameters**
<br>*filename : string*
> Path to PAX csv file to analyze, e.g. "Datasets/my_dataset.csv" if your csv file is in a folder named 'Datasets'

*set_range : None or tuple of two floats, optional*
> If None, the entire dataset will be analyzed (default behavior).
> <br>Use tuple of two floats to specify a part of a dataset to read in, given via percentages. This is usually useful to save time while reading in a small portion of a large dataset. For example, set_range=(0.05,0.1) means the chunk of the dataset between 5%-10%.
"""



"""
**Dataset.rate_hist(log=True, bins=50, xmax=None)**
<br>Visualize the consistency of the PAX's measurement rate. Generate and plot the histogram of time differences between subsequent points.

**Parameters**
<br>*log : boolean, optional*
> Whether to make the y-axis log-scale. Emphasizes uncommon occurrences. Default is True.

*bins : int, optional*
> Increase or decrease bin width depending on size and range of dataset. Default is 50.

*xmax : None or float, optional*
> Set a maximum on the x-axis (for easier viewing) if there is one outlying point far out on the x-axis.
"""



"""
**Dataset.plot(plot_param='s1', sample_range=None, birds_eye=True)**
<br>Plot some parameter in the dataset over time.

**Parameters**
<br>*plot_param : string, optional*
> The parameter(s) to plot. Default is "AllStokes" (all normalized stokes parameters). Options are:
> - AllStokes, s1, s2, s3, S0, S1, S2, S3, Azimuth, Ellipticity, DOP, DOCP, DOLP, Power, Power_pol, Power_unpol

*sample_range : None or tuple of two floats, optional*
> A portion of the plot to zoom in on, given in seconds, e.g. sample_range=(50,100)

*birds_eye : boolean, optional*
> Whether or not to show the birds eye view
"""

BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
#import os
#import seaborn as sns
import warnings


class Dataset:
    stokes_units = 'W/m^2'
    unitless = 'unitless'
    radians = 'Rad'; degrees = 'Degrees'
    ANGLE_UNITS_IS_RAD = False
    angle_units = radians if ANGLE_UNITS_IS_RAD else degrees
    DEGREES_COLUMNS = ["Azimuth", "AzimuthAvg", "AzimuthStd", "Ellipticity", "EllipticityAvg", "EllipticityStd", "angleSimMetric", "rotAngle", "angleDif"]
    RADIANS_COLUMNS = []
    UNITS = {'S0':stokes_units, 'S1':stokes_units, 'S2':stokes_units, 'S3':stokes_units, 
                      'S0Avg':stokes_units, 'S1Avg':stokes_units, 'S2Avg':stokes_units, 'S3Avg':stokes_units, 
                      'S0Std':stokes_units, 'S1Std':stokes_units, 'S2Std':stokes_units, 'S3Std':stokes_units,
                      's1':unitless, 's2':unitless, 's3':unitless, 'AllStokes':unitless,
                      's1Avg':unitless, 's2Avg':unitless, 's3Avg':unitless, 
                      's1Std':unitless, 's2Std':unitless, 's3Std':unitless, 
                      'Azimuth':angle_units, 'Ellipticity':angle_units, 
                      'AzimuthAvg':angle_units, 'EllipticityAvg':angle_units, 
                      'AzimuthStd':angle_units, 'EllipticityStd':angle_units, 
                      'NumPoints':'number',
                      'DOP':'%', 'DOCP':'%', 'DOLP':'%', 'Power':'mW', 'Power_pol':'mW', 'Power_unpol':'mW',
                      'DOPAvg':'%', 'PowerAvg':'mW',
                      'DOPStd':'%', 'PowerStd':'mW',
                      'distSimMetric':unitless, 'angleSimMetric':angle_units, 
                      'rotAngle':angle_units, 'angleDif':angle_units}
    
    def __init__(self, filename, set_range=None, time_offset=0.0, skip_default_signal_baseline=True):
        self.title = filename[filename.rindex("PAX"):-4]
        self.filename = filename
        self.plot_param = 'Azimuth'
        
        # Cleaning of the data happens in Dataset.read_pax_data
        df, num_points_dropped, device_id, serial_num, wavelength, basic_sample_rate, op_mode_period, \
            op_mode_FFT_num = Dataset.read_pax_data(filename,set_range,time_offset, skip_default_signal_baseline)
        self.device_id = device_id
        self.serial_num = serial_num
        self.wavelength = wavelength
        self.basic_sample_rate = basic_sample_rate
        self.op_mode_period = op_mode_period
        self.op_mode_FFT_num = op_mode_FFT_num
        self.num_points_dropped = num_points_dropped
        self.df = df
        self.num_points = df.shape[0]
        self.time_elapsed = df['TimeElapsed'][self.num_points-1]
        
        self.mintime = df.loc[0, 'TimeElapsed']
        self.maxtime = df.loc[df.shape[0]-1, 'TimeElapsed']
        print('Time range: min={}, max={}'.format(self.mintime, self.maxtime))
        
        self.nominal_sample_rate = basic_sample_rate / (2*op_mode_period)
        self.avg_sample_rate = 1/(df['TimeDiff'][1:].mean())   # See rate_hist for why this code
        return
    
    @staticmethod
    def fill_in_range(my_range, my_df):
        # Fill in the start or end if needed (if None is given instead of value)
        # If a negative time -t is given, it means t_max-t
        if my_range[0] is None:
            my_start = my_df.loc[0, 'TimeElapsed']
        elif my_range[0] < 0:   # If a negative time -t is given, it means t_max-t
            my_start = my_df.loc[my_df.shape[0]-1, 'TimeElapsed'] + my_range[0]
        else:
            my_start = my_range[0]
        if my_range[1] is None:
            my_end = my_df.loc[my_df.shape[0]-1, 'TimeElapsed']
        elif my_range[1] < 0:   # If a negative time -t is given, it means t_max-t
            my_end = my_df.loc[my_df.shape[0]-1, 'TimeElapsed'] + my_range[1]
        else:
            my_end = my_range[1]
        return my_start, my_end
    
    # skip_default_signal_baseline = True iff the first 10 seconds (no more than 20) are used only to identify
    # which signal is the default one. These seconds will be automatically cut out. A value of t seconds can also
    # be assigned, to give a custom number of seconds to skip (e.g. 100 if the first 100 seconds contains just one signal)
    @staticmethod
    def read_pax_data(filename, set_range=None, time_offset=0.0, skip_default_signal_baseline=True):
        # Get one-off information in the header
        df_temp = pd.read_csv(filepath_or_buffer=filename, delimiter=';', header=None, usecols=[1], nrows=7)
        device_id = df_temp.iat[0,0]
        serial_num = df_temp.iat[1,0]
        wavelength = float(df_temp.iat[4,0])
        basic_sample_rate = float(df_temp.iat[5,0])
        op_mode = df_temp.iat[6,0] #operating mode string
        p_end_index = op_mode.find(' rev')
        n_start_index = op_mode.find('ent, ') + 5
        n_end_index = op_mode.find(' poi')
        op_mode_period = float(op_mode[:p_end_index]) #P part of string
        op_mode_FFT_num = int(op_mode[n_start_index:n_end_index]) #N part of string
        # File looks like 7 rows of info (do display(df.head())), 1 row of whitespace, 1 row of column names, then the data
        
        ### We begin to take care of the range considerations. set_range can either be:
        # None, (time_start,time_end), (per_start,per_end)
        # If None, we use the whole dataset
        # If (time_start,time_end), we read in the entire dataset and then chop off the seconds we want (this
        # is usually useful to get rid of the first/last 10 minutes)
        # If (per_start,per_end), we'll only read in from per_start percentage of the dataset to per_end (these
        # parameters must be given between 0.0 and 1.0) (this is usually useful to save time while reading in
        # a small portion of a large dataset). For example, (0.05, 0.1) means the chunk of the dataset between 5%-10%
        with open(filename) as f:
            total_rows = sum(1 for line in f) - 9   # The first 9 rows are info, whitespace, and column names
        time_range = None   # Initially assume we don't want to cut off the times
        skiprows = nrows = None   # Initally assume we read all of the file
        if set_range is not None:
            per_start, per_end = set_range
            if ((per_start is not None) and (per_start>0) and (per_start<1)) or \
                ((per_end is not None) and (per_end>0) and (per_end<1)):
                #print('percentages detected')
                # set_range parameters refer to percentages (e.g. (0.4,0.6) is 40% to 60% of the dataset)
                # If per_start isn't None, skiprows is a number. If per_end isn't None, nrows is rows to use
                skiprows = None if per_start is None else int(total_rows * per_start)
                nrows = None if per_end is None else int(total_rows * (per_end-(0.0 if per_start is None else per_start)))
                # If either of these were changed from None, read_csv will notice and use them appropriately
            else:
                #print('times detected')
                time_range = set_range   # O/w, time_range is in seconds; this will be cut further on
                set_range = None
        # Coming out of this, either both time_range and set_range are None (no chopping necessary),
        # or time_range is None but set_range has percentage start and ends (use in read_csv via skiprows and nrows),
        # or set_range is None but time_range has time start and ends (cut further on)
        
        # TODO: fix warning
        # Catching a FutureWarning for infer_datetime_formet;
        # the code works and I don't want to deal with the red text right now
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ### read in, assign labels and variable types
            df = pd.read_csv(filepath_or_buffer=filename, delimiter=';', header=7, skiprows=skiprows, nrows=nrows,
                             parse_dates=['Timestamp'], infer_datetime_format=True,
                             names=['Timestamp', 'TimeDelta', 's1', 's2', 's3', 'S0', 'S1', 'S2', 'S3',
                                    'Azimuth', 'Ellipticity', 'DOP', 'DOCP', 'DOLP', 'Power', 'Power_pol', 'Power_unpol',
                                    'Power_total_dB', 'Power_pol_dB', 'Power_unpol_dB', 'Power_split_ratio',
                                    'Phase_difference', 'Warning'],
                             dtype={'Warning': 'string'})

        # drop unneeded columns
        df.drop(columns=['Power_total_dB', 'Power_pol_dB', 'Power_unpol_dB', 'Warning'],inplace=True)

        indices_with_missing_values = df[df.isna().any(axis=1)].index # Find the indices of rows with missing values
        df = df.drop(indices_with_missing_values) # Drop rows with missing values
        dfd = df.reset_index(drop=True) # Re-index the DataFrame
        num_points_dropped = len(indices_with_missing_values)

        # format timestamp
        def custom_to_timedelta(s):
            days, time = s.split('.')
            hours, minutes, seconds, milliseconds = map(int, time.split(':'))
            return timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        df['TimeDelta'] = df['TimeDelta'].apply(custom_to_timedelta) # convert to timedelta
        df['TimeElapsed'] = df['TimeDelta'].apply(lambda x: x.total_seconds()) # Seconds elapsed since first measurement
        # The raw DateTime and TimeDelta objects are no longer needed
        #df.drop(columns=['Timestamp','TimeDelta'], inplace=True)
        df.drop(columns=['TimeDelta'], inplace=True)   # Actually, want to display global datetime on plots
        
        ### If desired, we only want to keep the range specified (time_range is given in seconds)
        # This is mutually exclusive with cutting based on percentages given in set_range (see above)
        # We do not reset the timestamps to be 0.000s at the first entry, but we do reset the indices
        if time_range is not None:
            set_start, set_end = Dataset.fill_in_range(time_range, df)   # automatically fills in missing range vals
            set_start += time_offset   # include offset
            set_end += time_offset
            # Throw warning if including out of range data
            if set_end > df.loc[df.shape[0]-1, 'TimeElapsed']:
                print('WARNING: data out of range specified')
                print('set_end={:.2f} > {:.2f}'.format(set_end,df.loc[df.shape[0]-1, 'TimeElapsed']))
            if set_start < df.loc[0, 'TimeElapsed']:
                print('WARNING: data out of range specified')
                print('set_start={:.2f} < {:.2f}'.format(set_start,df.loc[0,'TimeElapsed']))
            df = df[(set_start<=df['TimeElapsed']) & (df['TimeElapsed']<=set_end)].reset_index(drop=True)

        
        ### Here, we also include functionality to automatically cut off the first 20 seconds of the dataset
        # (if it is the *true* first 20 seconds; not the first 20 seconds of a 5%-10% dataset)
        # If it is a 5%-10% dataset, the timestamps will start partway through the dataset (e.g. 2500-3500 seconds),
        # and this will have no effect (as desired)
        # This is optionally performed because I often use the first 10 seconds of every SwitchSet to solely
        # measure the default signal (not switching between the two signals), so we can match which signal is which
        # between the input and output SwitchSets or between the 1345 and 1560 output SwitchSets
        if skip_default_signal_baseline is not False:
            time_skip = 20 if skip_default_signal_baseline is True else skip_default_signal_baseline
            df = df[time_skip<=df['TimeElapsed']].reset_index(drop=True)
        
        
        ### Calculate the time difference between current and previous measurements
        df['TimeDiff'] = df['TimeElapsed'].diff()
        df.loc[0, 'TimeDiff'] = 0

        # rearrange columns
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2] # move TimeElapsed and TimeDiff to front
        df = df[cols]
        
        return df, num_points_dropped, device_id, serial_num, wavelength, basic_sample_rate, op_mode_period, \
            op_mode_FFT_num
    
    ### Generate histogram of time differences
    def rate_hist(self, log=True, bins=50, xmax=None):
        timedif = self.df['TimeDiff'][1:]   # Get time diffs, excluding first entry (which is 0)
        if (xmax != None) & np.any(timedif > xmax):
            print("Warning: Data beyond xmax={:f} in {:s}".format(xmax, self.title))
        fig, ax = plt.subplots(figsize=(12,3))
        # plot the histogram of time differences; only change x range if given
        ax.hist(timedif, bins=bins, align='mid', range=(0,xmax) if xmax else None, alpha=0.75)
        
        # Changing limits, adding titles
        if xmax != None:
            ax.set_xlim(0,xmax)
        else:
            ax.set_xlim(left=0)
        ax.set_xlabel('Time Between Samples [s]')
        if log:
            ax.set_yscale('log')
            #ax.set_ylabel('Frequency (log scale)')
            log_title_string = '(log scale) '
        else:
            #ax.set_ylabel('Frequency')
            log_title_string = ''
        ax.set_title('Time Between Samples (TBS) Histogram {:s}| {:s}'.format(log_title_string,self.title), fontsize=12, fontweight='bold')
        ax.grid(True)
        
        # Label statistics over plot
        ax.axvline(x=1/self.nominal_sample_rate, color='red', linestyle='--')   # Indicate nominal time dif
        timedif_avg = timedif.mean()   # Actual time dif (avg of all time diffs)
        timedif_std = timedif.std()   # STD of time diffs
        ax.axvline(x=timedif_avg, color='blue', linestyle='--')   # Indicate actual time dif
        ax.hlines(1, timedif_avg, timedif_avg+timedif_std, color='purple', lw=2)   # Indicate STD
        
        # Print statistics
        lines = [("Averaging Periods = {:.1f} ".format(self.op_mode_period), 'black'),
                ("FFT Points = {:d}".format(self.op_mode_FFT_num), 'black'),
                ("Basic Sample Rate = {:.0f} samples/s".format(self.basic_sample_rate), 'black'),
                ("Configured Sample Rate = {:.1f} samples/s".format(self.nominal_sample_rate), 'red'),
                ("Average Sample Rate = {:.1f} samples/s".format(self.avg_sample_rate), 'blue'),
                ("", 'black'),
                ("Configured TBS = {:.4f} s".format(1/self.nominal_sample_rate), 'red'),
                ("Average TBS = {:.4f} s".format(timedif_avg), 'blue'),
                ("TBS STD = ±{:.4f} s".format(timedif_std), 'purple')]
        x_pos = 0.95
        y_pos = 0.95
        line_height = 0.08
        # WARNING: this was giving errors before, I think bc the "range=(0,xmax) if..." line earlier redefined range
        for i in range(len(lines)):
            text, color = lines[i]
            ax.text(x_pos, y_pos - i*line_height, text, color=color, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        fig.tight_layout()
        display(fig); plt.close(fig)
        return fig
    
    # Used by plot functions to display the global time being plotted
    # E.g. a 2 hour dataset may be "11:36 - 13:45" (we use a 24 hour clock)
    # Input the dataframe we're plotting from, and the start/end time w/ respect to the dataframe (e.g. 2000 seconds)
    @staticmethod
    def gen_time_str(df, start_time=None, end_time=None):
        start_ind = 0 if start_time is None else df.index[df['TimeElapsed'] >= start_time].min()
        end_ind = len(df)-1 if end_time is None else df.index[df['TimeElapsed'] <= end_time].max()
        time_str_start = df.loc[start_ind,'Timestamp'].strftime('%H:%M')
        time_str_end = df.loc[end_ind,'Timestamp'].strftime('%H:%M')
        return '{:s} - {:s}'.format(time_str_start,time_str_end)

    ### Some measurements are stored in radians, some in degrees.
    # Sometimes we have to change the units based on what we want to plot
    # Also used to transfer self.angle_threshold from deg2rad, if necessary (input will be int or float)
    @staticmethod
    def get_correct_units(input):
        if isinstance(input, (int, float)):
            # Correcting self.angle_threshold
            if Dataset.ANGLE_UNITS_IS_RAD:
                return np.deg2rad(input)
            else:
                return input
        # input is pandas series (df column)
        if Dataset.ANGLE_UNITS_IS_RAD and input.name in Dataset.DEGREES_COLUMNS:
            return np.deg2rad(input)
        elif not Dataset.ANGLE_UNITS_IS_RAD and input.name in Dataset.RADIANS_COLUMNS:
            return np.rad2deg(input)
        else:
            return input
    
    ### Plots specified parameter over time (from the raw data)
    # birds_eye: plot the entirety of the avaliable data?
    # plot_param: choose from s1,s2,s3,S0,S1,S2,S3,Azimuth,Ellipticity,Power,DOP,...
    # sample_range: used to zoom in on a particular time range, e.g. (2000,2050) seconds
    # time_offset: used by SetPair (allows offsetting of plot by this constant; purely for plotting, no functional purpose)
    def plot_raw(self,birds_eye=True,plot_param='AllStokes',sample_range=None,time_offset=0.0):
        # sample_range should be of the form (sample_start, sample_end) if a smaller range is desired
        # if sample_start or sample_end are None themselves, they will be filled in
        if sample_range is not None:
            sample_start, sample_end = Dataset.fill_in_range(sample_range, self.df)
            sample_range = (sample_start+time_offset, sample_end+time_offset)   # Make sure we include the offset
        
        # Plot entire dataset if requested
        if plot_param=='AllStokes':
            params_array=[('s1','red'),
                    ('s2','blue'),
                    ('s3','purple')]
        else:
            params_array=[(plot_param,'red')]
        
        ### Plots specified parameter over time
        BE_fig=None
        if birds_eye:
            BE_fig, BE_ax = plt.subplots(figsize=(12,3))
            for param,color in params_array:
                BE_ax.plot(self.df['TimeElapsed'], Dataset.get_correct_units(self.df[param]), label=param, linewidth=0.5, marker='o', markersize=0.8, color=color, alpha=0.5)
            BE_ax.set_xlabel('Time [s]')
            BE_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df)), fontsize=14, fontweight='bold')
            BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,Dataset.UNITS.get(plot_param,'TODO')))
            BE_ax.grid(True)
            BE_ax.legend(loc='upper right')
            # We add green lines to the birds eye plot to denote where sample_range is located
            if sample_range is not None:
                for val in sample_range:
                    BE_ax.axvline(val, color = 'green', linewidth=2)
            display(BE_fig); plt.close(BE_fig)   # Show entire dataset
        
        # If requested, we'll also plot the smaller sample range
        ZI_fig=None
        if sample_range is not None:
            # Plot of specified sample
            ZI_fig, ZI_ax = plt.subplots(figsize=(12,3))
            for param,color in params_array:
                ZI_ax.plot(self.df['TimeElapsed'], Dataset.get_correct_units(self.df[param]), label=param, linewidth=1, marker='o', markersize=1.5, color=color)
            ZI_ax.set_xlabel('Time [s]')
            ZI_ax.set_title('{:s} (raw) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df, sample_range[0], sample_range[1])), fontsize=14, fontweight='bold')
            ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,Dataset.UNITS.get(plot_param,'TODO')))
            ZI_ax.grid(True)
            ZI_ax.legend(loc='upper right')
            ZI_ax.set_xlim(sample_range[0],sample_range[1])
            ZI_fig.tight_layout()
            display(ZI_fig); plt.close(ZI_fig)
        
        return BE_fig,ZI_fig
    
    # Below are a few similarity metrics we can use to quantify the "distance" between any observation and
    # a given reference polarization.
    # While the non-normalized stokes parameters are passed in (S1, S2, S3), we effectively normalize
    # when we divide the dot product by S0 in the end
    # Note: S0 is calculated at the time of taking the dot product, not used from the S0 parameter; this is
    # because the precision that comes with S0 is fine for graphing purposes, but not for ensuring the dot
    # product is normalized
    @staticmethod
    def dot(stokes_1, stokes_2):
        dot = np.dot(stokes_1, stokes_2)
        cos_value = dot / (np.linalg.norm(stokes_1) * np.linalg.norm(stokes_2))
        return cos_value
    @staticmethod
    def dist_metric(stokes_1, stokes_2):
        return 1-Dataset.dot(stokes_1, stokes_2)
    @staticmethod
    def angle_metric(stokes_1, stokes_2):
        cos_value = Dataset.dot(stokes_1, stokes_2)
        return np.degrees(np.arccos(cos_value))


