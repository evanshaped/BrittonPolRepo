BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings

# skip_default_signal_baseline = True iff the first 10 seconds (no more than 20) are used only to identify
# which signal is the default one. These seconds will be automatically cut out. A value of t seconds can also
# be assigned, to give a custom number of seconds to skip (e.g. 100 if the first 100 seconds contains just one signal)
def read_pax_data(filename, set_range=None, skip_default_signal_baseline=True):
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
        set_start, set_end = dataframe_management_utils.fill_in_range(time_range, df)   # automatically fills in missing range vals
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


# Used by plot functions to display the global time being plotted
# E.g. a 2 hour dataset may be "11:36 - 13:45" (we use a 24 hour clock)
# Input the dataframe we're plotting from, and the start/end time w/ respect to the dataframe (e.g. 2000 seconds)
def gen_time_str(df, start_time=None, end_time=None):
    start_ind = 0 if start_time is None else df.index[df['TimeElapsed'] >= start_time].min()
    end_ind = len(df)-1 if end_time is None else df.index[df['TimeElapsed'] <= end_time].max()
    time_str_start = df.loc[start_ind,'Timestamp'].strftime('%H:%M')
    time_str_end = df.loc[end_ind,'Timestamp'].strftime('%H:%M')
    return '{:s} - {:s}'.format(time_str_start,time_str_end)


stokes_units = 'W/m^2'
unitless = 'unitless'
radians = 'Rad'; degrees = 'Degrees'
ANGLE_UNITS_IS_RAD = False
angle_units = radians if ANGLE_UNITS_IS_RAD else degrees
DEGREES_COLUMNS = ["Azimuth", "AzimuthAvg", "AzimuthStd", "Ellipticity", "EllipticityAvg", "EllipticityStd", "angleSimMetric", "rotAngle", "angleDif", "rotAngleRolling"]
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
                  'rotAngle':angle_units, 'angleDif':angle_units, 'rotAngleRolling':angle_units}

### Some measurements are stored in radians, some in degrees.
# Sometimes we have to change the units based on what we want to plot
# Also used to transfer self.angle_threshold from deg2rad, if necessary (input will be int or float)
def transfer_value_units(my_input):
    if isinstance(my_input, (int, float)):
        # Correcting self.angle_threshold
        if ANGLE_UNITS_IS_RAD:
            return np.deg2rad(my_input)
        else:
            return my_input
    # input is pandas series (df column)
    if ANGLE_UNITS_IS_RAD and my_input.name in DEGREES_COLUMNS:
        return np.deg2rad(my_input)
    elif not ANGLE_UNITS_IS_RAD and my_input.name in RADIANS_COLUMNS:
        return np.rad2deg(my_input)
    else:
        return my_input


def get_param_units(my_param, default_value):
    return UNITS.get(my_param,default_value)