BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
import numpy as np
import pandas as pd

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