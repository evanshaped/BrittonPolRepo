BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
import numpy as np
import pandas as pd

# Below are a few similarity metrics we can use to quantify the "distance" between any observation and
# a given reference polarization.
# While the non-normalized stokes parameters are passed in (S1, S2, S3), we effectively normalize
# when we divide the dot product by S0 in the end
# Note: S0 is calculated at the time of taking the dot product, not used from the S0 parameter; this is
# because the precision that comes with S0 is fine for graphing purposes, but not for ensuring the dot
# product is normalized
def dot(stokes_1, stokes_2):
    dot = np.dot(stokes_1, stokes_2)
    cos_value = dot / (np.linalg.norm(stokes_1) * np.linalg.norm(stokes_2))
    return cos_value

def dist_metric(stokes_1, stokes_2):
    return 1-dot(stokes_1, stokes_2)

def angle_metric(stokes_1, stokes_2):
    cos_value = dot(stokes_1, stokes_2)
    return np.degrees(np.arccos(cos_value))


### Just calculates angle (in radians) between the two vectors (to make code more readable)
### Assumes unnormalized vectors
def angle_between_SOPs(stokes_vector_1, stokes_vector_2):
    vector_dot = np.dot(stokes_vector_1, stokes_vector_2)
    n1 = np.linalg.norm(stokes_vector_1)
    n2 = np.linalg.norm(stokes_vector_2)
    dot_normalized = vector_dot / (n1 * n2)
    angle = np.rad2deg(np.arccos(np.clip(dot_normalized, -1.0, 1.0))) # Clip to [-1,1] for numerical stability
    return angle


STOKES_PTF_DF_COLUMNS = ['EstTime', 'basesMatrix', 'sig1Stokes', 'sig2Stokes', 'axis', 'rotAngle', 'angleDif', 'rotAngleRolling', 'WasReset']

### Used by calc_stokes_ptf()
### Func to reverse-engineer the STOKES matrix from the input polarization and time-dependent output polarizations
### signal_1_current_row is one observation in the averaged signal 1 (same for signal_2_current_row)
### These rows should already be aligned in time
### Assume unnormalized stokes vectors
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
    current_matrix = np.column_stack(construct_orthonormal_bases(signal_1_current_stokes, \
                                                                          signal_2_current_stokes))
    
    ### Here is the meat of the function; we calculate the rotation matrix and deduce the axis/angle representation
    rotation_matrix = np.dot(current_matrix, reference_matrix.T)
    axis = np.array([rotation_matrix[2,1] - rotation_matrix[1,2],
                     rotation_matrix[0,2] - rotation_matrix[2,0],
                     rotation_matrix[1,0] - rotation_matrix[0,1]])
    angle = np.rad2deg(np.arccos((np.trace(rotation_matrix)-1)/2))

    # Get angle between SOPs of the two output signals; should be similar to angle between SOPs of input signals
    # TODO: calculate angle between SOPs of input signals
    angle_between_current_stokes = angle_between_SOPs(signal_1_current_stokes, signal_2_current_stokes)

    return pd.Series([time, current_matrix, signal_1_current_stokes, signal_2_current_stokes, \
                      axis, angle, angle_between_current_stokes, 0.0, False],
                     index=STOKES_PTF_DF_COLUMNS)


### Used by calc_stokes_ptf in SwitchSet
### Constructs 3 orthonormal bases given two linearly independent vectors
### See notebook for details
### Assumes unnormalized input vectors
# TODO: see below
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


# ### Used by SwitchSet.calc_stokes_ptf(...)
# ### Returns the average polarization state of both signals
# ### Meant for short datasets, like those used for the input measurements
# def average_stokes(signal_1_df, signal_2_df, change_point_params):
#     if signal_1_df is None:
#         print('Error: averages not yet calculated')
#         return
    
#     # Base z scores off of change_point_param
#     switch_param = change_point_params[0][3]
#     z_score_param = 'S' + switch_param[1] + 'Avg'   # E.g. "S2Avg"
    
#     # Get z scores of the averages for the switch parameter used
#     signal_1_z_scores = np.abs((signal_1_df[z_score_param] - signal_1_df[z_score_param].mean()) / signal_1_df[z_score_param].std())
#     signal_2_z_scores = np.abs((signal_2_df[z_score_param] - signal_2_df[z_score_param].mean()) / signal_2_df[z_score_param].std())
#     # Set a Z-score threshold for outliers (e.g., 2)
#     z_score_threshold = 2
#     # Filter out values above the Z-score threshold
#     signal_1_df_filtered = signal_1_df[signal_1_z_scores <= z_score_threshold]
#     signal_2_df_filtered = signal_2_df[signal_2_z_scores <= z_score_threshold]
    
#     # For the non-outlier rows, get the averages of S0, S1, etc
#     signal_1_stokes = [signal_1_df_filtered[param].mean() for param in ['S1Avg','S2Avg','S3Avg']]
#     signal_2_stokes = [signal_2_df_filtered[param].mean() for param in ['S1Avg','S2Avg','S3Avg']]
    
#     return signal_1_stokes, signal_2_stokes

### Used by SwitchSet.calc_stokes_ptf(...)
### Returns the first observed polarization state of both signals
def first_stokes(signal_1_df, signal_2_df):
    signal_1_stokes = [signal_1_df[param][0] for param in ['S1Avg','S2Avg','S3Avg']]
    signal_2_stokes = [signal_2_df[param][0] for param in ['S1Avg','S2Avg','S3Avg']]
    return signal_1_stokes, signal_2_stokes