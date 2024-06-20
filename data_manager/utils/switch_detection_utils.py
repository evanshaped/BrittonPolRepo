BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_manager.utils.dataframe_management_utils as dataframe_management_utils

### The waveform generator does not output *exactly* a 2 Hz square wave; it has a resolution limit, and instead generates a 1.998 Hz or 2.002 Hz wave.
### For long datasets (multiple hours) we cannot use the nominal switch time of 1 / (2 * 2Hz) = 0.25 seconds (finding the switch offset and extending the
### switch markers, the markers will drift from the actual switch times). We must locate the switch offset at the beginning *and* at ~1 hour in, and calculate
### the actual switch time.
def calc_actual_switch_time(mean_offset_1, change_point_range_1, mean_offset_2, change_point_range_2, nominal_switch_time, print_process=False):
    # Difference in the two offsets; this is the total drift between the two ranges
    offset_drift = mean_offset_2 - mean_offset_1
    # Occasionally, stuff will happen with the mod of the offset drift, and the phase range will have to be adjusted
    if abs(offset_drift) > (nominal_switch_time/2):
        if offset_drift > 0:
            offset_drift -= nominal_switch_time
        else:
            offset_drift += nominal_switch_time
    # Now, offset_drift should be a little above 0 or a little below 0
    # A positive offset_drift means the actual switch time is a little more than the nominal switch time
    # Vice versa for a negative offset drift
    # How many switches it took for the offset to drift by offset_drift:
    cp_range_1_midpoint = (change_point_range_1[1]+change_point_range_1[0])/2
    cp_range_2_midpoint = (change_point_range_2[1]+change_point_range_2[0])/2
    num_switches_between_cp_ranges = (cp_range_2_midpoint - cp_range_1_midpoint) / nominal_switch_time
    # Correction to the nominal_switch_time to account for limited resolution in the waveform generator
    # How much drift is contained in each switch between the two ranges
    switch_time_correction = offset_drift / num_switches_between_cp_ranges
    # Add the delta; this is the actual time between switches
    actual_switch_time = nominal_switch_time + switch_time_correction

    # If the offset is referenced from 0, we need to adjust it now that the switch time has changed
    # The offset needs to be the remainder of the detection location divided by the switch time
    # mean_offset_1_location is the detection location
    mean_offset_1_location = mean_offset_1 + nominal_switch_time * np.floor(cp_range_1_midpoint/nominal_switch_time)
    # We calculate the new offset based on the newly corrected switch time
    actual_switch_offset = mean_offset_1_location % actual_switch_time

    if print_process: print(BOLD_ON+'Nominal Switch Time = {:.7f}\nOffset change of {:.3f} seconds over {} switches\nCorrected Switch Time = {:.7f}'.format(nominal_switch_time,offset_drift,num_switches_between_cp_ranges,actual_switch_time)+BOLD_OFF)

    return actual_switch_time, actual_switch_offset


### Used by change_point in this module
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


### Used by change_point in this module
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



def change_point(main_df, nominal_switch_time, change_point_range, n_exclude, print_process=False):
    if print_process: print(BOLD_ON+'=== performing change point detection ==='+BOLD_OFF)
    # change_point_range should be of the form (change_point_start, change_point_end) if a smaller range is desired
    # if either start or end are None themselves, they will be filled in automatically
    change_point_start, change_point_end = dataframe_management_utils.fill_in_range(change_point_range, main_df)
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
            left_edge_iter, right_edge_iter = compute_range(hist_df_iter, first_tolerance, second_tolerance, patches_iter, print_process)
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
    mean_offset = compute_offset(midtime_array, nominal_switch_time, print_process)
    if print_process: print('offset = {:.3f} seconds'.format(mean_offset))
    if print_process: print(BOLD_ON+'=== end change point detection ==='+BOLD_OFF)
    return change_point_range, change_point_df, points_skipped, switch_param, mean_offset, left_edge, right_edge, jump_fig


### Used by SwitchSet.average_data
### There are often large gaps left in the dataset when the PAX lags for seconds at a time.
### If one of these gaps spans an entire segment where a polarization signal was supposed
### to be recorded (i.e., if there are no recorded points in that segment),
### we interpolate the data using the nearest segments for that signal where there *is*
### valid data.
def interpolate(df, time_gap, print_process=False):
    ### This function works by finding the missing sections and imputing (linearly interpolating) the values.
    ### For each gap we create and store the necessary rows; at the end, we add them all
    ### to the final dataframe and sort by time to get everything back in order
    if print_process: print('--- interpolating data ---')
    #print_interpolate = print_process
    print_interpolate = False   # Make this True if you wish to debug interpolation
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
    # Concatenate the original DataFrame and the interpolated DataFrame, re-organizing by EstTime
    result_df = pd.concat([df, interpolated_df]).sort_values(by='EstTime').reset_index(drop=True)

    if print_process: print('length before: {}'.format(len(df)))
    if print_process: print('length after: {}'.format(len(result_df)))
    if print_process: print('{} total added'.format(len(result_df)-len(df)))

    result_df.drop(columns=['TimeDiff'], inplace=True)

    if print_process: print('--- end interpolating data ---')
    return result_df