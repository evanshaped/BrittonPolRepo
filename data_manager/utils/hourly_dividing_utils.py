import numpy as np
import pandas as pd
import data_manager.utils.dataframe_management_utils as dataframe_management_utils
BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"

class Divider:
    def __init__(self, dataset):
        self.data = dataset
        self.segment_dict

    def _compute_hourly_segments(self):
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
        initial_timestamp = self.data.df['Timestamp'].iloc[0]
        initial_day = 0  # Day starts from 0
        initial_hour = initial_timestamp.hour

        # Calculate the number of seconds from the beginning of the dataset to the start of the first full hour
        seconds_to_first_full_hour = (60 - initial_timestamp.minute) * 60 - initial_timestamp.second
        
        # Iterate over each full hour after the initial time
        start_time = initial_timestamp + pd.Timedelta(seconds=seconds_to_first_full_hour)
        end_time = start_time + pd.Timedelta(hours=1)
        idx = 0

        while end_time <= self.data.df['Timestamp'].iloc[-1] + pd.Timedelta(hours=1):
            current_day = (start_time - initial_timestamp).days
            current_hour = start_time.hour
            
            # Find indices in df that are within the current hour segment
            mask_df = (self.data.df['Timestamp'] >= start_time) & (self.data.df['Timestamp'] < end_time)
            relevant_df = self.data.df.loc[mask_df]
            
            if not relevant_df.empty:
                # TimeElapsed range
                timestamp_range = (relevant_df['Timestamp'].min(), relevant_df['Timestamp'].max())
                time_elapsed_range = (relevant_df['TimeElapsed'].min(), relevant_df['TimeElapsed'].max())
                df_idx_range = (relevant_df.index.min(), relevant_df.index.max())
                
                # Corresponding indices in stokes_ptf_df
                mask_stokes = (self.data.stokes_ptf_df['EstTime'] >= time_elapsed_range[0]) & \
                              (self.data.stokes_ptf_df['EstTime'] <= time_elapsed_range[1])
                relevant_stokes = self.data.stokes_ptf_df.loc[mask_stokes]
                
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

    # def print_segment_dict(self):
    #     """
    #     prints keys and values of dataset segments
    #     """
    #     for (key, value) in self.segment_dict.items():
    #         print(key)

    def _get_segment_adev(self, key=None):
        """
        Processes a specific segment of `stokes_ptf_df` based on the given key which corresponds
        to a tuple of (day, hour). This method retrieves the indexed range from `segment_dict`
        and performs data analysis on the `rotAngleDif` values within this segment.
    
        The function drops any NaN values from `rotAngleDif`, calculates the cumulative sum,
        and then performs Allan deviation analysis on this data. The results, along with other
        metadata like a generated title and time string, are returned as a tuple.
    
        Parameters:
            key (tuple): A tuple of (day, hour) that identifies the specific hour segment for which
                         the data should be processed. If equal to None, will use entire dataset
    
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
        if key is not None:
            # Retrieve the index range from the segment_dict for the given key
            value_dict = self.segment_dict[key]
            stokes_idx_range = value_dict['StokesIdxRange']
            
            # Slice the stokes_ptf_df dataframe using the retrieved index range
            stokes_ptf_df_slice = self.data.stokes_ptf_df.iloc[stokes_idx_range[0]:stokes_idx_range[1]+1]
        else:
            stokes_ptf_df_slice = self.data.stokes_ptf_df

        # Perform the specified operations
        dif_data = stokes_ptf_df_slice['rotAngleDif'].dropna()
        walk_data = np.cumsum(dif_data)
        meas_rate = 1 / (2 * self.data.switch_time)
        elm = allantools.oadev(walk_data.values, rate=meas_rate, taus='all', data_type="freq")
        if key is not None:
            timestamp_range = value_dict['TimestampRange']
            time_str_start = timestamp_range[0].strftime('%H:%M')
            time_str_end = timestamp_range[1].strftime('%H:%M')
            time = '{:s} - {:s}'.format(time_str_start,time_str_end)
        else:
            time = dataframe_management_utils.gen_time_str(self.data.df)
        set_title = self.data.title
        
        # Combine all parameters into a tuple
        params = (*elm, set_title, time)
        
        return params

    def calc_adev_divided(self):
        """
        Divides the dataset into hourly segments using "compute_hourly_segments()", and
        calculates adev for each segment. params_arr can be plotted using plotting.plot_adev
        """
        self._compute_hourly_segments()
        adev_params_arr = []
        for key in self.segment_dict.keys():
            adev_params_arr.append(self._get_segment_adev(key))
        self.adev_params_arr = adev_params_arr
        return adev_params_arr