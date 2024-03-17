"""
**How IOPair works**

IOPair is an object to match Input and Output SwitchSets so we can correctly calculate the PTF. Specifically, we need to know which signal in each dataset is the "default", and match them together. Ensure there are 3-5 seconds of default signal behavior for each dataset; it is used to visually match which signal corresponds to which, and will be chopped off for actual datasets.

When using my_pair.match_IO_plot() properly:

* Ensure you know the "skip time" you are going to use for further analysis; this skip time (default 8 seconds) should skip over the first few seconds of non-switching data. Set via "input_skip_time=8"
* Same goes for "output_skip_time=8" for the output dataset
* Program identifies the stokes param that is most distinguishable (assign_by), and will create a small new dataset just to the first few seconds of that parameter.
* Another dataset (the one actually being used for further analysis) is created and signal "1" vs "2" are identified **deterministically based on the skip time it was given**.
* When it displays the "first measurement of signal 1/2, visually identify which one is the default signal (the only signal present in the first few seconds). Use "input_plot_range=(5,10)" and "output_plot_range=(5,10)" if these first few seconds aren't visible.
* If it says "Using assign_by = "s2Avg"", identify the S2 values of the first measurements.
    * Below, 0.118 for signal 1 and 2.047 for signal 2
* Visually identify which signal is the default
    * In the below example, default for input dataset is signal 2, and for output dataset is signal 1
* If the default signals are both 1 or both 2 for the input/output dataset, do nothing (or perform my_pair.set_signal_switch(False)) (the program guesses that the signals are already matched).
    * Like below, if the default signals are different between the input/output datasets, perform my_pair.set_signal_switch(True) to tell the program to switch the matchings for the actual datasets.


Now, the input/output default signals are properly matched. When you perform "ds = my_pair.get_full_ds()", the program will automatically give the correct matching to ds_output.calc_stokes_ptf so that the PTF will be properly calculated.
* Code is: "ds_output.calc_stokes_ptf(input_stokes=(ds_input.average_stokes()), switch_inputs=self.switch_signals)"
The dataset can now be used for further analysis, with the PTF properly calculated.
"""



class IOPair:
    def __init__(self, filepath_output, filepath_input):
        self.filepath_output = filepath_output
        self.filepath_input = filepath_input
        self.switch_signals = None
        return
    
    def match_IO_plot(self, output_set_range=(0,0.001), input_plot_range=(5,10), output_plot_range=(5,10), input_skip_time=8, output_skip_time=8, birds_eye=False):
        self.input_skip_time = input_skip_time
        print(BOLD_ON+'Using skip time of {:.1f} seconds for input dataset'.format(input_skip_time)+BOLD_OFF)
        ds_input = SwitchSet(self.filepath_input,skip_default_signal_baseline=input_skip_time)
        ds_input.find_switches(nominal_switch_rate=2, print_process=False)
        ds_input.average_data(print_assignment=True)
        # Default is signal ?
        
        input_digit_str = ''.join([char for char in ds_input.assign_by if char.isdigit()])
        ds_input_plot = Dataset(self.filepath_input,skip_default_signal_baseline=False)
        ds_input_plot.plot_raw(plot_param='S'+input_digit_str, birds_eye=birds_eye, sample_range=input_plot_range)
        
        
        self.output_skip_time = output_skip_time
        print(BOLD_ON+'Using skip time of {:.1f} seconds for output dataset'.format(output_skip_time)+BOLD_OFF)
        ds_output = SwitchSet(self.filepath_output,set_range=output_set_range,skip_default_signal_baseline=output_skip_time)
        ds_output.find_switches(nominal_switch_rate=2, print_process=False)
        ds_output.average_data(print_assignment=True)
        # Default is signal ?
        
        output_digit_str = ''.join([char for char in ds_output.assign_by if char.isdigit()])
        ds_output_plot = Dataset(self.filepath_output,set_range=output_set_range,skip_default_signal_baseline=False)
        ds_output_plot.plot_raw(plot_param='S'+output_digit_str, birds_eye=birds_eye, sample_range=output_plot_range)
        return
    
    def set_signal_switch(self, switch_signals):
        self.switch_signals = switch_signals
        return
    
    def get_full_ds(self, output_set_range=None, print_process_output=True):
        print(BOLD_ON+'\t======Begin averaging INPUT dataset...======\n'+BOLD_OFF)
        ds_input = SwitchSet(self.filepath_input,skip_default_signal_baseline=self.input_skip_time)
        ds_input.find_switches(nominal_switch_rate=2, print_process=False)
        ds_input.average_data(print_process=False)
        print(BOLD_ON+'\n\t======INPUT dataset averaged successfully======\n\n'+BOLD_OFF)
        
        print(BOLD_ON+'\t======Begin averaging OUTPUT dataset...======\n'+BOLD_OFF)
        ds_output = SwitchSet(self.filepath_output,set_range=output_set_range,skip_default_signal_baseline=self.output_skip_time)
        ds_output.find_switches(nominal_switch_rate=2, print_process=print_process_output)
        ds_output.average_data(print_process=print_process_output)
        print(BOLD_ON+'\n\t======OUTPUT dataset averaged successfully======\n\n\n'+BOLD_OFF)
        
        print(BOLD_ON+'\t======Begin PTF calculations for INPUT/OUTPUT dataset pair...======\n'+BOLD_OFF)
        ds_output.calc_stokes_ptf(input_stokes=(ds_input.average_stokes()), switch_inputs=self.switch_signals)
        print(BOLD_ON+'\n\t======PTF calculations for INPUT/OUTPUT dataset pair performed successfully!!======'+BOLD_OFF)
        
        return ds_output