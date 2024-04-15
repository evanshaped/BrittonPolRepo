from .dataset import Dataset
from .switch import SwitchSet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import pandas as pd
from scipy.optimize import curve_fit
BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"

### Plots rotation angles for each of two datasets that have had their PTFs calculated
def plot_rot_angle(ds_1, ds_2, birds_eye=True, sample_range=None, plot_raw=True):
    if ds_1.signal_1_df is None or ds_2.signal_1_df is None:
        print('Error: averages not yet calculated')
        return
    if ds_1.stokes_ptf_df is None or ds_2.stokes_ptf_df is None:
        print('Error: PTF not yet calculated')
        return

    plot_param = 'rotAngle'
    
    # Plot entire dataset if specified
    if birds_eye:
        # Birds Eye plot
        BE_fig, BE_ax = plt.subplots(figsize=(12,3))
        custom_palette = sns.color_palette("dark")
        
        # Plot rotAngle from ds 1
        if plot_raw: BE_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df[plot_param]), label=plot_param+' ds 1', \
                       alpha=0.5, linestyle='-', linewidth=0.4, marker='', markersize=0.5, color=custom_palette[0])
        BE_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling'+' ds 1', \
                       linestyle='-', linewidth=0.8, marker='', markersize=0.5, color=custom_palette[2])
        if ds_1.angle_threshold_deg is not None: BE_ax.axhline(y=Dataset.get_correct_units(ds_1.angle_threshold_deg), color='red', linewidth=1)
        if ds_1.reset_times is not None and len(ds_1.reset_times) > 0:
            for time in ds_1.reset_times:
                BE_ax.axvline(time, color = 'red', linewidth=0.5)
        # Plot rotAngle from ds 2
        if plot_raw: BE_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df[plot_param]), label=plot_param+' ds 2', \
                       alpha=0.5, linestyle='-', linewidth=0.4, marker='', markersize=0.5, color=custom_palette[1])
        BE_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling'+' ds 2', \
                       linestyle='-', linewidth=0.8, marker='', markersize=0.5, color=custom_palette[3])
        if ds_2.angle_threshold_deg is not None: BE_ax.axhline(y=Dataset.get_correct_units(ds_2.angle_threshold_deg), color='orange', linewidth=1)
        if ds_2.reset_times is not None and len(ds_2.reset_times) > 0:
            for time in ds_2.reset_times:
                BE_ax.axvline(time, color = 'orange', linewidth=0.5)
        # Labels and such for plot
        BE_ax.set_xlabel('Time [s]')
        BE_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
        BE_ax.grid(True)
        BE_ax.set_title('{:s} | ds 1: {:s} | ds 2: {:s}'.format(plot_param, ds_1.title, ds_2.title, fontsize=14, fontweight='bold'))
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
        # Plot rotAngle from ds 1
        if plot_raw: ZI_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df[plot_param]), label=plot_param+' ds 1', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[0])
        ZI_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling'+' ds 1', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[2])
        if ds_1.angle_threshold_deg is not None: ZI_ax.axhline(y=Dataset.get_correct_units(ds_1.angle_threshold_deg), color='red', linewidth=1)
        if ds_1.reset_times is not None and len(ds_1.reset_times) > 0:
            for time in ds_1.reset_times:
                if sample_range[0] <= time <= sample_range[1]:
                    ZI_ax.axvline(time, color = 'red', linewidth=0.5)
        # Plot rotAngle from ds 2
        if plot_raw: ZI_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df[plot_param]), label=plot_param+' ds 2', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[1])
        ZI_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df['rotAngleRolling']), label='rotAngleRolling'+' ds 2', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[3])
        if ds_2.angle_threshold_deg is not None: ZI_ax.axhline(y=Dataset.get_correct_units(ds_2.angle_threshold_deg), color='orange', linewidth=1)
        if ds_2.reset_times is not None and len(ds_2.reset_times) > 0:
            for time in ds_2.reset_times:
                if sample_range[0] <= time <= sample_range[1]:
                    ZI_ax.axvline(time, color = 'orange', linewidth=0.5)
        # Labels and such for plot
        ZI_ax.set_xlabel('Time [s]')
        ZI_ax.set_ylabel('{:s} [{:s}]'.format(plot_param,SwitchSet.UNITS.get(plot_param,'TODO')))
        ZI_ax.grid(True)
        ZI_ax.set_title('{:s} | ds 1: {:s} | ds 2: {:s}'.format(plot_param, ds_1.title, ds_2.title, fontsize=14, fontweight='bold'))
        ZI_ax.legend(loc='upper left')
        # zoom in on sample range
        ZI_ax.set_xlim(sample_range[0],sample_range[1])
        display(ZI_fig); plt.close(ZI_fig)
    
    #return BE_fig, ZI_fig
    return

### For a switch set with calculated point-to-reference rotAngleDif values, fit a gaussian distribution and plot.
### Returns popt, perr (optimal parameters and uncertainties)
def ptr_walk_hist(ds_switch, bins=150):
    # Gaussian function
    def gauss(x, mu, sigma, A):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    # Histogram data
    hist_data = ds_switch.stokes_ptf_df['rotAngleDif'].dropna()
    bin_list = np.linspace(min(hist_data), max(hist_data), bins)
    hist, bin_edges = np.histogram(hist_data, bins=bin_list)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit the Gaussian function to the histogram data
    popt, pcov = curve_fit(gauss, bin_centers, hist, p0=[np.mean(hist_data), np.std(hist_data), 1])
    # Extracting the uncertainties (standard errors) of the fitted parameters
    perr = np.sqrt(np.diag(pcov))

    time_str = Dataset.gen_time_str(ds_switch.df)
    print(BOLD_ON + '----- Gauss fit params | {:s} | {:s} -----'.format(ds_switch.title, time_str) + BOLD_OFF)
    parameter_names = ['mu', 'sigma', 'A']
    for param, uncertainty, name in zip(popt, perr, parameter_names):
        print(f"\t{name} = {param:.5f} ± {uncertainty:.5f}")
    print(BOLD_ON + '-----------------------------------------------------------' + BOLD_OFF)
    
    # Plot the histogram and the fit
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(hist_data, bins=bin_list, alpha=0.6, color='blue')
    ax.plot(bin_centers, gauss(bin_centers, *popt), 'orange', linewidth=2)
    
    # Adding a vertical red line at the mean
    ax.axvline(x=popt[0], color='red', lw=2, label=f'Mean: {popt[0]:.5f}')
    # Adding a horizontal purple line for the standard deviation
    ax.hlines(popt[2]*0.05, popt[0], popt[0] + popt[1], color='purple', lw=2, label=f'STD: {popt[1]:.5f}')
    
    ax.set_title('rotAngleDif hist with Gauss Fit | {:s} | {:s}'.format(ds_switch.title, time_str), fontweight='bold')
    ax.set_xlabel('rotAngleDif')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)
    
    display(fig); plt.close(fig)

    return (ds_switch.title, time_str, popt, perr)

### For a switch set with calculated point-to-reference rotAngleDif values, fit a gaussian distribution and plot.
### Returns popt, perr (optimal parameters and uncertainties)
# not done!!!!!
def ptr_walk_hist_double(ds_switch, fig, ax, color, time_range=None, bins=150):
    # Gaussian function
    def gauss(x, mu, sigma, A):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    # Histogram data
    stokes_df = ds_switch.stokes_ptf_df
    if time_range is not None: stokes_df = stokes_df[(stokes_df['EstTime'] > time_range[0]) & (stokes_df['EstTime'] < time_range[1])]
    hist_data = stokes_df['rotAngleDif'].dropna()
    bin_list = np.linspace(min(hist_data), max(hist_data), bins)
    hist, bin_edges = np.histogram(hist_data, bins=bin_list)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit the Gaussian function to the histogram data
    popt, pcov = curve_fit(gauss, bin_centers, hist, p0=[np.mean(hist_data), np.std(hist_data), 1])
    # Extracting the uncertainties (standard errors) of the fitted parameters
    perr = np.sqrt(np.diag(pcov))

    time_str = Dataset.gen_time_str(ds_switch.df)
    print(BOLD_ON + '----- Gauss fit params | {:s} | {:s} -----'.format(ds_switch.title, time_str) + BOLD_OFF)
    parameter_names = ['mu', 'sigma', 'A']
    for param, uncertainty, name in zip(popt, perr, parameter_names):
        print(f"\t{name} = {param:.5f} ± {uncertainty:.5f}")
    print(BOLD_ON + '-----------------------------------------------------------' + BOLD_OFF)
    
    # Plot the histogram and the fit
    
    ax.hist(hist_data, bins=bin_list, alpha=0.6, color='blue')
    ax.plot(bin_centers, gauss(bin_centers, *popt), 'orange', linewidth=2)
    
    # Adding a vertical red line at the mean
    ax.axvline(x=popt[0], color='red', lw=2, label=f'Mean: {popt[0]:.5f}')
    # Adding a horizontal purple line for the standard deviation
    ax.hlines(popt[2]*0.05, popt[0], popt[0] + popt[1], color='purple', lw=2, label=f'STD: {popt[1]:.5f}')

    return (ds_switch.title, time_str, popt, perr)

### Plots adev plots on same plot               ...plot
### params_arr is array of tuples, one tuple for each allan deviation plot
### Each tuple is: (averaging_times, allan_dev, allan_dev_error, number_of_samples, plot_label)
def plot_adev(params_arr):
    ADev_fig,ADev_ax = plt.subplots(figsize=(12,4))
    for taus2,ad,ade,_,set_title,time in params_arr:
        ADev_ax.errorbar(taus2, ad, yerr=ade, label=time)
    title="ADev signal stability | {:s}".format(set_title)
    ADev_ax.set_xscale("log")
    ADev_ax.set_yscale("log")
    ADev_ax.set_xlabel('Tau [s]')
    ADev_ax.set_ylabel('Allan Deviation')
    ADev_ax.set_title(title)
    ADev_ax.grid(True)
    #ADev_ax.legend(loc='upper left')
    #ADev_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ADev_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ADev_fig.tight_layout()
    display(ADev_fig); plt.close(ADev_fig)
    #plt.show()

# Plots adev all on one plot, in different colors
def plot_adev_color(params_arr):
    def create_red_to_blue_palette(n=20):
        # Red is (1, 0, 0) and Blue is (0, 0, 1)
        red = np.array([1, 0, 0])
        blue = np.array([0, 0, 1])
        
        # Generate a list of colors between red and blue
        palette = [tuple((1 - i/(n-1)) * red + i/(n-1) * blue) for i in range(n)]
        return palette
    
    # Generate the palette
    color_palette = create_red_to_blue_palette(len(params_arr))

    ADev_fig,ADev_ax = plt.subplots(figsize=(12,4))
    for i, (taus2,ad,ade,_,set_title,time) in enumerate(params_arr):
        ADev_ax.errorbar(taus2, ad, yerr=ade, label=time, color=color_palette[i])
    title="ADev signal stability | {:s}".format(set_title)
    ADev_ax.set_xscale("log")
    ADev_ax.set_yscale("log")
    ADev_ax.set_xlabel('Tau [s]')
    ADev_ax.set_ylabel('Allan Deviation')
    ADev_ax.set_title(title)
    ADev_ax.grid(True)
    #ADev_ax.legend(loc='upper left')
    #ADev_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ADev_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ADev_fig.tight_layout()
    #display(ADev_fig); plt.close(ADev_fig)
    plt.show()