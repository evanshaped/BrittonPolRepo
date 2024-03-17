from dataset import Dataset
from switch import SwitchSet

### Plots rotation angles for each of two datasets that have had their PTFs calculated
def plot_rot_angle(ds_1, ds_2, birds_eye=True, sample_range=None):
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
        BE_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df[plot_param]), label=plot_param+' ds 1', \
                       linestyle='-', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[0])
        if ds_1.angle_threshold_deg is not None: BE_ax.axhline(y=Dataset.get_correct_units(ds_1.angle_threshold_deg), color='red', linewidth=1)
        if ds_1.reset_times is not None and len(ds_1.reset_times) > 0:
            for time in ds_1.reset_times:
                BE_ax.axvline(time, color = 'red', linewidth=0.5)
        # Plot rotAngle from ds 2
        BE_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df[plot_param]), label=plot_param+' ds 2', \
                       linestyle='-', linewidth=0.6, marker='', markersize=0.5, color=custom_palette[1])
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
        ZI_ax.plot(ds_1.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_1.stokes_ptf_df[plot_param]), label=plot_param+' ds 1', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[0])
        if ds_1.angle_threshold_deg is not None: ZI_ax.axhline(y=Dataset.get_correct_units(ds_1.angle_threshold_deg), color='red', linewidth=1)
        if ds_1.reset_times is not None and len(ds_1.reset_times) > 0:
            for time in ds_1.reset_times:
                if sample_range[0] <= time <= sample_range[1]:
                    ZI_ax.axvline(time, color = 'red', linewidth=0.5)
        # Plot rotAngle from ds 2
        ZI_ax.plot(ds_2.stokes_ptf_df['EstTime'], Dataset.get_correct_units(ds_2.stokes_ptf_df[plot_param]), label=plot_param+' ds 2', \
                   linestyle='-', linewidth=0.7, marker='+', markersize=0.8, color=custom_palette[1])
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