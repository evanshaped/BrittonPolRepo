BOLD_ON = "\033[1m"
BOLD_OFF = "\033[0m"
from .dataset import Dataset
import data_manager.utils.plotting_utils as plotting_utils

class StationarySet(Dataset):
    def average_stokes(self):
        input_stokes = [self.df[param].mean() for param in ['S0','S1','S2','S3']]
        return input_stokes
    
    def first_stokes(self):
        input_stokes = [self.df[param][0] for param in ['S0','S1','S2','S3']]
        return input_stokes
    
    def adev(self, plot_param='s1', num_taus=5000, plot_adev=False, plot_psd=False):
        signal = self.df[plot_param].values
        meas_rate = self.avg_sample_rate
        
        if plot_psd:
            # Plot PSD
            PSD_fig,PSD_ax = plt.subplots(figsize=(12,3))
            PSD_ax.psd(signal, NFFT=1024, Fs=meas_rate)
            PSD_ax.set_title('Power Spectral Density (PSD) of Signal')
            PSD_ax.set_xlabel('Frequency [Hz]')
            PSD_ax.set_ylabel('Power')
            PSD_ax.grid(True)
            PSD_fig.tight_layout()
            display(PSD_fig); plt.close(PSD_fig)

        # Calc taus
        #num_taus = 5000
        base = 10
        #power = int(np.log(signal.size/2)/np.log(base))
        #pnts = np.logspace(0, power, num=num_taus, base=base, dtype=int) # exact integer number of points
        #taus = np.unique(pnts/meas_rate)
        min_value = 1/meas_rate
        max_value = min_value * (signal.size-1)
        taus = np.logspace(np.log(min_value) / np.log(base), np.log(max_value) / np.log(base), num=num_taus, base=base)
        #taus = 'all'

        taus2, ad, ade, ns = allantools.oadev(signal, rate = meas_rate, taus=taus, data_type="freq")
        #print('\t\trate={:.2f} | min tau = {:.5f} | max tau = {:.5f}'.format(meas_rate, taus[0], taus[-1]))
        print('Number of taus used: {:d}'.format(taus2.size))
        
        label = '{:s} (ADev) | {:s} | {:s}'.format(plot_param, self.title, Dataset.gen_time_str(self.df))

        if plot_adev:
            plotting_utils.plot_adev([(taus2,ad,ade,ns,label)], plot_param=plot_param)
#             ADev_fig,ADev_ax = plt.subplots(figsize=(12,4))
#             ADev_ax.errorbar(taus2, ad, yerr=ade)
#             ADev_ax.set_xscale("log")
#             ADev_ax.set_yscale("log")
#             ADev_ax.set_xlabel('Tau [s]')
#             ADev_ax.set_ylabel('Allan Deviation')
#             ADev_ax.set_title('Signal Stability')
#             ADev_ax.grid(True)
#             ADev_fig.tight_layout()
#             display(ADev_fig); plt.close(ADev_fig)
        
        return [(taus2, ad, ade, ns, label)]
    
    