import numpy as np
import neo
import scipy.io
from scipy import signal
import matplotlib.pylab as plt
import matplotlib
from quantities import Hz, uV, ms, s
from numpy import meshgrid
import scipy.ndimage as ndi
#from ephys_analysis.lfp_processing.filters import butter_lowpass_filter, iir_notch
#from ephys_analysis.mcd_conv import get_electrode_data

def plot_spectrogram(data, fs, levels=100, sigma=1, perc_low=1, perc_high=99, nfft=1024, noverlap=512):
    """
    Data
    ------------
    data: Quantity array or Numpy ndarray
        Your time series of voltage values
    fs: Quantity
        Sampling rate in Hz


    Spectrogram parameters
    ------------
    levels: int
        The number of color levels displayed in the contour plot (spectrogram)
    sigma: int
        The standard deviation argument for the gaussian blur
    perc_low, perc_high: int
        Out of the powers displayed in your spectrogram, these are the low and high percentiles
        which mark the low and high ends of your colorbar.

        E.g., there might be a period in the start of the experiment where the voltage time series shifts
        abruptly, which would appear as a vertical 'bar' of high power in the spectrogram; setting perc_high
        to a value lower than 100 will make the color bar ignore these higher values (>perc_high), and
        display the 'hottest' colors as the highest power values other than these (<perc_high), allowing
        for better visualization of actual data. Similar effects can be accomplished with vmin/vmax args
        to countourf.
    nfft: int
        The number of data points used in each window of the FFT.
        Argument is directly passed on to matplotlib.specgram(). See the documentation
        `matplotlib.specgram()` for options.
    noverlap: int
        The number of data points that overlap between FFT windows.
        Argument is directly passed on to matplotlib.specgram(). See the documentation
        `matplotlib.specgram()` for options.
    """

    plt.rcParams['image.cmap'] = 'jet'

    spec, freqs, bins, __ = plt.specgram(data, NFFT=nfft*int((fs/1000)), Fs=int(fs), noverlap=noverlap, scale='dB') #gives us time and frequency bins with their power
    plt.close()
    plt.figure(figsize=(18.0, 10.0))
    max_idx = (np.abs(freqs-200)).argmin()
    freqs = freqs[0:max_idx]
   # spec = spec / np.mean(spec, 0)
    Z = np.flipud(np.log10(spec[:max_idx,:])*10) #make sure output isn't already in decibels
    Z = ndi.gaussian_filter(Z, sigma)
    extent = 0, np.amax(bins), freqs[0], freqs[-1]
    levels = np.linspace(np.percentile(Z, perc_low), np.percentile(Z, perc_high), levels)
    x1, y1 = np.meshgrid(bins, freqs)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=12)
    # plt.suptitle("Spectrogram title", fontsize=15, y=.94)
    plt.contourf(x1, list(reversed(y1)), Z, vmin=None, vmax=None, extent=extent, levels=levels)
   # plt.imshow(spec, cmap='hot', interpolation='none')
  #  plt.contourf(x1, list(reversed(y1)), Z, vmin=-50, vmax=10, extent=extent, levels=levels)
  #   plt.colorbar()
    plt.axis('auto')
    plt.tight_layout()
  #  plt.axis(ymin=0, ymax=200)
  #   plt.show()




if __name__ == "__main__":
    base_filename = r'C:\Users\Jesse\Desktop\buonviso\data_iot\{name}.{ext}'
    fname = "Tem04a00"

    h5_filename = base_filename.format(name=fname, ext="h5")
    raw_signal = get_electrode_data(h5_filename, "A_odor", seg=0)
    rfs = 1000 * Hz
    num_samples = int(raw_signal.duration.rescale(s) * rfs/Hz)

    # # #If you're looking at an electrode, electrode_name will be "E0", "E1", "E2", etc.; seg=2
    resampled_signal = butter_lowpass_filter(raw_signal, rfs/2, raw_signal.sampling_rate)
    resampled_signal = signal.resample(resampled_signal, num_samples)
    resampled_signal = neo.core.AnalogSignal(resampled_signal, units=uV, sampling_rate=rfs, t_start=raw_signal.t_start*ms)
    plot_spectrogram(resampled_signal.reshape(resampled_signal.shape[0],), rfs)