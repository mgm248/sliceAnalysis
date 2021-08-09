from scipy.signal import butter, filtfilt, freqz, lfilter, decimate, bessel, firls, iirnotch, cheby2
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from quantities import Hz, uV, ms, s


def iir_notch(data, fs, frequency=60, quality=15., axis=-1):
    """
    For removing 60 Hz noise (sort of).

    Filter parameters
    -------
    data: Numpy ndarray
        Your time series of voltage values
    fs: quantity
        Sampling rate in Hz
    frequency: quantity
        Frequency we want to notch out, in Hz
    quality: float
        Scipy's iirnotch parameter; relates to the bandwidth of the notch
        quality = norm_freq/bandwidth (higher quality means lower bandwidth)
        EXAMPLE: quality=15., frequency=60, fs=1000
                 15 = .12/bandwidth
                 bandwidth = .008 (normalized) = 4 Hz
    axis: int
        axis along which data is filtered
    """
    frequency = int(frequency.rescale(Hz))
    norm_freq = frequency/(fs/2) #normalize by nyquist
    b, a = iirnotch(norm_freq, quality) # b,a are our filter coefficients
# PLOT FREQUENCY RESPONSE (TRANSFER FUNCTION)
#     w, h = signal.freqz(b, a)
#     plt.plot(w*0.5*fs/np.pi, 20 * np.log10(abs(h)), 'b')
#     plt.show()
    y = filtfilt(b, a, data, padlen=0, axis=axis) #filtfilt filters both ways (non-causal), so there is no phase shift.
    return y


def butter_lowpass_filter(data, fs, cutoff, order=4, axis=-1) -> object:
    """
    Used mostly for filtering prior to resampling (where cutoff is nyquist).

    Filter parameters
    -------
    data: Numpy ndarray
        Your time series of voltage values
    fs: quantity
        Sampling rate in Hz
    cutoff: quantity
        Frequency we want to notch out, in Hz
    order: int
        Order of the filter
    axis: int
        axis along which data is filtered
    """
    cutoff
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     w, h = signal.freqz(b, a)
#     plt.plot(w*0.5*fs/np.pi, 20 * np.log10(abs(h)), 'b')
#     plt.show()
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y

def butter_highpass_filter(data, fs, cutoff, order=4, axis=-1):
    """
    Filter parameters
    -------
    data: Numpy ndarray
        Your time series of voltage values
    fs: quantity
        Sampling rate in Hz
    cutoff: quantity
        Frequency we want to notch out, in Hz
    order: int
        Order of the filter
    axis: int
        axis along which data is filtered
    """
    cutoff
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    # w, h = signal.freqz(b, a)
    # plt.plot(w*0.5*fs/np.pi, 20 * np.log10(abs(h)), 'b')
    # plt.show()
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y

def butter_bp_filter(data, fs, low, hi, order=5, axis=-1):
    """
    Standard, non-aggressive Butterworth bandpass filter.

    Filter parameters
    -------
    data: Numpy ndarray
        Your time series of voltage values
    fs: quantity
        Sampling rate in Hz
    low: quantity
        low end of bandpass, in Hz
    hi: quantity
        high end of bandpass, in Hz
    order: int
        Order of the filter
    axis: int
        axis along which data is filtered
    """
    nyq = 0.5 * fs
    low = low / nyq
    hi = hi / nyq
    b, a = butter(order, [low, hi], btype='band')
    w, h = signal.freqz(b, a)
    # plt.plot(w*0.5*fs/np.pi, 20 * np.log10(abs(h)), 'b')
    # plt.show()
    y = filtfilt(b, a, data, axis=axis, padtype=None)
    #y = lfilter(b, a, data)
    return y


def cheby2_bp_filter(data, fs, low, hi, order=4, rs=40, axis=-1):
    """
    More aggressive, type II Chebyshev bandpass filter.
    Type II Chebyshev filter has ringing only in the stopband.

    Filter parameters
    -------
    data: Numpy ndarray
        Your time series of voltage values
    fs: quantity
        Sampling rate in Hz
    low: quantity
        low end of bandpass, in Hz
    hi: quantity
        high end of bandpass, in Hz
    order: int
        Order of the filter
    rs: int
        rs parameter fed into Scipy's Cheby2 function.
        positive number specifying the minimum attenuation desired in decibels
    axis: int
        axis along which data is filtered
    """
    nyquist = 0.5 * fs
    rsf_low, rsf_hi = (low-5)/nyquist, (hi+5)/nyquist
    rsf = [np.float(rsf_low), np.float(rsf_hi)] #rsf contains the frequencies at which the signal is attenuated to rs
    b, a = cheby2(order, rs, rsf, 'bandpass')
    y = filtfilt(b, a, data, axis=axis, padtype=None)
    return y


