import neo
import math
from scipy.stats import circmean
import scipy.stats as stats
from scipy.signal import hilbert, resample
from quantities import ms, Hz, uV, s
from filters import butter_bp_filter, cheby2_bp_filter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neo.io.nixio import NixIO
from neo.core import AnalogSignal, SpikeTrain
from pip._vendor.progress import spinner
from cmath import polar
from tqdm import tqdm
import sys
import itertools
import random


degree_sign = u"\u00b0"


def custom_round(x, base=5, return_int=True):
    if return_int:
        return int(np.float(base) * round(np.float(x)/np.float(base)))
    else:
        return np.float(base) * round(np.float(x)/np.float(base))


def time_resolved_plv(signal, spikes, start, stop, window, t_step=1000*ms):

    """
    Plots the time-resolved phase-locking value (PLV) measure for a given spike train and LFP signal.

    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
    window: Quantity scalar with dimension time
        Size of the window over which each individual PLV measure is calculated. Note: Smaller window sizes afford
        greater time resolution at the cost of sample size.
    t_step: Quantity scalar with dimension time
        Size of the window step
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    """

    # low, high = filter_range
    duration = stop.rescale(s) - start.rescale(s)
    fs = signal.sampling_rate

    signal = signal.time_slice(start, stop+.001*s)
    # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0) #I'd rather not filter on the chopped signal, creates edge artifact

    # Alternative filters to consider
    # signal = butter_bandpass_filter(signal, low, high, fs, axis=0)
    # signal = firls_bp_filter(signal, low, high, fs, axis=0)

    signal = neo.core.AnalogSignal(signal, units=uV, sampling_rate=fs, t_start=start)
    analytic_signal = hilbert(signal, None, 0)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
    phase_dictionary = dict(zip(list(map(str, (round(signal.times.rescale(ms), 1)))), instantaneous_phase))

    for spike_train in spikes:

        if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
            raise TypeError('spikes is not a Neo.SpikeTrain object!')

        spike_train = spike_train.time_slice(start, stop)
        if len(spike_train) == 0:
            raise TypeError('Given neuron did not spike in the specified time window!')

        all_phases = [None] * len(spike_train)
        all_times = [None] * len(spike_train)

        t = 0
        for n in spike_train:

            n = custom_round(n * 1000, 1000./fs) * ms
            ip_of_spike = phase_dictionary[str(n)]
            all_times[t] = n
            all_phases[t] = ip_of_spike
            t += 1

        all_times = np.array(all_times)
        num_steps = duration.rescale(ms) / t_step.rescale(ms)
        all_uv_radii = [None] * int(num_steps)
        all_uv_phases = [None] * int(num_steps)
        all_scounts = [None] * int(num_steps)
        x_ax = 0
        for i in range(int(start.rescale(ms)), int(stop.rescale(ms)), int(t_step.rescale(ms))):
            if i + int(window.rescale(ms)) < int(stop.rescale(ms)):
                window_start = i
                window_stop = i + int(window.rescale(ms))
                window_times = np.array(np.where((all_times > window_start) & (all_times < window_stop)))
                if window_times.size == 0:
                    all_uv_radii[x_ax] = 0
                    x_ax += 1
                    continue
                window_phases = [all_phases[int(j)] for j in window_times[0]]
                a_rad = map(lambda x: math.radians(x), window_phases)
                a_rad = np.fromiter(a_rad, dtype=np.float)

                a_cos = map(lambda x: math.cos(x), a_rad)
                a_sin = map(lambda x: math.sin(x), a_rad)
                a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)

                uv_x = sum(a_cos)/len(a_cos)
                uv_y = sum(a_sin)/len(a_sin)
                uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
                uv_phase = np.angle(complex(uv_x, uv_y))
                sig = 100 * (1. - np.exp(-1*len(window_phases)*(uv_radius**2)))

                all_uv_radii[x_ax] = sig
                all_uv_phases[x_ax] = uv_phase
                all_scounts[x_ax] = window_times[0].shape[0]
                x_ax += 1

       #  ax1 = plt.subplot(211)
       #  ax2 = plt.subplot(212, polar=True, rmax=np.float(duration))
       #  ax1.plot(np.linspace(int(start.rescale(s)), int(stop.rescale(s)), int(num_steps)), all_uv_radii, 'k', linewidth=0.8, alpha=0.8)
       #  ax1.plot([int(start.rescale(s)), int(stop.rescale(s))], [95, 95], 'limegreen', lw=1, label='95th percentile')
       #  ax1.plot([int(start.rescale(s)), int(stop.rescale(s))], [99, 99], 'green', lw=1, label='99th percentile')
       #  ax1.set_ylabel('Percentile of Coherence under H0', fontsize=8)
       #  ax2.plot(all_uv_phases, np.linspace(1, int(duration), int(num_steps), endpoint=True), 'k', linewidth=0.8, alpha=0.8)
       #  #ax2.plot(all_uv_phases, np.linspace(1, int(duration), int(num_steps), endpoint=True), 'o', linewidth=0.8,
       # #           alpha=0.8)
       # # ax2.set_yticks(())
       #  plt.show()
        return all_uv_radii, all_uv_phases, all_scounts, duration, num_steps


def tf_resolved_plv(signal, spikes, start, stop, window, t_step=1000*ms, bw=5, bw_range=[0, 100], bw_step=5):

    """
    Plots the time-resolved phase-locking value (PLV) measure for a given spike train and LFP signal.

    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
    window: Quantity scalar with dimension time
        Size of the window over which each individual PLV measure is calculated. Note: Smaller window sizes afford
        greater time resolution at the cost of sample size.
    t_step: Quantity scalar with dimension time
        Size of the window step
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    """
    num_steps_f = (bw_range[1] - bw_range[0]) / bw_step
    num_steps_f = round(num_steps_f)
    duration = stop.rescale(s) - start.rescale(s)
    num_steps_t = duration.rescale(ms) / t_step.rescale(ms)
    all_uv_radii = np.empty((int(num_steps_f), int(num_steps_t)))
    all_uv_phases = np.empty((int(num_steps_f), int(num_steps_t)))
    all_uv_radii[:] = np.nan
    all_uv_phases[:] = np.nan
    f_idx = 0
    for f in range(bw_range[0], bw_range[1], bw_step):
        low = f
        high = f+bw_step
        fs = signal.sampling_rate

        signal = signal.time_slice(start, stop+.001*s)
        signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)

        # Alternative filters to consider
        # signal = butter_bandpass_filter(signal, low, high, fs, axis=0)
        # signal = firls_bp_filter(signal, low, high, fs, axis=0)

        signal = neo.core.AnalogSignal(signal, units=uV, sampling_rate=fs, t_start=start)
        analytic_signal = hilbert(signal, None, 0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
        phase_dictionary = dict(zip(list(map(str, (round(signal.times.rescale(ms), 1)))), instantaneous_phase))

        for spike_train in spikes:

            if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
                raise TypeError('spikes is not a Neo.SpikeTrain object!')

            spike_train = spike_train.time_slice(start, stop)
            if len(spike_train) == 0:
                raise TypeError('Given neuron did not spike in the specified time window!')

            all_phases = [None] * len(spike_train)
            all_times = [None] * len(spike_train)

            t = 0
            for n in spike_train:

                n = custom_round(n * 1000, 1000./fs) * ms
                ip_of_spike = phase_dictionary[str(n)]
                all_times[t] = n
                all_phases[t] = ip_of_spike
                t += 1

            all_times = np.array(all_times)
            curr_uv_radii = [None] * int(num_steps_t)
            curr_uv_phases = [None] * int(num_steps_t)
            x_ax = 0
            for i in range(int(start.rescale(ms)), int(stop.rescale(ms)), int(t_step.rescale(ms))):
                window_start = i
                window_stop = i + int(window.rescale(ms))
                window_times = np.array(np.where((all_times > window_start) & (all_times < window_stop)))
                if window_times.size == 0:
                    curr_uv_radii[x_ax] = 0
                    x_ax += 1
                    continue
                window_phases = [all_phases[int(j)] for j in window_times[0]]
                a_rad = map(lambda x: math.radians(x), window_phases)
                a_rad = np.fromiter(a_rad, dtype=np.float)

                a_cos = map(lambda x: math.cos(x), a_rad)
                a_sin = map(lambda x: math.sin(x), a_rad)
                a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)

                uv_x = sum(a_cos)/len(a_cos)
                uv_y = sum(a_sin)/len(a_sin)
                uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
                uv_phase = np.angle(complex(uv_x, uv_y))
                sig = 100 * (1. - np.exp(-1*len(window_phases)*(uv_radius**2)))

                curr_uv_radii[x_ax] = sig
                curr_uv_phases[x_ax] = uv_phase
                x_ax += 1
            all_uv_radii[f_idx,:] = curr_uv_radii
            all_uv_phases[f_idx, :] = curr_uv_phases
            f_idx+=1
    # plt.pcolor(np.linspace(int(start.rescale(s)), int(stop.rescale(s)), int(num_steps_t)), np.linspace(bw_range[0], bw_range[1], num_steps_f), all_uv_radii)
    # plt.show()
    return all_uv_radii

def get_spike_phase_hist(signal, spikes, start, stop, filter_range=[20, 55], nbins=18, plot=False):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    phase_bins = np.linspace(-180, 180, nbins+1)
    low, high = filter_range
    fs = signal.sampling_rate
    start = round(start,3)
    if stop:
        stop = round(stop,3)
        signal = signal.time_slice(start, stop + .001*s)
    # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
    signal = neo.core.AnalogSignal(signal, units=uV, sampling_rate=fs, t_start=start)

    analytic_signal = hilbert(signal, None, 0)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
    phase_dictionary = dict(zip(list(map(str, (round(signal.times.rescale(ms),0)))), instantaneous_phase))

    for spike_train in spikes:

        if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
            raise TypeError('spikes is not a Neo.SpikeTrain object!')
        if stop:
            spike_train = spike_train.time_slice(start, stop)

        all_phases = [None] * len(spike_train)

        t = 0
        for n in spike_train:
          #  n = round(n, 3)
            n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
            ip_of_spike = phase_dictionary[str(n)]
            all_phases[t] = ip_of_spike
            t += 1

        a_deg = map(lambda x: np.ndarray.item(x), all_phases)
        a_rad = map(lambda x: math.radians(x), a_deg)

        dig_phases = np.digitize(all_phases, phase_bins, nbins + 1)
        spike_phase_hist = np.zeros(nbins+1)
        for bin in range(0, nbins + 1):
            spike_phase_hist[bin] = np.sum(dig_phases == bin)

        if plot:
            bins = np.linspace(-np.pi, np.pi, nbins + 1)
            ax = plt.subplot(1, 1, 1, polar=True)
            plt.bar(bins, spike_phase_hist,
                    width=bins[1] - bins[0],
                    bottom=0.0)

        return spike_phase_hist

def get_spike_phase_hist_onhilb(analytic_sig, spikes, start, stop, filter_range=[20, 55], nbins=18, plot=False, fs=1000):

    """
    Parameters
    ----------
    analytic_sig: hilbert transformed lfp
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    phase_bins = np.linspace(-180, 180, nbins+1)
    low, high = filter_range
    start = round(start,3)
    if stop:
        stop = round(stop,3)

    analytic_sig = analytic_sig[int(round(start*fs)):int(round(stop*fs))]
    instantaneous_phase = np.unwrap(np.angle(analytic_sig, deg=True))
    phase_dictionary = dict(zip(list(map(str, (np.round(np.arange(start*1000 / s,stop*1000 / s,1))))), instantaneous_phase))

    for spike_train in spikes:

        if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
            raise TypeError('spikes is not a Neo.SpikeTrain object!')
        if stop:
            spike_train = spike_train.time_slice(start, stop - .001*s) #end time is inclusive

        all_phases = [None] * len(spike_train)

        t = 0
        for n in spike_train:
          #  n = round(n, 3)
            n = custom_round(n * 1000, 1000. / fs)   # how is custom round different than normal round?
            try:
                ip_of_spike = phase_dictionary[str(n)+'.0']
            except:
                print('t')
            all_phases[t] = ip_of_spike
            t += 1

        a_deg = map(lambda x: np.ndarray.item(x), all_phases)
        a_rad = map(lambda x: math.radians(x), a_deg)

        dig_phases = np.digitize(all_phases, phase_bins, nbins + 1)
        spike_phase_hist = np.zeros(nbins+1)
        for bin in range(0, nbins + 1):
            spike_phase_hist[bin] = np.sum(dig_phases == bin)

        if plot:
            bins = np.linspace(-np.pi, np.pi, nbins + 1)
            ax = plt.subplot(1, 1, 1, polar=True)
            plt.bar(bins, spike_phase_hist,
                    width=bins[1] - bins[0],
                    bottom=0.0)

        return spike_phase_hist

def ppc(signal, spikes, start, stop, filter_range=[20, 55]):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    low, high = filter_range
    fs = signal.sampling_rate
    if stop:
        signal = signal.time_slice(start, stop + .001*s)
    # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
    signal = neo.core.AnalogSignal(signal, units=uV, sampling_rate=fs, t_start=start)

    analytic_signal = hilbert(signal, None, 0)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
    phase_dictionary = dict(zip(list(map(str, (round(signal.times.rescale(ms),0)))), instantaneous_phase))

    for spike_train in spikes:

        if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
            raise TypeError('spikes is not a Neo.SpikeTrain object!')
        if stop:
            spike_train = spike_train.time_slice(start, stop)
        if len(spike_train) == 0:
            raise TypeError('Given neuron did not spike in the specified time window!')
        if len(spike_train) < 50:
            print('Given neuron spikes less than 50 times')
            raise TypeError('Given neuron spikes less than 50 times')

        all_phases = [None] * len(spike_train)

        t = 0
        for n in spike_train:
          #  n = round(n, 3)
            n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
            ip_of_spike = phase_dictionary[str(n)]
            all_phases[t] = ip_of_spike
            t += 1

        a_deg = map(lambda x: np.ndarray.item(x), all_phases)
        a_rad = map(lambda x: math.radians(x), a_deg)

        a_rad = np.fromiter(a_rad, dtype=np.float)
        a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad)

        all_com = list(itertools.combinations(a_complex, 2))
        dp_array = np.empty(int(len(a_rad) * (len(a_rad) - 1) / 2))

        pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
        d = 0
        for combination in pbar:
            dp = np.dot(combination[0], combination[1])
            dp_array[d] = dp
            d += 1
        dp_sum = np.sum(dp_array)
        ppc = dp_sum / len(dp_array)

        return ppc

def ppc_overtrials(signal, spikes, segmentTimes, segment):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    segment: range around events to segment

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    all_phases = []

    for i in range(0, len(segmentTimes)):
        start = (round(segmentTimes[i], 3) + segment[0])*s
        stop = (round(segmentTimes[i], 3) + segment[1])*s
        fs = signal.sampling_rate
        if stop:
            seg_signal = signal.time_slice(start, stop + .001*s)
        # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
        seg_signal = neo.core.AnalogSignal(seg_signal, units=uV, sampling_rate=fs, t_start=start)

        analytic_signal = hilbert(seg_signal, None, 0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
        phase_dictionary = dict(zip(list(map(str, (round(seg_signal.times.rescale(ms),0)))), instantaneous_phase))

        for spike_train in spikes:

            if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
                raise TypeError('spikes is not a Neo.SpikeTrain object!')
            if stop:
                spike_train = spike_train.time_slice(start, stop)
            # if len(spike_train) == 0:
            #     raise TypeError('Given neuron did not spike in the specified time window!')

            for n in spike_train:
              #  n = round(n, 3)
                n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
                ip_of_spike = phase_dictionary[str(n)]
                all_phases.append(ip_of_spike)

    a_deg = map(lambda x: np.ndarray.item(x), all_phases)
    a_rad = map(lambda x: math.radians(x), a_deg)

    a_rad = np.fromiter(a_rad, dtype=np.float)

    if len(a_rad) < 2:
        print('Less than 2 spikes in time window')
        raise TypeError('Less than 2 spikes in time window')
    a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad)

    all_com = list(itertools.combinations(a_complex, 2))
    dp_array = np.empty(int(len(a_rad) * (len(a_rad) - 1) / 2))

    pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    d = 0
    for combination in pbar:
        dp = np.dot(combination[0], combination[1])
        dp_array[d] = dp
        d += 1
    dp_sum = np.sum(dp_array)
    ppc = dp_sum / len(dp_array)

    # #Bootstrap
    # n_bs = 200
    # ppc_bs = [None] * n_bs
    # for b in range(0, n_bs):
    #     a_rad_bs = np.empty((a_rad.shape))
    #     a_rad_bs[:] = np.nan
    #     for i in range(0, a_rad.shape[0]):
    #         a_rad_bs[i] = random.choice(a_rad)
    #     a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad_bs)
    #
    #     all_com = list(itertools.combinations(a_complex, 2))
    #     dp_array = np.empty(int(len(a_rad_bs) * (len(a_rad_bs) - 1) / 2))
    #
    #     pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    #     d = 0
    #     for combination in pbar:
    #         dp = np.dot(combination[0], combination[1])
    #         dp_array[d] = dp
    #         d += 1
    #     dp_sum = np.sum(dp_array)
    #     ppc_bs[b] = dp_sum / len(dp_array)

    return ppc, np.nan, all_phases #  np.asarray(ppc_bs)

def plv_overtrials(signal, spikes, segmentTimes, segment, pow_sig=None, justCount=False, countByPhase=False, onlyifPow=False, prefPhase=135, returnPhase=False):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    segment: range around events to segment

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    all_phases = []
    use_phasepref = []
    if prefPhase is not 'auto':
        prefPhase = math.radians(prefPhase)
    for i in range(0, len(segmentTimes)):
        start = (round(segmentTimes[i], 3) + segment[0])*s
        stop = (round(segmentTimes[i], 3) + segment[1])*s
        fs = signal.sampling_rate
        if stop:
            seg_signal = signal.time_slice(start, stop + .001*s)
        # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
        seg_signal = neo.core.AnalogSignal(seg_signal, units=uV, sampling_rate=fs, t_start=start)

        analytic_signal = hilbert(seg_signal, None, 0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
        phase_dictionary = dict(zip(list(map(str, (round(seg_signal.times.rescale(ms),0)))), instantaneous_phase))

        for spike_train in spikes:

            if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
                raise TypeError('spikes is not a Neo.SpikeTrain object!')
            if stop:
                spike_train = spike_train.time_slice(start, stop)
            if justCount:
                return len(spike_train), 0
            # if len(spike_train) == 0:
            #     raise TypeError('Given neuron did not spike in the specified time window!')

            for n in spike_train:
              #  n = round(n, 3)
                n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
                ip_of_spike = phase_dictionary[str(n)]
                all_phases.append(ip_of_spike)
                if onlyifPow:
                    if pow_sig[int(n-start)-1] > 6:
                        use_phasepref.append(True)
                    else:
                        use_phasepref.append(False)

    a_deg = map(lambda x: np.ndarray.item(x), all_phases)
    a_rad = map(lambda x: math.radians(x), a_deg)

    a_rad = np.fromiter(a_rad, dtype=np.float)

    a_cos = map(lambda x: math.cos(x), a_rad)
    a_sin = map(lambda x: math.sin(x), a_rad)
    a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)
    if len(a_rad) < 1:
        print('No spikes in time window')
        if countByPhase or justCount:
            return np.nan, np.nan
        raise TypeError('No spikes in time window')
    uv_x = sum(a_cos) / len(a_cos)
    uv_y = sum(a_sin) / len(a_sin)
    uv_radius = np.sqrt((uv_x * uv_x) + (uv_y * uv_y))
    uv_phase = np.angle(complex(uv_x, uv_y))
    if returnPhase:
        return uv_phase
    if countByPhase:
        all_dist = []
        if prefPhase is 'auto':
            prefPhase = uv_phase
            print(str(prefPhase))
        total_dist = 0
        binary = False
        participation = True #(on a scale of -.5 to .5)
        for p in range(0, len(all_phases)):
            if onlyifPow:
                if use_phasepref[p]:
                    phase = all_phases[p]
                    v_phase = [np.cos(math.radians(phase[0])), np.sin(math.radians(phase[0]))]
                    v_prefPhase = [np.cos(prefPhase), np.sin(prefPhase)]
                    dist = np.arccos(np.dot(v_phase, v_prefPhase))  # dist in radians
                    # test = math.atan2(v_prefPhase[0] - v_phase[0], v_prefPhase[1] - v_phase[1])
                    if binary:
                        if dist < 0.174533:  # corresponds to 10 degree distance
                            total_dist += 1
                    else:
                        if participation:
                            total_dist += (np.pi/2 - dist) / np.pi  # supposed to be 1 - angular distance between points, normalized from 0 to 1
                        else:
                            total_dist += (np.pi - dist) / np.pi
                else:
                    if participation:
                        total_dist = np.nan
                    else:
                        total_dist += .5
            else:
                phase = all_phases[p]
                v_phase = [np.cos(math.radians(phase[0])), np.sin(math.radians(phase[0]))]
                v_prefPhase = [np.cos(prefPhase), np.sin(prefPhase)]
                dist = np.arccos(np.dot(v_phase, v_prefPhase)) #dist in radians
                # test = math.atan2(v_prefPhase[0] - v_phase[0], v_prefPhase[1] - v_phase[1])
                if binary:
                    if dist < 0.174533: #corresponds to 10 degree distance
                        total_dist += 1
                else: #Do by 1 - distance between phases
                    if participation:
                        total_dist += (np.pi / 2 - dist) / np.pi  # supposed to be 1 - angular distance between points, normalized from 0 to 1
                        all_dist.append((np.pi / 2 - dist) / np.pi)
                    else:
                        total_dist += (np.pi - dist) / np.pi
                        all_dist.append((np.pi / 2 - dist) / np.pi)
        if onlyifPow:
            print('Proportion of spikes weighted by phase: ', str(sum(use_phasepref) / len(use_phasepref)))
        return all_dist, 1 #currently designed to take average distance of spikes

    if len(a_rad) < 2:
        print('Less than 2 spikes in time window')
        raise TypeError('Less than 2 spikes in time window')
    p, pctile = random_phase_bootstrapping(len(all_phases), 250, uv_radius) #2 should be 250 or 1000

    return uv_radius, p, pctile, uv_phase, all_phases, uv_x, uv_y

def plv_overtrials_euc(signal, spikes, segmentTimes, segment, fs, pow_sig=None, justCount=False, countByPhase=False, onlyifPow=False, prefPhase=135, returnPhase=False):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    segment: range around events to segment

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    all_phases = []
    use_phasepref = []
    if prefPhase is not 'auto':
        prefPhase = math.radians(prefPhase)
    for i in range(0, len(segmentTimes)):
        start = (round(segmentTimes[i], 3) + segment[0])*s
        stop = (round(segmentTimes[i], 3) + segment[1])*s
        if not justCount:
            if stop:
                seg_signal = signal.time_slice(start, stop + .001*s)
        # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
            seg_signal = neo.core.AnalogSignal(seg_signal, units=uV, sampling_rate=fs, t_start=start)

            analytic_signal = hilbert(seg_signal, None, 0)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
            phase_dictionary = dict(zip(list(map(str, (round(seg_signal.times.rescale(ms),0)))), instantaneous_phase))

        for spike_train in spikes:

            if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
                raise TypeError('spikes is not a Neo.SpikeTrain object!')
            if stop:
                try:
                    spike_train = spike_train.time_slice(start, stop)
                except:
                    print('No spikes found in segment')
                    return 0, 0
            if justCount:
                return len(spike_train), 0
            # if len(spike_train) == 0:
            #     raise TypeError('Given neuron did not spike in the specified time window!')

            for n in spike_train:
              #  n = round(n, 3)
                n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
                ip_of_spike = phase_dictionary[str(n)]
                all_phases.append(ip_of_spike)
                if onlyifPow:
                    if pow_sig[int(n-start)-1] > 6:
                        use_phasepref.append(True)
                    else:
                        use_phasepref.append(False)

    a_deg = map(lambda x: np.ndarray.item(x), all_phases)
    a_rad = map(lambda x: math.radians(x), a_deg)

    a_rad = np.fromiter(a_rad, dtype=np.float)

    a_cos = map(lambda x: math.cos(x), a_rad)
    a_sin = map(lambda x: math.sin(x), a_rad)
    a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)
    if len(a_rad) < 1:
        print('No spikes in time window')
        if countByPhase or justCount:
            return np.nan, np.nan
        raise TypeError('No spikes in time window')
    uv_x = sum(a_cos) / len(a_cos)
    uv_y = sum(a_sin) / len(a_sin)
    uv_radius = np.sqrt((uv_x * uv_x) + (uv_y * uv_y))
    uv_phase = np.angle(complex(uv_x, uv_y))
    if returnPhase:
        return uv_phase
    if countByPhase:
        all_dist = []
        if prefPhase is 'auto':
            prefPhase = uv_phase
            print(str(prefPhase))
        total_dist = 0
        binary = False
        participation = False #(on a scale of -.5 to .5)
        for p in range(0, len(all_phases)):
            if onlyifPow:
                if use_phasepref[p]:
                    phase = all_phases[p]
                    v_phase = [np.cos(math.radians(phase[0])), np.sin(math.radians(phase[0]))]
                    v_prefPhase = [np.cos(prefPhase), np.sin(prefPhase)]
                    dist = np.arccos(np.dot(v_phase, v_prefPhase))  # dist in radians
                    # test = math.atan2(v_prefPhase[0] - v_phase[0], v_prefPhase[1] - v_phase[1])
                    if binary:
                        if dist < 0.174533:  # corresponds to 10 degree distance
                            total_dist += 1
                    else:
                        if participation:
                            total_dist += (np.pi/2 - dist) / np.pi  # supposed to be 1 - angular distance between points, normalized from 0 to 1
                        else:
                            total_dist += (np.pi - dist) / np.pi
                else:
                    if participation:
                        total_dist = np.nan
                    else:
                        total_dist += .5
            else:
                phase = all_phases[p]
                v_phase = [np.cos(math.radians(phase[0])), np.sin(math.radians(phase[0]))]
                v_prefPhase = [np.cos(prefPhase), np.sin(prefPhase)]
                dist = np.arccos(np.dot(v_phase, v_prefPhase)) #dist in radians
                # test = math.atan2(v_prefPhase[0] - v_phase[0], v_prefPhase[1] - v_phase[1])
                if binary:
                    if dist < 0.174533: #corresponds to 10 degree distance
                        total_dist += 1
                else: #Do by 1 - distance between phases
                    if participation:
                        total_dist += (np.pi / 2 - dist) / np.pi  # supposed to be 1 - angular distance between points, normalized from 0 to 1
                    else:
                        total_dist += (np.pi - dist) / np.pi
        if onlyifPow:
            print('Proportion of spikes weighted by phase: ', str(sum(use_phasepref) / len(use_phasepref)))
        return total_dist, 1 #currently designed to take average distance of spikes

    if len(a_rad) < 2:
        print('Less than 2 spikes in time window')
        raise TypeError('Less than 2 spikes in time window')
    p, pctile = random_phase_bootstrapping(len(all_phases), 250, uv_radius) #2 should be 250 or 1000

    return uv_radius, p, pctile, uv_phase, all_phases, uv_x, uv_y

def ppc_overtrials_byc(data, spikes, evtTimes, byc_df):

    """
    Parameters
    ----------
    signal: Neo AnalogSignal
        Resampled electrode signal from which we will derive LFP phase information
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter
    segment: range around events to segment

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    all_phases = []
    segment = [0, 1]
    fs = 1000 * Hz
    for i in range(0, len(evtTimes)):
        start = (round(evtTimes[i], 3) + segment[0])
        end = (round(evtTimes[i], 3) + segment[1])
        curr_byc = byc_df[byc_df['is_burst']]
        curr_byc = curr_byc[curr_byc['sample_last_trough'] > start * 1000]
        curr_byc = curr_byc[curr_byc['sample_last_trough'] < end * 1000]
        # choose electrode where bursts have highest amplitude
        max_amp = 0
        for e in range(0, data.shape[0]):
            elec_byc = curr_byc[curr_byc['subject_id'] == e]
            if np.mean(elec_byc['amp_fraction']) > max_amp:
                max_amp = np.mean(elec_byc['amp_fraction'])
                max_elec = e
        elec_byc = curr_byc[curr_byc['subject_id'] == max_elec]
        # Do one cycle at a time, because there might be multiple bursts within the stimuli
        for b in range(0, elec_byc.shape[0]):
            signal = neo.core.AnalogSignal(data[max_elec, :], units=uV,
                                           sampling_rate=fs, t_start=0 * s)
            start = (elec_byc['sample_last_trough'].iloc[b] / 1000) * s
            stop = (elec_byc['sample_next_trough'].iloc[b] / 1000) * s
            if stop:
                try:
                    seg_signal = signal.time_slice(start, stop + .001*s)
                except:
                      print('t')
            # signal = cheby2_bp_filter(signal, fs, low, high, order=5, axis=0)
            seg_signal = neo.core.AnalogSignal(seg_signal, units=uV, sampling_rate=fs, t_start=start)

            analytic_signal = hilbert(seg_signal, None, 0)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal, deg=True))
            phase_dictionary = dict(zip(list(map(str, (round(seg_signal.times.rescale(ms),0)))), instantaneous_phase))

            for spike_train in spikes:

                if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
                    raise TypeError('spikes is not a Neo.SpikeTrain object!')
                if stop:
                    spike_train = spike_train.time_slice(start, stop)
                # if len(spike_train) == 0:
                #     raise TypeError('Given neuron did not spike in the specified time window!')

                for n in spike_train:
                  #  n = round(n, 3)
                    n = custom_round(n * 1000, 1000. / fs) * ms  # how is custom round different than normal round?
                    ip_of_spike = phase_dictionary[str(n)]
                    all_phases.append(ip_of_spike)

    a_deg = map(lambda x: np.ndarray.item(x), all_phases)
    a_rad = map(lambda x: math.radians(x), a_deg)

    a_rad = np.fromiter(a_rad, dtype=np.float)

    if len(a_rad) < 10:
        print('Less than 10 spikes in time window')
        raise TypeError('Less than 10 spikes in time window')
    a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad)

    all_com = list(itertools.combinations(a_complex, 2))
    dp_array = np.empty(int(len(a_rad) * (len(a_rad) - 1) / 2))

    pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    d = 0
    for combination in pbar:
        dp = np.dot(combination[0], combination[1])
        dp_array[d] = dp
        d += 1
    dp_sum = np.sum(dp_array)
    ppc = dp_sum / len(dp_array)

    # #Bootstrap
    # n_bs = 200
    # ppc_bs = [None] * n_bs
    # for b in range(0, n_bs):
    #     a_rad_bs = np.empty((a_rad.shape))
    #     a_rad_bs[:] = np.nan
    #     for i in range(0, a_rad.shape[0]):
    #         a_rad_bs[i] = random.choice(a_rad)
    #     a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad_bs)
    #
    #     all_com = list(itertools.combinations(a_complex, 2))
    #     dp_array = np.empty(int(len(a_rad_bs) * (len(a_rad_bs) - 1) / 2))
    #
    #     pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    #     d = 0
    #     for combination in pbar:
    #         dp = np.dot(combination[0], combination[1])
    #         dp_array[d] = dp
    #         d += 1
    #     dp_sum = np.sum(dp_array)
    #     ppc_bs[b] = dp_sum / len(dp_array)

    return ppc, np.nan, all_phases #  np.asarray(ppc_bs)


def random_phase_bootstrapping(n, k=10000, plv=None, ppc=None):
    """
    Parameters
    ----------
    n: int
        Sample size (number of spikes)
    k: int
        # of trials to bootstrap
    plv: float
        PLV score to calculate the probability of being observed in a random distribution of n phases.
    ppc: float
        PPC value to calculate the probability of being observed in a random distribution of n phases

    Returns
    ----------
    pvals: list
        List of one or two values, corresponding to the p-values of observing the given PLV/PPC values under a null
        hypothesis that phases are randomly distributed.
    """
    plv_boot, ppc_boot = np.zeros(k), np.zeros(k)

    for i in tqdm(range(k), desc="Bootstrapping trials..."):
        all_phases = (np.random.rand(n) * 360) - 180
        a_rad = map(lambda x: math.radians(x), all_phases)
        a_rad = np.fromiter(a_rad, dtype=np.float)
        a_cos = map(lambda x: math.cos(x), a_rad)
        a_sin = map(lambda x: math.sin(x), a_rad)

        a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)

        uv_x = sum(a_cos) / len(a_cos)
        uv_y = sum(a_sin) / len(a_sin)
        uv_radius = np.sqrt((uv_x * uv_x) + (uv_y * uv_y))

        if plv is not None:
            plv_boot[i] = uv_radius

        if ppc_boot is not None:
            a_complex = map(lambda x: [math.cos(x), math.sin(x)], a_rad)
            all_com = list(itertools.combinations(a_complex, 2))
            dp_array = np.empty(int(len(a_rad) * (len(a_rad) - 1) / 2))
            d = 0
            for combination in all_com:
                dp = np.dot(combination[0], combination[1])
                dp_array[d] = dp
                d += 1
            dp_sum = np.sum(dp_array)
            ppc_random = dp_sum / len(dp_array)
            ppc_boot[i] = ppc_random

    # plt.hist(ppc_boot, bins=100)
    # plt.ylabel('Probability', fontsize=10)
    # plt.xlabel('Vector sum length', fontsize=10)
    # plt.show()

    pvals = []
    if plv is not None:
        plv_percentile = stats.percentileofscore(plv_boot, plv)
        plv_pval = (100. - plv_percentile) / 100.
        pvals.append(plv_pval)
        zscore = (plv - np.mean(plv_boot)) / np.std(plv_boot)
    if ppc is not None:
        ppc_percentile = stats.percentileofscore(ppc_boot, ppc)
        ppc_pval = (100. - ppc_percentile) / 100.
        pvals.append(ppc_pval)

    return pvals, plv_percentile

