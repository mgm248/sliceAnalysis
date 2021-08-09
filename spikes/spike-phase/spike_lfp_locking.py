from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from organize_sc_results import get_circus_manual_result_DF
import filters
import neo
from quantities import uV, Hz, s
from scipy.signal import hilbert
import math
import itertools
from tqdm import tqdm
import sys

fname = 'slice1_merged.h5'
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\\'
Fs = 20000
rfs = 1000
rec_fname = '2021-08-04T11-26-10McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'

def PPC(SP):
   # a_deg = map(lambda x: np.ndarray.item(x), SP)
   # a_rad = map(lambda x: math.radians(x), a_deg)

   # a_rad = np.fromiter(a_rad, dtype=np.float)
    a_complex = map(lambda x: [math.cos(x), math.sin(x)], SP)

    all_com = list(itertools.combinations(a_complex, 2))
    dp_array = np.empty(int(len(SP) * (len(SP) - 1) / 2))

    pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    d = 0
    for combination in pbar:
        dp = np.dot(combination[0], combination[1])
        dp_array[d] = dp
        d += 1
    dp_sum = np.sum(dp_array)
    return dp_sum / len(dp_array)

lfp_data = np.load(ffolder + 'filt_and_rsamp\\' + fname + '.npy', allow_pickle=True)
circus_df = get_circus_manual_result_DF(ffolder+'Analysis\\spyking-circus\\'+rec_fname+'\\'+rec_fname+'times.GUI\\{file}.{ext}', get_electrodes=True, get_groups=False, fs=20000*Hz)

elec = 'C8'
filt_gamma = filters.butter_bp_filter(lfp_data.item()[elec], rfs, 20, 50, order=3, axis=0)
neo_signal = neo.core.AnalogSignal(filt_gamma, units=uV,
                                   sampling_rate=rfs * Hz, t_start=0 * s)
analytic_signal = hilbert(neo_signal, None, 0)
instantaneous_phases = np.unwrap(np.angle(analytic_signal, deg=False))
circus_df['PPC'] = [np.nan for x in range(0, circus_df.shape[0])]
circus_df['N spikes'] = [np.nan for x in range(0, circus_df.shape[0])]

for i, row in circus_df.iterrows():
    phases = []
    for sp in row['Data']:
        phases.append(instantaneous_phases[np.round(sp).astype(int)])

    nbins = 18
    phase_bins = np.linspace(-np.pi, np.pi, nbins + 1)
    dig_phases = np.digitize(phases, phase_bins, nbins + 1)  # nbins+1 is not doing anything, right??
    spike_phase_hist = np.zeros(nbins + 1)
    for bin in range(0, nbins + 1):
        spike_phase_hist[bin] = np.sum(dig_phases == bin)

    plt.subplot(1,1,1,polar=True)
    plt.bar(phase_bins, spike_phase_hist,
            width=phase_bins[1] - phase_bins[0],
            bottom=0.0)
    frame1 = plt.gca()
    plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\spike-phase\C8_lfp\\'+row['Electrode']+'_'+str(row['ID']))
    plt.close()

    circus_df.loc[i,'PPC'] = PPC(phases)
    circus_df.loc[i,'N spikes'] = len(phases)

    circus_df.to_pickle(
        r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\spike-phase\C8_lfp\circus_df.pkl')
    circus_df.to_csv(
        r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\spike-phase\C8_lfp\\circus_df.csv')

