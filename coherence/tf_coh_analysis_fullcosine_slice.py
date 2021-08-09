from ceed.analysis import CeedDataReader

import read_experiments
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt

fname = 'slice3_merged.h5'
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-10\\'
segment = (-4,8)
baseline=(-3, -1)
Fs = 20000
rfs = 1000
ceed_data = ffolder+fname

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()

reader.electrodes_data = np.load(ffolder+'filt_and_rsamp\\' + fname+'.npy',allow_pickle=True)

# Init data for mne
# Read in relevant experiment times
my_annot = read_experiments.read_experiment_fullCosine(reader, Fs)

#elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
elecCombo = ['F2','F6']
info = mne.create_info(elecCombo, sfreq=rfs)

mne_data = np.empty((len(elecCombo), reader.electrodes_data.item()['B5'].shape[0]))
i = 0
for elec in elecCombo:
    mne_data[i, :] = reader.electrodes_data.item()[elec]
    i += 1

raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
events, event_ids = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_ids['Experiment Stage A+BCondition 0s'], tmin=segment[0], tmax=segment[1], event_repeated='drop', baseline=None)

freqs = np.linspace(1, 100, 100)  # define frequencies of interest
# n_cycles = np.arange(1, 10, 3.)
n_cycles = freqs / 2
tmp_pow, tmp_itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True, average=True, picks=epochs.picks)
#  tmp_pow.apply_baseline(mode='percent', baseline=(baseline[0] - segment_size[0], baseline[1] - segment_size[0]))
tmp_pow.apply_baseline(mode='percent', baseline=baseline)

file_pow = tmp_pow.data

plt.figure()

for i in range(0,2): #plot tf pow for each electrode
    plt.subplot(1, 4, i+1)
    fig = plt.imshow(np.squeeze(tmp_pow.data[i,:,:]),
                     extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                     aspect='auto', origin='lower', interpolation='none')
    # plt.clim([0, 50000])
    plt.title(elecCombo[i])
    plt.clim(0, 2)

"""Compute coherence between 2 electrodes"""

con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method=['coh'],
                                                                               sfreq=int(rfs), mode='cwt_morlet',
                                                                               cwt_freqs=freqs, cwt_n_cycles=n_cycles,
                                                                               verbose=True)
#   con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs_data, method=['coh'], sfreq=int(rfs), mode ='fourier', verbose=True)

con_norm = con[1, 0, :, :].T / np.mean(con[1, 0, :, int((baseline[0] - segment[0]) * int(rfs)): int(
    (baseline[1] - segment[0]) * int(rfs))], 1)

plt.subplot(1, 4, 3)
plt.imshow(con[1, 0, :, :], extent=[tmp_pow.times[0], tmp_pow.times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower')
plt.title('Coh raw')

plt.subplot(1, 4, 4)
plt.imshow(con_norm[:, :].T, extent=[tmp_pow.times[0], tmp_pow.times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower')
plt.title('Coh norm')
plt.show()

