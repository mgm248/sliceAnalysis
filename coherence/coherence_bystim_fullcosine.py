import read_experiments
from elephant.spectral import welch_coherence
import neo
import math
from scipy import signal
from scipy.signal import hilbert
from quantities import ms, Hz, uV, s
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ceed.analysis import CeedDataReader
import sys
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook
import mne

new = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9',
      'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
      'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
      'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
      'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
      'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
      'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
      'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
      'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12',
      'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11',
      'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10',
      'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
      ]


if __name__ == "__main__":

    Fs = 1000
    segment=[0,5]
    #don't forget to make an excel file!
    fold = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-10\\'
    slice = '5'
    freqs = np.linspace(1, 100, 100)  # define frequencies of interest
    n_cycles = freqs
    ceed_file = fold + 'slice'+slice+'_merged.h5'
    h5_file = fold +'filt_and_rsamp\\' + 'slice'+slice+'_merged.h5.npy'
    slice_data = np.load(h5_file, allow_pickle=True)
    # create instance that can load the data
    reader = CeedDataReader(ceed_file)
    print('Created reader for file {}'.format(reader.filename))
    # open the data file
    reader.open_h5()

    sheet = "slice" + str(slice)
    my_annot = read_experiments.read_experiment_fullCosine(reader, Fs=20000)

    ref_elec = 'J11'
    mne_data = np.empty((2, slice_data.item()['B5'].shape[0]))
    mne_data[0, :] = slice_data.item()[ref_elec]
    coh_0s = []
    coh_pt083s = []
    coh_pt0083s = []
    for elec in new:
        info = mne.create_info([ref_elec, elec], sfreq=Fs)
        mne_data[1, :] = slice_data.item()[elec]

        raw = mne.io.RawArray(mne_data, info)
        raw.set_annotations(my_annot)
        events, event_ids = mne.events_from_annotations(raw)
        for experiment in event_ids:
            epochs = mne.Epochs(raw, events, event_ids[experiment], tmin=segment[0], tmax=segment[1],
                                event_repeated='drop', baseline=None)

            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method=['coh'],
                                                                                           sfreq=int(Fs),
                                                                                           mode='cwt_morlet',
                                                                                           cwt_freqs=freqs,
                                                                                           cwt_n_cycles=n_cycles,
                                                                                           verbose=True)
            if experiment == 'Experiment Stage A+BCondition 0s':
                coh_0s.append(con[1,0,:,:])
            if experiment == 'Experiment Stage A+BCondition .083s':
                coh_pt083s.append(con[1,0,:,:])
            if experiment == 'Experiment Stage A+BCondition .0083s':
                coh_pt0083s.append(con[1,0,:,:])

    coh_0s_arr = np.asarray(coh_0s)
    coh_pt083s_arr = np.asarray(coh_pt083s)
    coh_pt0083s_arr = np.asarray(coh_pt0083s)
    # plt.figure()
    # plt.plot(np.squeeze(np.mean(coh_0s_arr,(0,1))))
    # plt.plot(np.squeeze(np.mean(coh_pt083s_arr,(0,1))))
    # plt.plot(np.squeeze(np.mean(coh_pt0083s_arr,(0,1))))
    # plt.legend(['0s','.083s','.0083s'])

    plt.subplot(1,3,1)
    plt.imshow(np.median(coh_0s_arr[:,1,0,:,:],0), extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower')
    plt.title('0s')
    plt.subplot(1, 3, 2)
    plt.imshow(np.median(coh_pt0083s_arr[:,1,0,:,:],0), extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower')
    plt.title('.0083s')
    plt.subplot(1, 3, 3)
    plt.imshow(np.median(coh_pt083s_arr[:,1,0,:,:],0), extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower')
    plt.title('.083s')

    print('done')
    #quiver_plot_from_excel(excel, "F12", 30*Hz)
