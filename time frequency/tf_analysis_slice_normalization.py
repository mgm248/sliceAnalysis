from ceed.analysis import CeedDataReader
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\\'
fname = 'slice5_merged.h5'
segment = (-1, 3)
baseline=(-.8, -.1)
Fs = 20000
rfs = 1000
get_exp_data = True
# open the data file
import os
slicedir = ffolder+'Figures\\tf\\'+fname+'\\'
try:
    os.mkdir(slicedir)
except:
    print(slicedir + ' already made')
ceed_data = ffolder+fname

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()

exp_df = pd.read_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl')
init = True
for evt in range(0, exp_df.shape[0]):
    if init:
        my_annot = mne.Annotations(onset=exp_df.iloc[evt]['t_start'],
                                   duration=.5,
                                   description=exp_df.iloc[evt]['substage'])
        init = False
    else:
        my_annot += mne.Annotations(onset=exp_df.iloc[evt]['t_start'],
                                   duration=.5,
                                   description=exp_df.iloc[evt]['substage'])

reader.electrodes_data = np.load(ffolder+'filt_and_rsamp\\' + fname +'.npy',allow_pickle=True)

# elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2',
#               'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
#               'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11',
#               'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5',
#               'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1',
#               'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5',
#               'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8',
#               'M9']
elecs2plot = ['J3'] #for testing
info = mne.create_info(elecs2plot, sfreq=rfs)

mne_data = np.empty((len(elecs2plot), reader.electrodes_data.item()['B5'].shape[0]))
i = 0
for elec in elecs2plot:
    mne_data[i, :] = reader.electrodes_data.item()[elec]
    i += 1

raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
events, event_ids = mne.events_from_annotations(raw)

# experiments = ['A strong B medium cos']
bytrial = False
for experiment in exp_df['substage'].unique():
    try:
        os.mkdir(slicedir + '\\pow' + experiment + '_perelec\\')
    except:
        print('directory already exists')
    epochs = mne.Epochs(raw, events, event_ids[experiment], tmin=segment[0], tmax=segment[1], event_repeated='drop',
                        baseline=None)

    freqs = np.linspace(1, 250, 100)  # define frequencies of interest
    # n_cycles = np.arange(1, 10, 3.)
    n_cycles = freqs / 2
    tmp_pow = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                     average=~bytrial, picks=epochs.picks)
    #  tmp_pow.apply_baseline(mode='percent', baseline=(baseline[0] - segment_size[0], baseline[1] - segment_size[0]))
    tmp_pow.apply_baseline(mode='percent', baseline=baseline)
    # tmp_pow.plot([0])
    # tmp_itc.plot([0])
    file_pow = tmp_pow.data
    plt.close()
    if bytrial:
        savedir = slicedir+'J3\\pertrial\\'+experiment+'\\'
        try:
            os.mkdir(savedir)
        except:
            print('directory already exists')
        for trial in range(0, tmp_pow.data.shape[0]):
            plt.figure(figsize=(20,12))
            plt.imshow(np.squeeze(tmp_pow.data[trial,0,:,:]),
                             extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                             aspect='auto', origin='lower', interpolation='none')
            plt.savefig(savedir+'trial'+str(trial))
    else:
        fig = plt.imshow(np.mean(tmp_pow.data, (0)),
                         extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                         aspect='auto', origin='lower', interpolation='none')
        plt.clim(0,5)
        plt.savefig(slicedir + 'meanall_pow' + experiment + '.png')
        for i in range(tmp_pow.data.shape[0]):
            plt.close()
            fig = plt.imshow(np.squeeze(tmp_pow.data[i, :, :]),
                             extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                             aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=5)
            plt.savefig(slicedir + '\\pow' + experiment + '_perelec\\' + elecs2plot[i] + '.png')



