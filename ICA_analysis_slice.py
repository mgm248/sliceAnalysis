from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt

layout = mne.channels.read_layout(kind='120_mea_channellocs_MNE.lay', path='C:\\Users\Michael\\Documents')
fname = '1-27-20___slice4b_merged.h5_nonotch.npy'
#ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-04-26\filt_and_rsamp\\'
ffolder = r'D:\myRecordings\Jesse_goodGammaf\filt_and_rsamp\\'
Fs = 20000
rfs = 1000
data = np.load(ffolder+ fname,allow_pickle=True)

#Config info for mne
elecs = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
info = mne.create_info(elecs, sfreq=rfs, ch_types='eeg')
mne_data = np.empty((len(elecs), data.item()['A4'].shape[0]))
i = 0
for elec in elecs:
    mne_data[i, :] = data.item()[elec]
    i += 1

layout.pos[:, 0] = layout.pos[:, 0]-.5
layout.pos[:, 1] = layout.pos[:, 1]-.5
for elec in range(0,len(info['chs'])):
    info['chs'][elec]['loc'] = np.append(layout.pos[elec,:],np.zeros(8))

raw = mne.io.RawArray(mne_data, info)

ica = mne.preprocessing.ICA(n_components=32, max_iter=200)
ica.fit(raw, verbose=True)

ica.plot_sources(raw)
ica.plot_components()