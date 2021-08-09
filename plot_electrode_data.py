import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import h5py
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
import quantities as pq

powseg = [2, 10]
baseline=(3, 4)
segment = [2, 10]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\\'
fname = 'slice1_merged.h5'

exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')
filtered = True
filter = False
if filtered:
    lfp_data = np.load(ffolder+'filt_and_rsamp\\' + fname +'.npy',allow_pickle=True)
else:
    from ceed.analysis import CeedDataReader
    reader = CeedDataReader(ffolder+fname)
    reader.open_h5()
    reader.load_mcs_data()

elec = 'C6'
rfs = 1000
fs = 20000
plt_w_pwr = True
freqs = np.linspace(1, 200, 100)  # define frequencies of interest
n_cycles = freqs * 2
# n_cycles[np.where(n_cycles > 15)] = 15
if filter:
    lowpass_sig = butter_lowpass_filter(reader.electrodes_data[elec], fs, 200)
    bandpass_sig = iir_notch(reader.electrodes_data[elec], rfs * pq.Hz, frequency=60 * pq.Hz, quality=60.,
                                             axis=-1)
    bandpass_sig = butter_bp_filter(bandpass_sig, rfs * pq.Hz, 40, 175, order=8, axis=-1)
    highpass_sig = butter_highpass_filter(reader.electrodes_data[elec], rfs, 100)


powseg_length = powseg[1] - powseg[0]
ch_types = []
ch_types.append('eeg')
if filtered:
    info = mne.create_info(ch_names=list(map(str, np.arange(1, 2, 1))), sfreq=rfs, ch_types=ch_types)
else:
    info = mne.create_info(ch_names=list(map(str, np.arange(1, 2, 1))), sfreq=fs, ch_types=ch_types)
if not filtered:
    lfp_data = reader.electrodes_data[elec]
    rfs = 20000
for index, row in exp_df.iterrows():
    plt.close()
    plt.figure(figsize=(19, 12))
    start = row['t_start'] + segment[0]
    end = row['t_start'] + segment[1]
    if plt_w_pwr:
        plt.subplot(2,1,1)
    if filter:
        plt.subplot(3,1,1)
    if filtered:
        plt.plot(np.arange(segment[0], segment[1], 1/rfs),lfp_data.item()[elec][int(round(start*rfs,4)):int(round(end*rfs,4))])
    else:
        plt.plot(np.linspace(segment[0], segment[1], len(lfp_data[int(round(start*fs,4)):int(round(end*fs,4))])),lfp_data[int(round(start*fs,4)):int(round(end*fs,4))])

    plt.xlim(segment[0], segment[1])
    # plt.xlim(segment[0], segment[0]+.1)
    plt.title(row['substage'])
    # plt.show()
    pow_start = row['t_start'] + powseg[0]
    pow_end = row['t_start'] + powseg[1]
    seg_data = np.empty((1, 1, int(powseg_length * rfs)))
    if filtered:
        seg_data[0, :, :] = lfp_data.item()[elec][int(round(pow_start*rfs,4)):int(round(pow_end*rfs,4))]
    else:
        seg_data[0, :, :] = lfp_data[int(round(pow_start*rfs,4)):int(round(pow_end*rfs,4))]
    if plt_w_pwr:
        epochs_data = mne.EpochsArray(seg_data, info)
        power = mne.time_frequency.tfr_morlet(epochs_data, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)
        # power.apply_baseline(mode='percent', baseline=(baseline[0] - powseg[0], baseline[1] - powseg[0]))
        plt.subplot(2,1,2)
        plt.imshow(
            np.squeeze(power.data[:, :, :, int((segment[0] - powseg[0]) * 1000):int((segment[1] - powseg[0]) * 1000)]),
            extent=[segment[0], segment[1], power.freqs[0], power.freqs[-1]],
            aspect='auto', origin='lower')
        # plt.clim(-2, 25)
    # plt.show()

    if filter:
        plt.subplot(3,1,2)
        plt.plot(np.linspace(segment[0], segment[1], len(highpass_sig[int(round(start*fs,4)):int(round(end*fs,4))])),highpass_sig[int(round(start*fs,4)):int(round(end*fs,4))])
        plt.xlim(segment[0], segment[1])
        plt.title('100Hz highpass')
        plt.subplot(3,1,3)
        plt.plot(np.linspace(segment[0], segment[1], len(bandpass_sig[int(round(start*fs,4)):int(round(end*fs,4))])),bandpass_sig[int(round(start*fs,4)):int(round(end*fs,4))])
        plt.xlim(segment[0], segment[1])

    # plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\Figures\tf\slice5_merged.h5\J3\withtf\raw\\'+row['substage']+str(index))