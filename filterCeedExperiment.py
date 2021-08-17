from ceed.analysis import CeedDataReader
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt
import os

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-12\\'
try:
    os.mkdir(ffolder+'filt_and_rsamp')
except:
    print('Directory already created')
for fname in os.listdir(ffolder):
   # if 'harp' in fname:
    if '_merged' in fname:
        ceed_data = ffolder+fname

        # create instance that can load the data
        reader = CeedDataReader(ceed_data)
        print('Created reader for file {}'.format(reader.filename))
        # open the data file
        reader.open_h5()
        # load the mcs data into memory

        reader.load_mcs_data()
        electrodes = sorted(reader.electrodes_data.keys())

        # #Filter and resample data
        rfs = 1000
        Fs = 20000

        sig_length = float(reader.electrodes_data['A4'].shape[0] / Fs)  # get signal length in seconds
        num_samples = int(sig_length * rfs)
        for elec in electrodes:
            reader.electrodes_data[elec] = butter_lowpass_filter(reader.electrodes_data[elec], Fs, 200)
            reader.electrodes_data[elec] = signal.resample(reader.electrodes_data[elec], num_samples)
            reader.electrodes_data[elec] = butter_highpass_filter(reader.electrodes_data[elec], rfs, 1)
            reader.electrodes_data[elec] = iir_notch(reader.electrodes_data[elec], rfs * Hz, frequency=60 * Hz, quality=60., axis=-1)

        np.save(ffolder+'filt_and_rsamp\\' + fname+'.npy',reader.electrodes_data)
