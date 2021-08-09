from ceed.analysis import CeedDataReader
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt

fname = 'slice1_merged'
ffolder = 'D:\myRecordings\\2-18-21\\'
readinraw = True

ceed_data = ffolder+fname+'.h5'

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()
# load the mcs data into memory
if readinraw:
    reader.load_mcs_data()
    electrodes = sorted(reader.electrodes_data.keys())

# #Filter and resample data
rfs = 1000
Fs = 20000
if readinraw:
    sig_length = float(reader.electrodes_data['A4'].shape[0] / Fs)  # get signal length in seconds
    num_samples = int(sig_length * rfs)
    for elec in electrodes:
        reader.electrodes_data[elec] = butter_lowpass_filter(reader.electrodes_data[elec], Fs, 200)
        reader.electrodes_data[elec] = signal.resample(reader.electrodes_data[elec], num_samples)
        reader.electrodes_data[elec] = butter_highpass_filter(reader.electrodes_data[elec], rfs, .1)
        reader.electrodes_data[elec] = iir_notch(reader.electrodes_data[elec], rfs * Hz, frequency=60 * Hz, quality=60., axis=-1)

    np.save(ffolder+'filt_and_rsamp\\' + fname+'.npy',reader.electrodes_data)

reader.electrodes_data = np.load(ffolder+'filt_and_rsamp\\' + fname+'.npy',allow_pickle=True)
# for key in reader.electrodes_data.item().keys():
#     reader.electrodes_data.item()[key] = butter_bp_filter(reader.electrodes_data.item()[key], rfs, 50, 80, order=8, axis=-1)
#     # reader.electrodes_data.item()[elec] = butter_bp_filter(reader.electrodes_data.item()[elec], rfs, 1, 8, order=2, axis=-1)
elecs2plot = ['J9', 'C11', 'B10', 'C10', 'D10', 'K7']
info = mne.create_info(elecs2plot, sfreq=rfs)

mne_data = np.empty((len(elecs2plot),reader.electrodes_data.item()['B5'].shape[0]))
i = 0
for elec in elecs2plot:
    mne_data[i,:] = reader.electrodes_data.item()[elec]
    i += 1

init = True
for exp in range(0,len(reader.experiments_in_file)):
    reader.load_experiment(exp)
    if reader.electrode_intensity_alignment is not None:
        #find the peak of the stimulus
        for shape in reader.shapes_intensity.keys():
            peak_idxs = np.where(reader.shapes_intensity[shape][:,2] == 1)
            for i in range(0,peak_idxs[0].shape[0]):
                idx = peak_idxs[0][i]
                if not idx >= reader.electrode_intensity_alignment.shape[0]:
                    if init:
                        my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs,
                                           duration=.2,
                                           description=shape)
                        init = False
                    else:
                        try:
                            my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs,
                                               duration=.2,
                                               description=shape)
                        except:
                            print('t')


raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
raw.plot(duration=10, n_channels=len(elecs2plot), scalings='auto', title='Data')
plt.show()
