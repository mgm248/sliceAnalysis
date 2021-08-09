"""Amplitude based rejection of segmented events"""

from ceed.analysis import CeedDataReader
import read_experiments
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt

def rej_segs_byamp(reader, amp_thresh, elecs2plot, plot=False):

    segment = (-4,5)
    Fs = 20000
    rfs = 1000
    # Init data for mne
    # Read in relevant experiment times
    my_annot = read_experiments.read_experiment_fullCosine(reader, Fs)

    info = mne.create_info(['Average'], sfreq=Fs)

    mne_data = np.empty((1, reader.electrodes_data.item()['B5'].shape[0]))
    for elec in elecs2plot:
        mne_data[0,:] += reader.electrodes_data.item()[elec]

    mne_data[0,:] = mne_data[0,:] / len(elecs2plot)

    raw = mne.io.RawArray(mne_data, info)
    raw.set_annotations(my_annot)

    #loop through events, see if average of electrodes passes threshold
    total_evt = 0
    n_rej = 0
    my_annot_new = my_annot.copy()
    for evt in range(0,len(my_annot)):
        total_evt+=1
        curr_data = mne_data[0,int((my_annot[evt]['onset']+segment[0])*rfs):int((my_annot[evt]['onset']+segment[1])*rfs)]
        if plot:
            plt.close()
            plt.figure()
            plt.plot(curr_data)
        if np.any(curr_data>amp_thresh) or np.any(curr_data<-amp_thresh):
            my_annot_new += mne.Annotations(onset=my_annot[evt]['onset'],  # always use first
                            duration=5,
                            description='bad')
            n_rej+=1
            if plot:
                plt.title('Artifact')
        else:
            if plot:
                plt.title('No Artifact')


    return my_annot_new, n_rej/total_evt

    # raw.plot(duration=10, n_channels=33, scalings='auto', title='Data')
    #
    #
    # plt.figure()
    # plt.plot(mne_data[0,0:10000])