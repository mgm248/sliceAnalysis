from ceed.analysis import CeedDataReader

import read_experiments
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-10\\'
segment = (-4,8)
baseline=(-3, -1)
Fs = 20000
rfs = 1000
import os
for fname in os.listdir(ffolder):
    if 'slice5_merged' in fname:
   # if '_merged' in fname:
        slicedir = ffolder + 'Figures\\tf\\'+fname+'\\'
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

        reader.electrodes_data= np.load(ffolder+'filt_and_rsamp\\' + fname +'.npy',allow_pickle=True)
        bad_channels = ['A7', 'B7', 'C11', 'C7', 'D1', 'D10', 'E1', 'E8', 'F10', 'F6', 'F9', 'G11', 'G12', 'G7', 'H10', 'H9', 'J7', 'J8', 'L5', 'L7', 'L9', 'M5', 'M8']
        for elec in bad_channels:
            del reader.electrodes_data.item()[elec]

        elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
        elecs2plot = [x for x in elecs2plot if x not in bad_channels]
        # Init data for mne
        # Read in relevant experiment times
        my_annot = read_experiments.read_experiment_fullCosine(reader, Fs)

        """reject bad trials"""
        bad_trials = []
        from reject_segments_amplitude import rej_segs_byamp

        my_annot, perc_rej = rej_segs_byamp(reader, amp_thresh=400, elecs2plot=elecs2plot, plot=True)
        print(str(perc_rej) + ' rejected')

        info = mne.create_info(elecs2plot, sfreq=rfs)


        mne_data = np.empty((len(elecs2plot), reader.electrodes_data.item()['B5'].shape[0]))
        i = 0
        for elec in elecs2plot:
            mne_data[i, :] = reader.electrodes_data.item()[elec]
            i += 1

        raw = mne.io.RawArray(mne_data, info)
        raw.set_annotations(my_annot)
        events, event_ids = mne.events_from_annotations(raw)

       # experiments = ['Experiment Stage ABCTimesep0s', 'Experiment Stage ABCTimesep.5s', 'Experiment Stage ABCTimesep1.5s']
        experiments = list(event_ids.keys())
        for experiment in experiments:
            try:
                os.mkdir(slicedir + '\\pow' + experiment.replace(' ', '') + '_perelec\\')
            except:
                print('directory already exists')
            epochs = mne.Epochs(raw, events, event_ids[experiment], tmin=segment[0], tmax=segment[1], event_repeated='drop', baseline=None, reject_by_annotation=True)
            epochs.drop_bad()

            freqs = np.linspace(1, 100, 100)  # define frequencies of interest
            # n_cycles = np.arange(1, 10, 3.)
            n_cycles = freqs
            #tmp_pow, tmp_itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True, average=True, picks=epochs.picks)
            tmp_pow = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, picks=epochs.picks)
            #  tmp_pow.apply_baseline(mode='percent', baseline=(baseline[0] - segment_size[0], baseline[1] - segment_size[0]))
            tmp_pow.apply_baseline(mode='percent', baseline=baseline)
            # tmp_pow.plot([0])
            # tmp_itc.plot([0])
            file_pow = tmp_pow.data

            # for evt in range(tmp_pow.data.shape[0]):
            #     # plt.figure()
            #     # fig = plt.imshow(np.mean(tmp_pow.data[evt,:,:,:], (0)),
            #     #                  extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
            #     #                  aspect='auto', origin='lower', interpolation='none', vmin=-.5, vmax=.7)
            #     # plt.title('Trial ' + str(evt))
            #     rej_trial = False
            #     for time in range(tmp_pow.data.shape[3]):
            #         if time > (segment[0]+1.5)*1000 and time < (segment[1]-1.5)*1000: #don't care about edge artifacts
            #             if np.count_nonzero(np.mean(tmp_pow.data[evt,:,:,time], 0) > .5) > 50: #if power for every frequency is above n for a given timepoint
            #                 rej_trial=True
            #     if rej_trial:
            #         bad_trials.append(evt)

            #check if working by plotting bad trials only
            # tmp_pow.data[bad_trials, :, :, :] = np.nan
            plt.close()
            np.save(ffolder+r'Analysis\tf\\'+fname+experiment, tmp_pow.data)
            fig = plt.imshow(np.nanmean(tmp_pow.data,(0,1)),
                             extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                             aspect='auto', origin='lower', interpolation='none', vmin=-.1, vmax=.5)
            plt.savefig(slicedir + 'meanall_pow'+experiment+'.png')



            for i in range(tmp_pow.data.shape[1]):
                plt.close()
                fig = plt.imshow(np.nanmean(tmp_pow.data[:,i,:,:], 0),
                                 extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                                 aspect='auto', origin='lower', interpolation='none', vmin=-.75, vmax=1.5)
               # plt.savefig(slicedir + '\\pow' + experiment.replace(' ','') + '_perelec\\' + elecs2plot[i] + '.png')
                plt.savefig(slicedir + '\\pow' + experiment.replace(' ', '') + '_perelec\\' + elecs2plot[i] + '.png')

