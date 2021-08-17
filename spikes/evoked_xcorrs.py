import h5py
import numpy as np
import matplotlib.pyplot as plt
import more_spike_lfp_scripts
import neo
import math
from scipy.stats import circmean
import scipy.stats as stats
from scipy.signal import hilbert, resample
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import csv
import read_experiments
import elephant
import quantities as pq
import pandas as pd
import pickle

segment = [-1, 6]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\\'
fname = 'slice7_merged.h5'
rec_fname = '2021-08-04T14-52-44McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
circus_df = pickle.load(open(ffolder+'Analysis\\'+rec_fname+'.pkl', "rb"))
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))


# clusterinfo_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.GUI\cluster_info.tsv'
# tsv_file = open(clusterinfo_f)
# read_tsv = csv.reader(tsv_file,delimiter="\t")
# cluster_info = []
# for row in read_tsv:
#     cluster_info.append(row)
ceed_data = ffolder+fname
Fs=20000
reader = CeedDataReader(ceed_data)
# open the data file
reader.open_h5()

# my_annot = read_experiments.read_experiment_halfcos_timedelays(reader, Fs)
all_ev_st = []
all_exp_desc = []
all_unit_n = []
all_d = []
sts = []
exp_df = pd.read_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl')

for unit in range(0, len(all_spikes)):
   # if cluster_info[unit + 1][5] == 'good':
    unitST = all_spikes[unit] / Fs
    tstop = float(exp_df.iloc[exp_df.shape[0] - 1]['t_start'])+10
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s, t_stop=tstop*s)
    sts.append(neo_st)

for i in range(0, exp_df.shape[0]):
    start = exp_df.iloc[i]['t_start']+segment[0]
    end = exp_df.iloc[i]['t_start']+segment[1]
    # seg_st = neo_st.time_slice(start * pq.s, (end - .001) * pq.s + .001 * pq.s)
    # all_ev_st.append(np.squeeze(seg_st))
    all_unit_n.append(unit)
    d = {
        # 'event spike response': np.squeeze(seg_st),
        'description': exp_df.iloc[i]['substage'], 'Event time': exp_df.iloc[i]['t_start']
    }
    all_d.append(d)

ev_sr_df = pd.DataFrame(all_d)

A_resp = [62]
B_resp = [51, 69]
both_resp = [20, 23, 30, 48, 53, 54, 59, 61, 69, 104, 110]
units = [20, 53, 48, 110]

def pairwise_odor_autoch_grid(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='Experiment: Whole experiment; Condition: 0; Shape A', window=.1, bin=.001, dither=False):
    #sts is a list of spiketrains
    exp_df = ev_sr_df[ev_sr_df['description']==event]
    if abs(offset) > abs(duration):
        color = "gray"
    counter = 1
    for i, unit1 in enumerate(units):

        all_odor_st1, all_odor_st2 = [], []
        st1 = sts[unit1]
        st2 = sts[unit1]
        for _, experiment in exp_df.iterrows():
            t_start = experiment['Event time']
            t_start = float(t_start) * s

            odor_st1 = st1.time_slice(t_start+offset, t_start+offset+duration)
            odor_st2 = st2.time_slice(t_start+offset, t_start+offset+duration)
            odor_st1 = [t.item() for t in odor_st1]
            odor_st2 = [t.item() for t in odor_st2]
            all_odor_st1 += odor_st1
            all_odor_st2 += odor_st2

        plt.subplot(len(units), len(units), counter)
        low, high = -window, window
        histo_window = (low, high)
        histo_bins = int((high - low) / bin)

        all_odor_diffs = []
        for spike in all_odor_st1:
            window_start, window_stop = spike - window, spike + window
            in_window = [x for x in all_odor_st2 if x >= window_start and x<= window_stop]
            for spike_time in in_window:
                spike_time = (spike_time - spike)
                all_odor_diffs.append(spike_time)
        counts, bins = np.histogram(all_odor_diffs, bins=histo_bins, range=histo_window)
        plt.bar(bins[101:-1], counts[101:], width=bin, linewidth=1.2, edgecolor='k', align='edge')
        # plt.bar(bins[:-1], counts, width=bin, linewidth=1.2, edgecolor='k', align='edge')
        plt.title(str(i))
        # plt.yticks([])
        # plt.xticks([-50, 0, 50])
        # plt.xlim([-.100, .100])
        plt.grid(True, axis='x')
        counter += 1
    fig = plt.gcf()
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.tight_layout()

    # plt.show()

def pairwise_odor_autoch_grid_pertrial(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='Experiment: Whole experiment; Condition: 0; Shape A', window=.1, bin=.001, dither=False):
    #sts is a list of spiketrains
    exp_df = ev_sr_df[ev_sr_df['description']==event]
    if abs(offset) > abs(duration):
        color = "gray"
    counter = 1
    for i, unit1 in enumerate(units):

        st1 = sts[unit1]
        st2 = sts[unit1]
        for t, experiment in exp_df.iterrows():
            plt.figure()
            t_start = experiment['Event time']
            t_start = float(t_start) * s
            all_odor_st1, all_odor_st2 = [], []
            odor_st1 = st1.time_slice(t_start+offset, t_start+offset+duration)
            odor_st2 = st2.time_slice(t_start+offset, t_start+offset+duration)
            odor_st1 = [t.item() for t in odor_st1]
            odor_st2 = [t.item() for t in odor_st2]
            all_odor_st1 += odor_st1
            all_odor_st2 += odor_st2

            plt.subplot(len(units), len(units), counter)
            low, high = -window, window
            histo_window = (low, high)
            histo_bins = int((high - low) / bin)

            all_odor_diffs = []
            for spike in all_odor_st1:
                window_start, window_stop = spike - window, spike + window
                in_window = [x for x in all_odor_st2 if x >= window_start and x<= window_stop]
                for spike_time in in_window:
                    spike_time = (spike_time - spike)
                    all_odor_diffs.append(spike_time)
            counts, bins = np.histogram(all_odor_diffs, bins=histo_bins, range=histo_window)
            plt.bar(bins[:-1], counts, width=bin, linewidth=1.2, edgecolor='k', align='edge')
            plt.title(str(i))
            plt.yticks([])
            # plt.xticks([-50, 0, 50])
            # plt.xlim([-.100, .100])
            plt.grid(True, axis='x')
            fig = plt.gcf()
            fig = plt.gcf()
            fig.set_size_inches(6, 6)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\xcorrs\recording6\evoked_autocorrs\\'+str(t))
            plt.close()
            # plt.show()

def pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='A strong sq', window=.1, bin=.001, evoked=False):
    #sts is a list of spiketrains
    if evoked:
        exp_df = ev_sr_df[ev_sr_df['description']==event]
        if abs(offset) > abs(duration):
            color = "gray"
    counter = 1
    for i, unit1 in enumerate(units):
        for j, unit2 in enumerate(units):

            all_odor_st1, all_odor_st2 = [], []
            st1 = sts[unit1]
            st2 = sts[unit2]
            if evoked:
                for _, experiment in exp_df.iterrows():
                    t_start = experiment['Event time']
                    t_start = float(t_start) * s

                    odor_st1 = st1.time_slice(t_start+offset, t_start+offset+duration)
                    odor_st2 = st2.time_slice(t_start+offset, t_start+offset+duration)
                    odor_st1 = [t.item() for t in odor_st1]
                    odor_st2 = [t.item() for t in odor_st2]
                    all_odor_st1 += odor_st1
                    all_odor_st2 += odor_st2
            else:
                odor_st1 = [t.item() for t in st1]
                odor_st2 = [t.item() for t in st2]
                all_odor_st1 += odor_st1
                all_odor_st2 += odor_st2

            plt.subplot(len(units), len(units), counter)
            low, high = -window, window
            histo_window = (low, high)
            histo_bins = int((high - low) / bin)

            all_odor_diffs = []
            for spike in all_odor_st1:
                window_start, window_stop = spike - window, spike + window
                in_window = [x for x in all_odor_st2 if x >= window_start and x<= window_stop]
                for spike_time in in_window:
                    spike_time = (spike_time - spike)
                    all_odor_diffs.append(spike_time)
            counts, bins = np.histogram(all_odor_diffs, bins=histo_bins, range=histo_window)
            plt.bar(bins[:-1], counts, width=bin, linewidth=1.2, edgecolor='k', align='edge')
            plt.title(str(i) + "-" + str(j))
            plt.yticks([])
            # plt.xticks([-50, 0, 50])
            # plt.xlim([-.100, .100])
            plt.grid(True, axis='x')
            counter += 1
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.tight_layout()
    # plt.show()

"""Evoked: segment by time around stimuli; false to just do overall xcorr"""
def xcorr(exp_df, st1, st2, offset, duration, window, bin, evoked=False):
    all_odor_st1, all_odor_st2 = [], []
    if evoked:
        for _, experiment in exp_df.iterrows():
            t_start = experiment['Event time']
            t_start = float(t_start) * s

            odor_st1 = st1.time_slice(t_start + offset, t_start + offset + duration)
            odor_st2 = st2.time_slice(t_start + offset, t_start + offset + duration)
            odor_st1 = [t.item() for t in odor_st1]
            odor_st2 = [t.item() for t in odor_st2]
            all_odor_st1 += odor_st1
            all_odor_st2 += odor_st2
    else:
        odor_st1 = [t.item() for t in st1]
        odor_st2 = [t.item() for t in st2]
        all_odor_st1 += odor_st1
        all_odor_st2 += odor_st2

    low, high = -window, window
    histo_window = (low, high)
    histo_bins = int((high - low) / bin)

    all_odor_diffs = []
    for spike in all_odor_st1:
        window_start, window_stop = spike - window, spike + window
        in_window = [x for x in all_odor_st2 if x >= window_start and x <= window_stop]
        for spike_time in in_window:
            spike_time = (spike_time - spike)
            all_odor_diffs.append(spike_time)
    counts, bins = np.histogram(all_odor_diffs, bins=histo_bins, range=histo_window)
    return counts, bins

def surr_pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='Experiment: Whole experiment; Condition: 0; Shape A', window=100, bin=.005, n_surr=200, dither=np.array(15.) * ms):
    #sts is a list of spiketrains
    exp_df = ev_sr_df[ev_sr_df['description']==event]
    if abs(offset) > abs(duration):
        color = "gray"
    counter = 1
    for i, unit1 in enumerate(units):
        for j, unit2 in enumerate(units):

            all_odor_st1, all_odor_st2 = [], []
            st1 = sts[unit1]
            st2 = sts[unit2]
            real_count, bins = xcorr(exp_df, st1, st2, offset, duration, window, bin)

            st2 = elephant.spike_train_surrogates.JointISI(st2, dither=dither).dithering(n_surr)
            surr_counts = []
            for surr_st in st2:
                counts, bins = xcorr(exp_df, st1, surr_st, offset, duration, window, bin)
                surr_counts.append(counts)

            plt.subplot(len(units), len(units), counter)
            plt.bar(bins[:-1], real_count - np.mean(np.asarray(surr_counts),0), width=bin, linewidth=1.2, edgecolor='k', align='edge')
            plt.title(str(i) + "-" + str(j))
            plt.yticks([])
            # plt.xticks([-50, 0, 50])
            # plt.xlim([-.100, .100])
            plt.grid(True, axis='x')
            counter += 1
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.tight_layout()
    plt.show()

def ISIs_per_spike(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='A strong sq', window=.1, bin=.001):
    exp_df = ev_sr_df[ev_sr_df['description']==event]
    for i, unit1 in enumerate(units):

            st1 = sts[unit1]
            ISIs_overall = []
            for _, experiment in exp_df.iterrows():
                t_start = experiment['Event time']
                t_start = float(t_start) * s
                odor_st1 = st1.time_slice(t_start+offset, t_start+offset+duration)
                ISIs = []
                for spi in range(0,len(odor_st1)):
                    if spi > 0:
                        ISIs.append(odor_st1[spi] - odor_st1[spi-1])
                ISIs_overall.append(ISIs)
    return ISIs_overall

# units = [7, 71]
# # units=[75, 67]
# """autocorrs, Xcorrs evoked by stimulus"""
# for i, unit in enumerate(sts):
#     # plt.figure()
#     pairwise_odor_autoch_grid(ev_sr_df, sts, [i], duration=.5 * s, offset=0 * s, event='B medium sq', window=.1, bin=.001)
#     plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\Figures\spikes\xcorrs\evoked\B medium sq\\'+str(i))
#     plt.close()

"""evoked autocorr for a single unit"""
# for unit in range(0, len(sts)):
#     pairwise_odor_autoch_grid(ev_sr_df, sts, [unit], duration= 1.3 * s, offset=.2 * s, event='1s stim const', window=.1, bin=.001)
#     plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\xcorrs\recording6\evoked_autocorrs\\Unit'+str(unit))
#     plt.close()
pairwise_odor_autoch_grid_pertrial(ev_sr_df, sts, [unit], duration= 1.3 * s, offset=.2 * s, event='1s stim const', window=.1, bin=.001)
#
# """grid of xcorrs"""
# units = [32, 33, 38]
# pairwise_odor_cch_grid([], sts, units, duration=[], offset=[.2*s], event='1s stim const', window=.1, bin=.002, evoked=False)
# plt.show()

unit = 32
isis = ISIs_per_spike(ev_sr_df, sts, [unit], duration= 1 * s, offset=0 * s, event='1s stim const', window=.1, bin=.001)
for i, isi in enumerate(isis):
    plt.figure()
    plt.plot(np.arange(1, len(isi)+1, 1), isi)
    plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\ISIs\unit32_pertrial\\'+str(i))
    plt.close()

# units = [121, 100, 46, 38, 35, 17]
# units = [46, 35]
# pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=5*s, offset=0*s, event=2, window=100, bin=.001)
#
# units = [46, 35]
# surr_pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=5*s, offset=0*s, event=2, window=100, bin=.001, n_surr=50, dither=np.array(25.) * ms)
# pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=.166*s, offset=-.083*s, event='Experiment: Whole experiment; Condition: 3; Shape A', window=100, bin=.01)
#
# plt.figure()
#
# pairwise_odor_cch_grid(ev_sr_df, sts, units, duration=5*s, offset=-6*s, event='Experiment: Whole experiment; Condition: 0; Shape A', window=100, bin=.005)

"""Autocorrs, Xcorrs, over recording period"""
# import os
# binsize=.001
# ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\\'
# Fs = 20000
# slicen = 1
# for file in os.listdir(ffolder):
#     if 'McsRecording.h5' in file:
#         try:
#             os.mkdir(r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\xcorrs\\recording'+str(slicen))
#         except:
#             print('dir already made')
#         savedir = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-04\Figures\spikes\xcorrs\\recording'+str(slicen)+'\\evoked_autocorrs\\'
#         try:
#             os.mkdir(savedir)
#         except:
#             print('dir already made')
#         spyk_f = ffolder + 'Analysis\\spyking-circus\\' + file.replace('.h5','') + '\\' + file.replace('.h5','') + 'times.result.hdf5'
#         all_spikes = []
#         with h5py.File(spyk_f, "r") as f:
#             # List all groups
#             for key in f['spiketimes'].keys():
#                 all_spikes.append(np.asarray(f['spiketimes'][key]))
#
#         sts = []
#         for unit in range(0, len(all_spikes)):
#             # if cluster_info[unit + 1][5] == 'good':
#             unitST = all_spikes[unit] / Fs
#             # tstop = float(exp_df.iloc[exp_df.shape[0] - 1]['t_start']) + 10
#             # if unitST[-1] > tstop:
#             if unitST.shape[0]>0:
#                 tstop = unitST[-1] + 2
#                 neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s, t_stop=tstop * s)
#                 sts.append(neo_st)
#
#         for i, unit in enumerate(sts):
#             counts, bins = xcorr([], st1=unit, st2=unit, offset=[], duration=[], window=.1, bin=binsize, evoked=False)
#             plt.figure()
#             # plt.bar(x=bins[101:-1],
#             #         height=counts[101:],
#             #         align='edge')
#             plt.bar(bins[101:-1], counts[101:], width=binsize, linewidth=1.2, edgecolor='k', align='edge')
#             plt.savefig(savedir+str(i))
#             plt.close()
#         slicen+=1


#
"""Do xcorrs of all neurons in comparison to 1"""
# unit1 = 7
# for i, unit in enumerate(sts):
#     counts, bins = xcorr([], st1=sts[unit1], st2=unit, offset=[], duration=[], window=.1, bin=binsize, evoked=False)
#     plt.figure()
#     plt.bar(x=bins[0:-1],
#             height=counts,
#             width=binsize,
#             align='edge')
#     plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\Figures\spikes\xcorrs\\slice5\\compw7\\'+str(i))
#     plt.close()


