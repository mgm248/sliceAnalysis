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

segment = [-2, 5]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-04-26\\'
fname = 'slice8_merged.h5'
rec_fname = '2021-04-26T18-05-22McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
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
n_trials = 20

my_annot = read_experiments.read_experiment_fullCosine(reader, Fs)
all_ev_st = []
all_exp_desc = []
all_unit_n = []
all_d = []
for unit in range(0, len(all_spikes)):
   # if cluster_info[unit + 1][5] == 'good':
    unitST = all_spikes[unit] / Fs
    neo_st = []
    neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                      t_stop=unitST[-1] + 5))
    sp_smooth = elephant.statistics.instantaneous_rate(neo_st[0], sampling_period=1 * pq.ms,
                                           kernel=elephant.kernels.GaussianKernel(20 * pq.ms))

    for evt in my_annot:
        start = evt['onset']+segment[0]
        end = evt['onset']+segment[1]
        try:
            seg_st = sp_smooth.time_slice(start * pq.s, (end - .001) * pq.s + .001 * pq.s)
            all_ev_st.append(np.squeeze(seg_st))
            all_exp_desc.append(evt['description'])
            all_unit_n.append(unit)
            d = {
                'event spike rate': np.squeeze(seg_st),
                'description': evt['description'],
                'Unit #': unit
            }
            all_d.append(d)
        except ValueError:
            print('Stimuli after recording stopped')

ev_sr_df = pd.DataFrame(all_d)
ev_sr_0s = ev_sr_df[ev_sr_df['description'] == 'Experiment Stage A+BCondition 0s']
ev_sr_083s = ev_sr_df[ev_sr_df['description'] == 'Experiment Stage A+BCondition .083s']
ev_sr_0083s = ev_sr_df[ev_sr_df['description'] == 'Experiment Stage A+BCondition .0083s']

plt.figure(figsize=(18,10))
plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0s['event spike rate'],0))
plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0083s['event spike rate'],0))
plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_083s['event spike rate'],0))
plt.legend(['0s', '.0083s', '.083s'])
plt.title('Mean all')
plt.savefig(ffolder+'Figures\spikes\evoked_spikerates\\'+fname+'_meanall.png')

def plot_all_responses_bytime(ev_sr_df, subf, ffolder):
    for unit in range(0, np.max(ev_sr_df['Unit #'])):
        if np.any(ev_sr_0s['Unit #']==unit):
            plt.close()
            plt.figure()
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_df[ev_sr_df['Unit #']==unit]['event spike rate']))
            plt.savefig(ffolder+'Figures\spikes\evoked_spikerates\\'+subf+'\\'+ fname+'\\'+'Unit'+str(unit))

def plot_responses_timestog(ev_sr_0s, ev_sr_0083s, ev_sr_083s):
    for unit in range(0, np.max(ev_sr_0s['Unit #'])):
        if np.any(ev_sr_0s['Unit #']==unit):
            plt.close()
            plt.figure(figsize=(18,10))
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0s[ev_sr_0s['Unit #']==unit]['event spike rate']))
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0083s[ev_sr_0083s['Unit #']==unit]['event spike rate']))
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_083s[ev_sr_083s['Unit #']==unit]['event spike rate']))
            plt.legend(['0s', '.0083s', '.083s'])
            plt.savefig(ffolder + r'Figures\spikes\evoked_spikerates\all\\' + fname + '\\' + 'Unit' + str(unit))


import os
try:
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\'+'0s')
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\'+'083s')
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\'+'0083s')
    os.mkdir(ffolder + r'\Figures\spikes\evoked_spikerates\\' + 'all')

except:
    print('Directory already made')
try:
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\0s\\'+fname)
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\083s\\'+fname)
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\0083s\\'+fname)
    os.mkdir(ffolder+r'\Figures\spikes\evoked_spikerates\\all\\'+fname)
except:
    print('File directory already made')

plot_responses_timestog(ev_sr_0s, ev_sr_0083s, ev_sr_083s)

plot_all_responses_bytime(ev_sr_0s, '0s', ffolder)
plot_all_responses_bytime(ev_sr_0083s, '0083s', ffolder)
plot_all_responses_bytime(ev_sr_083s, '083s', ffolder)

#
# good_units = [2]
# plt.figure()
# def goodunit_df(ev_sr_df, good_units):
#     ev_sr_gu = []
#     for unit in good_units:
#         ev_sr_gu.append(np.mean(ev_sr_df[ev_sr_df['Unit #']==unit]['event spike rate']))
#     return np.asarray(ev_sr_gu)
# ev_sr_0s_gu = goodunit_df(ev_sr_0s, good_units)
# ev_sr_083s_gu = goodunit_df(ev_sr_083s, good_units)
# ev_sr_0083s_gu = goodunit_df(ev_sr_0083s, good_units)
# plt.figure()
# plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0s_gu,0))
# plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_0083s_gu,0))
# plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_083s_gu,0))

