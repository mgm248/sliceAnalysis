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

segment = [-.25, .25]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\\'
fname = 'slice4_merged.h5'
rec_fname = '2021-05-28T13-43-43McsRecording'
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

my_annot = read_experiments.read_experiment_halfcos_timedelays(reader, Fs)
all_ev_st = []
all_exp_desc = []
all_unit_n = []
all_d = []
for unit in range(0, len(all_spikes)):
   # if cluster_info[unit + 1][5] == 'good':
    unitST = all_spikes[unit] / Fs
    tstop = my_annot[-1]['onset']+10
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s, t_stop=tstop)

    for evt in my_annot:
        start = evt['onset']+segment[0]
        end = evt['onset']+segment[1]
        try:
            seg_st = neo_st.time_slice(start * pq.s, (end - .001) * pq.s + .001 * pq.s)
            all_ev_st.append(np.squeeze(seg_st))
            all_exp_desc.append(evt['description'])
            all_unit_n.append(unit)
            d = {
                'event spike response': np.squeeze(seg_st),
                'description': evt['description'],
                'Unit #': unit, 'Event time': evt['onset'],
            }
            all_d.append(d)
        except ValueError:
            print('Stimuli after recording stopped')

ev_sr_df = pd.DataFrame(all_d)

# A_resp = [62]
# B_resp = [51, 69]
A_resp = []
B_resp = []
both_resp = [20, 23, 30, 48, 53, 54, 59, 61, 69, 104, 110, 112, 116, 125]
cm = plt.get_cmap('gist_rainbow')
import seaborn as sns
cycle = sns.color_palette('husl', n_colors=len(both_resp)) # a list of RGB tuples

# cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#TODO: on raster, plot expected response (if just A + just B)
plt.close('all')
times = np.arange(-18, 18.01, 3)
for tdiff in times:
    # plt.close('all')
    condition = 'Experiment: Whole experiment; Condition: ' + str(int(tdiff))+'; Shape A'
    curr_df = ev_sr_df[ev_sr_df['description'] == condition]
    plt.figure(figsize=(15,10))
    unit_i = 0
    for trial in curr_df['Event time'].unique():
        trial_df = curr_df[curr_df['Event time'] == trial]
        for unit in A_resp:
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike response'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike response'].iloc[0]), 'k.', markersize=4, color='blue')
            plt.xlim(-.25, .25)
            unit_i+=1
        for unit in B_resp:
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike response'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike response'].iloc[0]), 'k.', markersize=4, color='r')
            plt.xlim(-.25, .25)
            unit_i+=1
        for i, unit in enumerate(both_resp):
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike response'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike response'].iloc[0]), 'k.', markersize=4, color=cycle[i])
            plt.xlim(-.25, .25)
            unit_i+=1
        plt.axvline(0, color='blue')
        plt.axvline(0 + tdiff/120, color='red')
        plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\Figures\spikes\evoked_spikerates\example_raster\slice4\both_responsive\\slice4_units_resp_both' + str(tdiff) + '.png')


sep_conds = ['Experiment: Whole experiment; Condition: 1200; Shape A', 'Experiment: Whole experiment; Condition: 1200; Shape B']
colors = ['blue', 'red']
for i, condition in enumerate(sep_conds):
    curr_df = ev_sr_df[ev_sr_df['description'] == condition]
    plt.figure(figsize=(15,10))
    unit_i = 0
    for trial in curr_df['Event time'].unique():
        trial_df = curr_df[curr_df['Event time'] == trial]
        for j, unit in enumerate(both_resp):
            unit_df = trial_df[trial_df['Unit #'] == unit]
            plt.plot(unit_df['event spike response'].iloc[0] - trial * s,
                     unit_i * np.ones_like(unit_df['event spike response'].iloc[0]), 'k.', markersize=4, color=cycle[j])
            plt.xlim(-.25, .25)
            unit_i += 1
    plt.title(condition)
    plt.axvline(0, color=colors[i])