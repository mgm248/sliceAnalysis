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

segment = [-1, 2]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\\'
fname = 'slice1_merged.h5'
rec_fname = '2021-07-21T11-44-12McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))

exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')
"""Correct the start times of each condition based off the 0's preceding the actual signal stim in exp_df"""
# row = 0
# while row < exp_df.shape[0]-10:
#     if exp_df['substage'][row]=='A medium cos':
#         n_off = 0
#         idx = 1
#         while exp_df['signal A'][row][idx] == 0:
#             idx+=1
#             n_off +=1
#         time_offset = n_off*(1/120)
#     for row in range(row, row+10):
#         exp_df.loc[row, 't_start'] = exp_df['t_start'][row] + time_offset*s

ceed_data = ffolder+fname
Fs=20000
reader = CeedDataReader(ceed_data)
# open the data file
reader.open_h5()
all_d = []
for unit in range(0, len(all_spikes)):
   # if cluster_info[unit + 1][5] == 'good':
    unitST = all_spikes[unit] / Fs
    tstop = exp_df['t_start'].iloc[-1]*s+10*s
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                      t_stop=tstop)

    for row in range(0,exp_df.shape[0]):
        start = exp_df.iloc[row]['t_start']*s+segment[0]*s
        end = exp_df.iloc[row]['t_start']*s+segment[1]*s
        try:
            seg_st = neo_st.time_slice(start, (end - .001*s) + .001 * pq.s)
            d = {
                'event spike rate': np.squeeze(seg_st),
                'description': exp_df.iloc[row]['substage'],
                'Unit #': unit,
                'Event time': float(exp_df.iloc[row]['t_start'] / s)
            }
            all_d.append(d)
        except ValueError:
            print('Stimuli after recording stopped')

ev_sr_df = pd.DataFrame(all_d)

def compare_conditions(comparison, unit, shape):
    comparisons = {'delays': ['A delay B', 'B delay A'], 'varyA': ['A strong '+shape, 'B medium '+shape, 'A strong B medium '+shape, 'A weak B medium '+shape],
                   'varyB': ['A medium '+shape, 'B strong '+shape, 'A medium B strong '+shape, 'A medium B weak '+shape]}
    sep_conds = comparisons[comparison]
    plt.figure(figsize=(15, 10))
    for i, condition in enumerate(sep_conds):
        curr_df = ev_sr_df[ev_sr_df['description'] == condition]
        plt.subplot(len(sep_conds),1,i+1)
        unit_i = 0
        for trial in curr_df['Event time'].unique():
            trial_df = curr_df[curr_df['Event time'] == trial]
            unit_df = trial_df[trial_df['Unit #'] == unit]
            plt.plot(unit_df['event spike rate'].iloc[0] - trial * s,
                     unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), 'k.', markersize=4, color='black')
            plt.xlim(-.1, .7)
            unit_i += 1
        if i < len(sep_conds)-1:
            plt.xticks([])
        plt.ylim(-.5, unit_i+.5)
        plt.axvline(0)
        plt.title(condition)
        plt.savefig(ffolder+r'Figures\spikes\evoked_spikerates\A and B\\'+fname+'\\condition_comparisons\\rasters\\Unit'+str(unit)+'_'+comparison+shape)
# cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#sep_conds = ['A delay B', 'B delay A']
#sep_conds = ['A medium sq', 'A delay B', 'B delay A']
#colors = ['blue', 'red']
responsive_units = [121, 123]
comparisons = ['delays', 'varyA', 'varyB']
for unit in responsive_units:
    for comparison in comparisons:
        compare_conditions(comparison, unit, 'cos')
        compare_conditions(comparison, unit, 'sq')
        plt.close('all')

#TODO: on raster, plot expected response (if just A + just B)
A_resp = [54]
B_resp = [25, 37, 41, 118]
both_resp = []
cm = plt.get_cmap('gist_rainbow')
import seaborn as sns
cycle = sns.color_palette('husl', n_colors=len(both_resp)) # a list of RGB tuples
plt.close('all')
shape = 'sq'
conditions = ['A strong B medium '+shape, 'A medium B strong '+shape]
plt.figure(figsize=(18, 12))

for c_i, condition in enumerate(conditions):
    # plt.close('all')
    curr_df = ev_sr_df[ev_sr_df['description'] == condition]
    unit_i = 0
    plt.subplot(1,len(conditions), c_i+1)
    for trial in curr_df['Event time'].unique():
        trial_df = curr_df[curr_df['Event time'] == trial]
        for unit in A_resp:
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike rate'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), 'k.', markersize=4, color='blue')
            unit_i+=1
        for unit in B_resp:
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike rate'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), 'k.', markersize=4, color='r')
            unit_i+=1
        for i, unit in enumerate(both_resp):
            unit_df = trial_df[trial_df['Unit #']==unit]
            plt.plot(unit_df['event spike rate'].iloc[0] - trial*s, unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), 'k.', markersize=4, color=cycle[i])
            unit_i+=1
        plt.xlim(-.05, .1)
        plt.axvline(0, color='black')
        plt.title(condition)
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='red', lw=4)]
        plt.legend(custom_lines, ['A responsive', 'B responsive'])
        # plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\Figures\spikes\evoked_spikerates\example_raster\slice4\both_responsive\\slice4_units_resp_both' + str(tdiff) + '.png')


