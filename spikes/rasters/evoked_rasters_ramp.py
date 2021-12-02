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

segment = [25, 27]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-10\\'
fname = 'slice1_ramp_merged.h5'
rec_fname = '2021-11-10T11-47-47McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))

exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')

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
unit=77
plt.figure(figsize=(15, 10))
unit_i = 0
for trial in ev_sr_df['Event time'].unique():
    trial_df = ev_sr_df[ev_sr_df['Event time'] == trial]
    unit_df = trial_df[trial_df['Unit #'] == unit]
    plt.plot(unit_df['event spike rate'].iloc[0] - trial * s,
             unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), '|', markersize=12, color='black')
    unit_i+=1

def compare_conditions(comparison, unit, shape):
    comparisons = {'delays': ['A delay B', 'B delay A'], 'varyAB': ['A weak '+shape, 'A strong '+shape, 'B weak '+shape, 'B strong '+shape,  'A strong B weak '+shape, 'A weak B strong '+shape],
                   'ramps': ['ramp to .5 A', 'ramp to .5 B', 'A strong B weak ramp', 'A weak B strong ramp']}
    sep_conds = comparisons[comparison]
    plt.figure(figsize=(15, 10))
    for i, condition in enumerate(sep_conds):
        print(condition)
        curr_df = ev_sr_df[ev_sr_df['description'] == condition]
        print(curr_df['description'].unique())
        plt.subplot(len(sep_conds),1,i+1)
        unit_i = 0
        for trial in curr_df['Event time'].unique():
            trial_df = curr_df[curr_df['Event time'] == trial]
            unit_df = trial_df[trial_df['Unit #'] == unit]
            plt.plot(unit_df['event spike rate'].iloc[0] - trial * s,
                     unit_i * np.ones_like(unit_df['event spike rate'].iloc[0]), 'k.', markersize=4)
            if comparison=='ramps':
                plt.xlim(-.5, 2)
            else:
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
responsive_units = [35, 36, 53, 90, 91, 100, 101, 109, 119]
comparisons = ['delays', 'varyAB', 'ramps']
for unit in responsive_units:
    for comparison in comparisons:
        # compare_conditions(comparison, unit, 'cos')
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


