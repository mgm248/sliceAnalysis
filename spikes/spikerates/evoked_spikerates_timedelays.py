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

segment = [-.25, .75]
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
    neo_st = []
    tstop = my_annot[-1]['onset']+10
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                      t_stop=tstop))
    sp_smooth = elephant.statistics.instantaneous_rate(neo_st[0], sampling_period=1 * pq.ms,
                                           kernel=elephant.kernels.GaussianKernel(5 * pq.ms))

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
ev_sr_Asep = ev_sr_df[ev_sr_df['description'] == 'Experiment: Whole experiment; Condition: 1200; Shape A']
ev_sr_Bsep = ev_sr_df[ev_sr_df['description'] == 'Experiment: Whole experiment; Condition: 1200; Shape B']

plt.figure(figsize=(18,10))
plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_Asep['event spike rate'],0))
plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_Bsep['event spike rate'],0))
plt.legend(['A', 'B'])
plt.title('Mean all')
plt.savefig(ffolder+'Figures\spikes\evoked_spikerates\\'+fname+'_mean_Asep_Bsep.png')

def plot_all_responses_bytime(ev_sr_df, subf, ffolder):
    for unit in range(0, np.max(ev_sr_df['Unit #'])):
        if np.any(ev_sr_0s['Unit #']==unit):
            plt.close()
            plt.figure()
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_df[ev_sr_df['Unit #']==unit]['event spike rate']))
            plt.savefig(ffolder+'Figures\spikes\evoked_spikerates\\'+subf+'\\'+ fname+'\\'+'Unit'+str(unit))

def plot_responses_timestog(ev_sr_Asep, ev_sr_Bsep):
    for unit in range(0, np.max(ev_sr_Asep['Unit #'])):
        if np.any(ev_sr_Asep['Unit #']==unit):
            plt.close()
            plt.figure(figsize=(18,10))
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_Asep[ev_sr_Asep['Unit #']==unit]['event spike rate']))
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_Bsep[ev_sr_Bsep['Unit #']==unit]['event spike rate']))
            plt.legend(['Asep','Bsep'])
            plt.savefig(ffolder + r'Figures\spikes\evoked_spikerates\A and B\\' + fname + '\\' + 'Unit' + str(unit))


import os
try:
    os.mkdir(ffolder + r'\Figures\spikes\evoked_spikerates\\' + 'A and B')

except:
    print('Directory already made')
try:
    os.mkdir(ffolder + r'\Figures\spikes\evoked_spikerates\\' + 'A and B'+ '\\' + fname)
except:
    print('File directory already made')

try:
    os.mkdir(ffolder + r'\Figures\spikes\evoked_spikerates\\' + fname)
except:
    print('File directory already made')

plot_responses_timestog(ev_sr_Asep, ev_sr_Bsep)

# conditions=[]
# for evt in my_annot:
#     condition=evt['description'].replace('; Shape A', '')
#     condition=condition.replace('; Shape B', '')
#     if not (condition in conditions):
#         conditions.append(condition)

tdiffs = np.arange(-18,18.1, 3)
responsive_units = [20,22,23,29,30,48,51,53,54,59,61,62,69,104,109,110,112,113,115,116,125,126,127]
#responsive_units = [48]
for unit in responsive_units:
# for unit in range(0, np.max(ev_sr_df['Unit #'])):
    try:
        os.mkdir(ffolder + r'\Figures\spikes\evoked_spikerates\\' + fname + '\\Unit' + str(unit))
    except:
        print('File directory already made')
    for tdiff in tdiffs:
        condition='Experiment: Whole experiment; Condition: ' + str(int(tdiff))+'; Shape A'
        curr_df = ev_sr_df[ev_sr_df['description'] == condition]
        plt.close('all')
        plt.figure(figsize=(18, 10))
        plt.plot(np.arange(segment[0], segment[1], .001),
                 np.mean(curr_df[curr_df['Unit #'] == unit]['event spike rate']))
        plt.savefig(ffolder+r'\Figures\spikes\evoked_spikerates\\' + fname + '\\Unit' + str(unit) + '\\' + str(int(tdiff)))

peak_idxs = np.where(reader.shapes_intensity['Shape-11'][:, 2] == 1)
half_cos = reader.shapes_intensity['Shape-11'][0:21,2]
time_arr = np.arange(segment[0], segment[1], .001)

Amplitude = 1
Period = .16666
Function = 0.0

Function = Amplitude * np.sin(np.pi * time_arr / Period)
Function[250:416]

def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx

"""plot with stimulus"""
# A_resp = [29, 62, 115, 127]
# B_resp = [22, 30, 51, 69, 109, 113, 116]
A_resp = [62]
B_resp = [51, 69]
ev_sr_df_resp = ev_sr_df[ev_sr_df['Unit #'].isin(A_resp)]
for tdiff in tdiffs:
    condition='Experiment: Whole experiment; Condition: ' + str(int(tdiff))+'; Shape A'
    curr_df = ev_sr_df_resp[ev_sr_df_resp['description'] == condition]
    plt.figure(figsize=(4, 4))
    plt.subplot(2,1,1)
    plt.plot(np.arange(segment[0], segment[1], .001),
             np.mean(curr_df['event spike rate']))
    plt.title(str(int(tdiff)))
    plt.ylim([0, 200])
    plt.subplot(2,1,2)
    idx = find_nearest(time_arr, 0)
    stimulis = np.zeros(len(time_arr))
    stimulis[idx-83:idx+83] += Function[250:416]
    plt.plot(time_arr,stimulis)
    idx = find_nearest(time_arr, 0+tdiff/120)
    stimulis = np.zeros(len(time_arr))
    stimulis[idx-83:idx+83] += Function[250:416]
    plt.plot(time_arr,stimulis)
    plt.legend(['A','B'])
   # plt.axvline(0)
   # plt.axvline(0+tdiff/120)
    #idx=find_nearest(time_arr, 0+tdiff)
    #plt.savefig(ffolder+r'\Figures\spikes\evoked_spikerates\\' + fname + '\\' 'B responsive\\unit\\' + str(int(tdiff)))

"""Plot with stim, per neuron"""
A_resp = [29, 62, 115, 127]
B_resp = [22, 30, 51, 69, 109, 113, 116]
for unit in B_resp:
    unit_df = ev_sr_df[ev_sr_df['Unit #']==unit]
    for tdiff in tdiffs:
        condition='Experiment: Whole experiment; Condition: ' + str(int(tdiff))+'; Shape A'
        curr_df = unit_df[unit_df['description'] == condition]
        plt.figure(figsize=(4, 4))
        plt.subplot(2,1,1)
        plt.plot(np.arange(segment[0], segment[1], .001),
                 np.mean(curr_df['event spike rate']))
        plt.title(str(int(tdiff)))
        plt.ylim([0, 50])
        plt.subplot(2,1,2)
        idx = find_nearest(time_arr, 0)
        stimulis = np.zeros(len(time_arr))
        stimulis[idx-83:idx+83] += Function[250:416]
        plt.plot(time_arr,stimulis)
        idx = find_nearest(time_arr, 0+tdiff/120)
        stimulis = np.zeros(len(time_arr))
        stimulis[idx-83:idx+83] += Function[250:416]
        plt.plot(time_arr,stimulis)
        plt.legend(['A','B'])
       # plt.axvline(0)
       # plt.axvline(0+tdiff/120)
        #idx=find_nearest(time_arr, 0+tdiff)
        plt.savefig(ffolder+r'\Figures\spikes\evoked_spikerates\\' + fname + '\\' 'B responsive\\unit\\' +'Unit' + str(unit)+ ' Tdiff' + str(int(tdiff)))


tdiffs = np.arange(-18,18.1, 3)
def init():
    diff = 0
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    condition = 'Experiment: Whole experiment; Condition: ' + '0' + '; Shape A'
    curr_df = ev_sr_df[ev_sr_df['description'] == condition]
    sr = ax1.plot(np.arange(segment[0], segment[1], .001),
                  np.mean(curr_df['event spike rate']))
    ax2 = fig.add_subplot(212)
    idx = find_nearest(time_arr, 0 + tdiff / 120)
    stimulis = np.zeros(len(time_arr))
    stimulis[idx - 83:idx + 83] += Function[250:416]
    stima = ax2.plot(time_arr, stimulis)
    idx = find_nearest(time_arr, 0 + tdiff / 120)
    stimulis = np.zeros(len(time_arr))
    stimulis[idx - 83:idx + 83] += Function[250:416]
    stimb = ax2.plot(time_arr, stimulis)


def animate(i):
    tdiffs = np.arange(0, 18.1, 3)
    condition='Experiment: Whole experiment; Condition: ' + str(int(tdiffs[i]))+'; Shape A'
    curr_df = ev_sr_df[ev_sr_df['description'] == condition]
    sr.set_data(time_arr,np.mean(curr_df['event spike rate']))
    idx = find_nearest(time_arr, 0 + tdiffs[i] / 120)
    stimulis = np.zeros(len(time_arr))
    stimulis[idx - 83:idx + 83] += Function[250:416]
    stima.set_data(time_arr, stimulis)
    #rline.set_data((0, all_uv_ang[i]), (0, all_uv_rad[i]))


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
condition = 'Experiment: Whole experiment; Condition: ' + '0' +'; Shape A'
curr_df = ev_sr_df[ev_sr_df['description'] == condition]
sr, = ax1.plot(np.arange(segment[0], segment[1], .001),
         np.mean(curr_df['event spike rate']))
ax2 = fig.add_subplot(212)
idx = find_nearest(time_arr, 0 + tdiff / 120)
stimulis = np.zeros(len(time_arr))
stimulis[idx - 83:idx + 83] += Function[250:416]
stima, = ax2.plot(time_arr, stimulis)
idx = find_nearest(time_arr, 0 + tdiff / 120)
stimulis = np.zeros(len(time_arr))
stimulis[idx - 83:idx + 83] += Function[250:416]
stimb, = ax2.plot(time_arr, stimulis)

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, interval=20, frames=len(tdiffs))



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

