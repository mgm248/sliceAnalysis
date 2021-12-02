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


def plot_responses_timestog(ev_sr_df):
    for unit in range(0, np.max(ev_sr_df['Unit #'])):
        if np.any(ev_sr_df['Unit #'] == unit):
            plt.close()
            plt.figure(figsize=(18, 10))
            plt.plot(np.arange(segment[0], segment[1], .001),
                     np.mean(ev_sr_df[ev_sr_df['Unit #'] == unit]['event spike rate']))
            plt.savefig(
                ffolder + r'Figures\evoked_spikerates\\' + fname + '\\' + 'byneuron'+'\\' + 'Unit' + str(
                    unit))


def plot_responses_condtog(ev_sr_df, shape='sq', square_exp=False):
    if square_exp:
        ev_sr_Ast_Bmd = ev_sr_df[ev_sr_df['description'].isin(['A strong B weak ' + shape])]
        ev_sr_Amd_Bst = ev_sr_df[ev_sr_df['description'].isin(['A weak B strong ' + shape])]
    else:
        ev_sr_Ast_Bmd = ev_sr_df[ev_sr_df['description'].isin(['A strong B weak'])]
        ev_sr_Amd_Bst = ev_sr_df[ev_sr_df['description'].isin(['A weak B strong'])]
    for unit in range(0, np.max(ev_sr_Ast_Bmd['Unit #'])):
        if np.any(ev_sr_Ast_Bmd['Unit #'] == unit):
            plt.close()
            plt.figure(figsize=(18, 10))
            plt.plot(np.arange(segment[0], segment[1], .001),
                     np.mean(ev_sr_Ast_Bmd[ev_sr_Ast_Bmd['Unit #'] == unit]['event spike rate']))
            plt.plot(np.arange(segment[0], segment[1], .001),
                     np.mean(ev_sr_Amd_Bst[ev_sr_Amd_Bst['Unit #'] == unit]['event spike rate']))
            plt.legend(['A strong B medium', 'B strong A medium'])
            plt.savefig(ffolder + r'Figures\spikes\evoked_spikerates\A and B\\' + fname + '\\' + 'A+B\\' + 'Unit' + str(
                unit) + '_' + shape)

get_exp_data = False
segment = [-5, 30]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-10\\'
fnames = ['slice3_ramp_merged.h5']
rec_fnames = ['2021-11-10T15-20-02McsRecording']
get_spikerates = True
for f in range(0, len(fnames)):
    fname = fnames[f]
    rec_fname = rec_fnames[f]
    ceed_data = ffolder+fname
    Fs=20000
    reader = CeedDataReader(ceed_data)
    # open the data file
    reader.open_h5()

    if get_exp_data:
        from ceed_stimulus import get_all_exps_AB, read_exp_df_from_excel, write_exp_df_to_excel
        exp_df = get_all_exps_AB(ceed_data)
        exp_df.to_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')
        # write_exp_df_to_excel(exp_df, ffolder+'Analysis\\'+fname+'_experiment_df.xlsx', 'Sheet1')
    else:
        exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')

    if get_spikerates:
        spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
        all_spikes = []
        with h5py.File(spyk_f, "r") as f:
            # List all groups
            for key in f['spiketimes'].keys():
                all_spikes.append(np.asarray(f['spiketimes'][key]))

        all_d = []
        for unit in range(0, len(all_spikes)):
           # if cluster_info[unit + 1][5] == 'good':
            unitST = all_spikes[unit] / Fs
            neo_st = []
            tstop = exp_df['t_start'].iloc[-1]*s+10*s
            if unitST[-1] > tstop:
                tstop = unitST[-1] + 2
            neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                              t_stop=tstop))
            sp_smooth = elephant.statistics.instantaneous_rate(neo_st[0], sampling_period=1 * pq.ms,
                                                   kernel=elephant.kernels.GaussianKernel(50 * pq.ms))

            for row in range(0,exp_df.shape[0]):
                start = exp_df.iloc[row]['t_start']*s+segment[0]*s
                end = exp_df.iloc[row]['t_start']*s+segment[1]*s
                try:
                    seg_st = sp_smooth.time_slice(start, (end - .001*s) + .001 * pq.s)
                    d = {
                        'event spike rate': np.squeeze(seg_st),
                        'description': exp_df.iloc[row]['substage'],
                        'Unit #': unit, 't_start':exp_df.iloc[row]['t_start']
                    }
                    all_d.append(d)
                except ValueError:
                    print('Stimuli after recording stopped')

        ev_sr_df = pd.DataFrame(all_d)
        ev_sr_df.to_pickle(ffolder+'\Analysis\spikerates\\'+fname)
    else:
        ev_sr_df = pd.read_pickle(ffolder+'\Analysis\spikerates\\'+fname)

    import os
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname)
    except:
        print('Directory already made')
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname + '\\byneuron\\')
    except:
        print('Directory already made')

    plt.figure(figsize=(18,10))
    plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_df['event spike rate'],0))
    plt.title('Mean all')
    plt.savefig(ffolder + r'\Figures\evoked_spikerates\\' + fname+'\\mean_all')

    """Plot each neurons' response to stimuli, save"""

    plot_responses_timestog(ev_sr_df)

