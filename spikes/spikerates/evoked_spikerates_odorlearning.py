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


def plot_responses_timestog(ev_sr_df, drug):

    for unit in range(0, np.max(ev_sr_df['Unit #'])):
        if np.any(ev_sr_df['Unit #'] == unit):
            plt.close()
            plt.figure(figsize=(18, 10))
            ITIs = ['60.0', '15', '5.0', '2.0', '0.5']
            for ITI in ITIs:
                curr_df = ev_sr_df[ev_sr_df['description'] == drug + 'ITI ' + str(ITI) + 's']
                plt.plot(np.arange(segment[0], segment[1], .001),
                         np.mean(curr_df[curr_df['Unit #'] == unit]['event spike rate']))

            plt.legend(ITIs)
            plt.savefig(
                ffolder + r'Figures\evoked_spikerates\\' + fname + '\\' + 'byneuron'+drug.replace(' ','')+'\\' + 'Unit' + str(
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
segment = [-1, 1]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\\'
fnames = ['slice1_merged.h5']
rec_fnames = ['2021-11-17T11-38-14McsRecording']
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
        # old: get spiking results from raw files:
        # spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
        # all_spikes = []
        # with h5py.File(spyk_f, "r") as f:
        #     # List all groups
        #     for key in f['spiketimes'].keys():
        #         all_spikes.append(np.asarray(f['spiketimes'][key]))
        #
        all_d = []
        # for unit in range(0, len(all_spikes)):
        # Get spiking results from spyking dataframe:
        sk_df = pd.read_pickle(ffolder+'Analysis\\spyking-circus\\' + rec_fname+'.pkl')
        for i, row in sk_df.iterrows():
            if not row['Group'] == 'noise':
                #unitST = all_spikes[unit] / Fs
                unitST = row['Data']# / Fs
                neo_st = []
                tstop = exp_df['t_start'].iloc[-1]*s+10*s
                if unitST[-1] > tstop:
                    tstop = unitST[-1] + 2
                neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                                  t_stop=tstop))
                sp_smooth = elephant.statistics.instantaneous_rate(neo_st[0], sampling_period=1 * pq.ms,
                                                       kernel=elephant.kernels.GaussianKernel(10 * pq.ms))

                for row_i in range(0,exp_df.shape[0]):
                    start = exp_df.iloc[row_i]['t_start']*s+segment[0]*s
                    end = exp_df.iloc[row_i]['t_start']*s+segment[1]*s
                    try:
                        seg_st = sp_smooth.time_slice(start, (end - .001*s) + .001 * pq.s)
                        d = {
                            'event spike rate': np.squeeze(seg_st),
                            'description': exp_df.iloc[row_i]['substage'],
                            'Unit #': row['ID'], 't_start':exp_df.iloc[row_i]['t_start']
                        }
                        all_d.append(d)
                    except ValueError:
                        print('Stimuli after recording stopped')

        ev_sr_df = pd.DataFrame(all_d)
        try:
            os.mkdir(ffolder+'\Analysis\spikerates')
        except:
            print('directory already made')
        ev_sr_df.to_pickle(ffolder+'\Analysis\spikerates\\'+fname)
    else:
        ev_sr_df = pd.read_pickle(ffolder+'\Analysis\spikerates\\'+fname)

    """Plot each neurons' response to stimuli, save"""
    import os
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname)
    except:
        print('Directory already made')
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname + '\\byneuron\\')
    except:
        print('Directory already made')
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname + '\\byneuronNE\\')
    except:
        print('Directory already made')
    try:
        os.mkdir(ffolder + r'\Figures\evoked_spikerates\\' + fname + '\\byneuronWashout\\')
    except:
        print('Directory already made')

    plot_responses_timestog(ev_sr_df, '')
    plot_responses_timestog(ev_sr_df, 'NE ')
    plot_responses_timestog(ev_sr_df, 'Washout ')

    """For each neuron, compare its response in each condition"""
    per_trial = False
    units = [16]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for unit in units:
        unit_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
        fig, axs = plt.subplots(1, 3, sharey=True)
        conditions = ['', 'NE ']
        ymax = 0
        for ci, condition in enumerate(conditions):
            plt.subplot(1, 3, ci + 1)
            ITIs = ['60.0', '15', '5.0', '2.0', '0.5']
            for itii, ITI in enumerate(ITIs):
                curr_df = unit_df[unit_df['description'] == condition + 'ITI ' + str(ITI) + 's']
                if not per_trial:
                    sr = np.mean(curr_df['event spike rate'], 0)
                    axs[ci].plot(np.arange(segment[0], segment[1], .001), sr)
                else:
                    for i, row in curr_df.iterrows():
                        axs[ci].plot(np.arange(segment[0], segment[1], .001), row['event spike rate'], color=colors[itii], alpha=.2, label=condition)
                # if np.max(sr) > ymax:
                #     ymax = np.max(sr)
            plt.title(condition)
            plt.xlim(0, .5)
        # for ax in axs:
        #     ax.set_ylim(-1, float(ymax) + 4)
        fig.set_size_inches(18, 8)
        # plt.savefig(
        #     r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\Figures\evoked_spikerates\slice5_merged.h5\byneuron_compareconditions\\' + str(
        #         unit))
        # plt.close()

    plt.figure(figsize=(18,10))
    plt.plot(np.arange(segment[0], segment[1], .001), np.mean(ev_sr_df['event spike rate'],0))
    plt.title('Mean all')
    plt.figure()
    conditions = ['', 'NE ', 'Washout ']
    for ci, condition in enumerate(conditions):
        plt.subplot(1,3,ci+1)
        ITIs = ['15', '5.0', '1.0', '0.5']
        for ITI in ITIs:
            curr_df = ev_sr_df[ev_sr_df['description'] == condition+'ITI '+str(ITI)+'s']
            plt.plot(np.arange(segment[0], segment[1], .001), np.mean(curr_df['event spike rate'], 0))
        plt.legend(ITIs)
        plt.title(condition)
        plt.xlim(0, .5)
        plt.ylim(0, 35)

"""For each neuron, for each ITI, compare its response at baseline with NE"""
ITIs = ['60', '30', '10', '2']
# units = [3, 13, 80, 81]
units = [81]
for unit in units:
    plt.figure(figsize=(18,10))
    for i, ITI in enumerate(ITIs):
        unit_df = ev_sr_Asep[ev_sr_Asep['Unit #']==unit]
        mean_rate = np.mean(unit_df[unit_df['description'] == 'A repeat ITI ' + ITI + 's']['event spike rate'], 0)
        plt.plot(mean_rate, label='A bsln '+str(ITI)+'s', color='blue', alpha=1.2-(i+1)*.2)
        unit_df = ev_sr_Asep_NE[ev_sr_Asep_NE['Unit #']==unit]
        mean_rate = np.mean(unit_df[unit_df['description'] == 'NE A repeat ITI ' + ITI + 's']['event spike rate'], 0)
        plt.plot(mean_rate, label='A NE '+str(ITI)+'s', color='red', alpha=1.2-(i+1)*.2)
    plt.legend()
    plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-01\Figures\evoked_spikerates\A and B\slice4_merged.h5\baseline_v_NE\\'+str(unit)+'_A_byITI')
    plt.close()

"""For a neuron, compare its response for each ITI"""
units = []
ITIs = ['60', '30', '10', '2']
baseline = [-1, 0]
times = np.arange(segment[0], segment[1], .001)
baseline_norm = False
for unit in units:
    unit_df = ev_sr_df[ev_sr_df['Unit #']==unit]
    plt.figure(figsize=(18,10))
    for ITI in ITIs:
        mean_rate = np.mean(unit_df[unit_df['description'] == 'A repeat ITI '+ITI+'s']['event spike rate'], 0)
        if baseline_norm:
            mean_rate = mean_rate / np.mean(mean_rate[np.where(np.logical_and(times>=baseline[0], times<=baseline[1]))])
        plt.plot(times,mean_rate)
    plt.legend(ITIs)
    plt.title(unit)
    # plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-01\Figures\evoked_spikerates\A and B\slice4_merged.h5\byITI\NE\\'+str(unit)+'_notnorm')
    # plt.close()

"""Plot responses by trial # -- A"""
unit = 81
unit_df = ev_sr_df[ev_sr_df['Unit #']==unit]
baseline_norm = False
baseline = [-1, 0]
times = np.arange(segment[0], segment[1], .001)
NE_vals = [True, False]
for NE in NE_vals:
    for ITI in ITIs:
        if NE:
            curr_df = unit_df[unit_df['description']=='NE A repeat ITI '+ITI+'s']
        else:
            curr_df = unit_df[unit_df['description']=='A repeat ITI '+ITI+'s']
        plt.figure(figsize=(18, 10))
        curr_df['t_start'].unique()
        print(curr_df['t_start'].unique())
        for ti, t_start in enumerate(curr_df['t_start'].unique()):
            t_df = curr_df[curr_df['t_start']==t_start]
            mean_rate = np.mean(t_df['event spike rate'], 0)
            if baseline_norm:
                mean_rate = mean_rate / np.mean(mean_rate[np.where(np.logical_and(times>=baseline[0], times<=baseline[1]))])
            plt.plot(times,mean_rate, color='black', alpha=1.1-(ti+1)*.1, label=ti+1)
        plt.legend()
        plt.title(ITI+'s ITI')
        if NE:
            plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-01\Figures\evoked_spikerates\A and B\slice4_merged.h5\overtrials\\'+str(unit)+'_A_'+str(ITI)+'s_NE.png')
        else:
            plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-01\Figures\evoked_spikerates\A and B\slice4_merged.h5\overtrials\\'+str(unit)+'_A_'+str(ITI)+'s.png')
        plt.close()

"""Plot responses by trial # -- B"""
unit = 112
unit_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
baseline_norm = True
baseline = [-1, 0]
times = np.arange(segment[0], segment[1], .001)
curr_df = unit_df[unit_df['description'] == 'B']
plt.figure(figsize=(18, 10))
curr_df['t_start'].unique()
print(curr_df['t_start'].unique())
for ti, t_start in enumerate(curr_df['t_start'].unique()):
    t_df = curr_df[curr_df['t_start'] == t_start]
    mean_rate = np.mean(t_df['event spike rate'], 0)
    if baseline_norm:
        mean_rate = mean_rate / np.mean(
            mean_rate[np.where(np.logical_and(times >= baseline[0], times <= baseline[1]))])
    plt.plot(times, mean_rate, color='black', alpha=1.2 - (ti + 1) * .2, label=ti + 1)
plt.legend()

