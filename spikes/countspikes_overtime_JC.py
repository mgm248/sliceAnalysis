# from circus.shared.files import load_data
# from circus.shared.parser import CircusParser
# from pylab import *
# params    = CircusParser('D:\myRecordings\\1_22_20\spyking-circus\mydata.npy') #maybe should be parameter file?
# results = load_data(params, 'results')

import h5py
import numpy as np
import matplotlib as plt
import more_spike_lfp_scripts
import neo
import math
from scipy.stats import circmean
import scipy.stats as stats
from scipy.signal import hilbert, resample
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import csv


"""Read in spikes, along with info on whether they were marked good or not"""
ffolder = 'D:\myRecordings\\1_22_20\\'
fname = 'slice3_2_1_22_21_merged2.h5'
rec_fname = '2021-01-22T18-19-45McsRecording'
spyk_f = ffolder+'spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))

clusterinfo_f = ffolder+'spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.GUI\cluster_info.tsv'
tsv_file = open(clusterinfo_f)
read_tsv = csv.reader(tsv_file,delimiter="\t")
cluster_info = []
for row in read_tsv:
    cluster_info.append(row)
ceed_data = ffolder+fname

data_saveloc = 'D:\myRecordings\Analysis\spike_analysis\spikecounts_overtime_1_22_20_arange0_8_pt5_twpt5\\' #specify a folder where you'll save
Fs=20000 #I just hardcode to avoid having to load the file data through ceed
reader = CeedDataReader(ceed_data)
# open the data file
reader.open_h5()
n_trials = 20
t_step = .5
tlist = np.arange(-8, 8, t_step)

"""Most of this is figuring out when relevant trials are"""
for exp in range(0, len(reader.experiments_in_file)):
    reader.load_experiment(exp)
    if reader.experiment_stage_name == 'Stage ABC':
    # if reader.experiment_stage_name == 'enclosed blue':

        if reader.electrode_intensity_alignment is not None:
            spike_counts = np.empty((len(all_spikes), 3, n_trials, tlist.shape[0])) #where 3 is the number of conditions
            spike_counts[:] = np.nan
            ic0 = -1
            ic1 = -1
            ic2 = -1
            # find the peak of the stimulus

            peak_idxs_b = np.where(reader.shapes_intensity['B'][:, 2] == 1)
            peak_idxs_c = np.where(reader.shapes_intensity['C'][:, 2] == 1)
            for i in range(0, peak_idxs_c[0].shape[0]):
                idx = peak_idxs_c[0][i]
                if idx == peak_idxs_b[0][i]:
                    condition = 0
                    ic0+=1
                    c_idx = ic0
                if (idx - peak_idxs_b[0][i]) / 120 == 0.5:
                    condition = 1
                    ic1+=1
                    c_idx = ic1
                if (idx - peak_idxs_b[0][i]) / 120 == 1.5:
                    condition = 2
                    ic2+=1
                    c_idx = ic2
                if not idx >= reader.electrode_intensity_alignment.shape[0]:
                    t_start = reader.electrode_intensity_alignment[idx] / Fs
                    #That section above is pretty specific to me -- just know I loop through each possible trial, and if it's relevant, I mark when it starts (t_start)
                    #and what condition index it is
                    for unit in range(0, len(all_spikes)):

                        if cluster_info[unit+1][5]=='good':
                            unitST = all_spikes[unit] / Fs
                            neo_st = []
                            neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                                              t_stop=unitST[-1] + 5))
                            for t in range(0,tlist.shape[0]):
                                segment = [tlist[t]-.5,tlist[t]+.5]
                                spike_counts[unit, condition, c_idx, t], x\
                                    = more_spike_lfp_scripts.plv_overtrials_euc([], neo_st, fs=Fs,
                                              segmentTimes=[t_start[0]], segment=segment, justCount=True, countByPhase=False,
                                                                            returnPhase=False, prefPhase=200)

                np.save(data_saveloc + fname.replace('.h5',''), spike_counts)


"""Do for enclosed, a good test that things are working ok"""
# for exp in range(0, len(reader.experiments_in_file)):
#     reader.load_experiment(exp)
#     # if reader.experiment_stage_name == 'Stage ABC':
#     if reader.experiment_stage_name == 'enclosed blue':
#
#         if reader.electrode_intensity_alignment is not None:
#             spike_counts = np.empty((len(all_spikes), 3, n_trials, tlist.shape[0]))
#             spike_counts[:] = np.nan
#             ic0 = -1
#             ic1 = -1
#             ic2 = -1
#             # find the peak of the stimulus
#
#             peak_idxs_enclosed = np.where(reader.shapes_intensity['enclosed'][:, 2] == 1)
#             idx = peak_idxs_enclosed[0][0]
#             condition = 1
#             if not idx >= reader.electrode_intensity_alignment.shape[0]:
#                 t_start = reader.electrode_intensity_alignment[idx] / Fs
#
#                 for unit in range(0, len(all_spikes)):
#
#                     if cluster_info[unit+1][5]=='good':
#                         unitST = all_spikes[unit] / Fs
#                         neo_st = []
#                         neo_st.append(neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
#                                                           t_stop=unitST[-1] + 5))
#                         for t in range(0,tlist.shape[0]):
#                             segment = [tlist[t]-.5,tlist[t]+.5]
#                             spike_counts[unit, condition, 0, t], x\
#                                 = more_spike_lfp_scripts.plv_overtrials_euc([], neo_st, fs=Fs,
#                                           segmentTimes=[t_start[0]], segment=segment, justCount=True, countByPhase=False,
#                                                                             returnPhase=False, prefPhase=200)
#                 np.save(data_saveloc + fname.replace('.h5',''), spike_counts)