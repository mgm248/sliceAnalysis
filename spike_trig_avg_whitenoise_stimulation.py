from ceed.analysis import CeedDataReader
import mne
import numpy as np
import h5py
import neo
import quantities as pq
import matplotlib.pyplot as plt

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\\'
fname = 'slice5_merged.h5'
rec_fname = '2021-05-28T14-14-07McsRecording'
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))


ceed_data = ffolder+fname
Fs=20000
reader = CeedDataReader(ceed_data)
reader.open_h5()
for exp in range(0,len(reader.experiments_in_file)):
    reader.load_experiment(exp)
    print(reader.experiment_stage_name)
    if reader.electrode_intensity_alignment is not None:
        start_times = []
        exp_int_data = []
        trial_ns = []
        trial_n = 0
        i = 0
        while i < reader.shapes_intensity['enclosed'].shape[0]:
            if not reader.shapes_intensity['enclosed'][i,2] == 0:
                start_times.append(reader.electrode_intensity_alignment[i] / Fs)
                trial_ns.append(trial_n)
                start_i = i
                while not reader.shapes_intensity['enclosed'][i,2] == 0:
                    i+=1
                end_i = i
                trial_n += 1
                exp_int_data.append(reader.shapes_intensity['enclosed'][start_i:end_i, 2])
                print(str(len(reader.shapes_intensity['enclosed'][start_i:end_i, 2])))
            i+=1

    exp_dict = {'Start_Time': start_times, 'Trial_number': trial_ns, 'Intensity': exp_int_data}

neo_st = []
for unit in all_spikes:
    unitST = unit / Fs
    tstop = exp_dict['Start_Time'][-1] + 10
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st.append(neo.core.SpikeTrain(unitST, units=pq.s, t_start=0 * pq.s, t_stop=tstop))

segment = [.31, 4.59]
seg_win = .3
seglength_i = int(seg_win * 120)*2
all_sta = []
for curr_st in neo_st:
    currsp_sta = []
    for trial in exp_dict['Trial_number']:
        start = exp_dict['Start_Time'][trial] + segment[0]
        end = exp_dict['Start_Time'][trial] + segment[1]

        seg_st = curr_st.time_slice(start * pq.s, (end - .001) * pq.s + .001 * pq.s)
        for spike in seg_st:
            stim_i = int(round((spike / pq.s - exp_dict['Start_Time'][trial]) * 120)) #120 is our sampling rate
            if not exp_dict['Intensity'][trial][stim_i - int(seg_win * 120):stim_i + int(seg_win*120)].shape[0] == seglength_i:
                print('uh oh')
            currsp_sta.append(exp_dict['Intensity'][trial][stim_i - int(seg_win * 120):stim_i + int(seg_win*120)])
            # if stim_i > 200:
            #     plt.figure()
            #     plt.plot(np.linspace(-seg_win*1000, seg_win*1000, seglength_i), exp_dict['Intensity'][trial][stim_i - int(seg_win * 120):stim_i + int(seg_win*120)])

    all_sta.append(currsp_sta)

total_sta = np.zeros((seglength_i))
n_tot_sta = 0
for i, curr_sp_sta in enumerate(all_sta):
    plt.close()
    plt.figure()
    for sta in curr_sp_sta:
        total_sta += sta
        n_tot_sta += 1
        if i==1:
            plt.close()
            plt.figure()
            plt.plot(np.linspace(-100, 100, seglength_i), sta)
    curr_sp_sta = np.asarray(curr_sp_sta)
    if curr_sp_sta.size > 0:
        plt.plot(np.linspace(-100, 100, seglength_i), np.mean(curr_sp_sta,0))
        plt.title(str(i))
        plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\Figures\whitenoise\perunit\slice4\\' + 'Unit' + str(i))


plt.figure()
plt.plot(np.linspace(-seg_win*1000, seg_win*1000, seglength_i), total_sta / n_tot_sta)
plt.ylabel('Stimulation intensity')
plt.xlabel('Time from spike')