from ceed.analysis import CeedDataReader
import mne
import numpy as np
import h5py
import elephant
import neo
import quantities as pq

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\\'
fname = 'slice5_merged.h5'
rec_fname = '2021-05-28T14-14-07McsRecording'

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

# import scipy
# scipy.io.savemat(r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\Analysis\slice5_whitenoise_expdata.mat', exp_dict)

spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
all_spikes = []
with h5py.File(spyk_f, "r") as f:
    # List all groups
    for key in f['spiketimes'].keys():
        all_spikes.append(np.asarray(f['spiketimes'][key]))

neo_st = []
for unit in all_spikes:
    unitST = unit / Fs
    tstop = exp_dict['Start_Time'][-1] + 10
    if unitST[-1] > tstop:
        tstop = unitST[-1] + 2
    neo_st.append(neo.core.SpikeTrain(unitST, units=pq.s, t_start=0 * pq.s, t_stop=tstop))

bsts = []
for unit in neo_st:
    unit = unit.time_slice(0 * pq.s, (exp_dict['Start_Time'][-1] + 10) * pq.s)
    bsts.append(np.squeeze(elephant.conversion.BinnedSpikeTrain(unit, bin_size=(1/120)*pq.s).to_array()))

bsts = np.asarray(bsts)
np.save(r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\Analysis\slice5_whitenoise_spikedata.npy', bsts)