import numpy as np
import sys
import neo
from neo.io.nixio import NixIO
from neo.core import (
    Block, ChannelIndex, AnalogSignal, Segment, SpikeTrain)
from quantities import uV, Hz, ms, s
from tqdm import tqdm
from numpy.lib.format import open_memmap
# from pylab import *
import matplotlib.pyplot as plt
# from ephys_analysis.elephant_main import get_electrode_data, get_spikes_in_interval, spectrogram
import h5py
import pandas as pd


### This version accounts for electrode 15 being reference
old_map = {0: 12, 1: 13, 2: 14, 3: 16, 4: 17,
       5: 21, 6: 22, 7: 23, 8: 24, 9: 25, 10: 26, 11: 27, 12: 28,
       13: 31, 14: 32, 15: 33, 16: 34, 17: 35, 18: 36, 19: 37, 20: 38,
       21: 41, 22: 42, 23: 43, 24: 44, 25: 45, 26: 46, 27: 47, 28: 48,
       29: 51, 30: 52, 31: 53, 32: 54, 33: 55, 34: 56, 35: 57, 36: 58,
       37: 61, 38: 62, 39: 63, 40: 64, 41: 65, 42: 66, 43: 67, 44: 68,
       45: 71, 46: 72, 47: 73, 48: 74, 49: 75, 50: 76, 51: 77, 52: 78,
       53: 82, 54: 83, 55: 84, 56: 85, 57: 86, 58: 87}

mea120 = [
    'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
    'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
    'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
    'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12',
    'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11',
    'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10',
    'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]


def convert_mergedh5_to_np(in_filename, out_filename, chunks=1e9):
    """
        Converts an H5 file (generated from the "new rig" in Uris basement) to a new numpy file
        which Spyking Circus can use for spike sorting

        EXAMPLE:
            h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")
            circus_raw_filename = base_filename.format(wave_clus="", electrode="", ext="npy")
            convert_mergedh5_to_np(h5_filename, circus_raw_filename)
    """
    # in_file = NixIO(in_filename)
    f = h5py.File(in_filename, 'r')

    mcs_data = f["data"]["mcs_data"]["data_arrays"]
    sample_electrode = mcs_data["electrode_A4"]

    num_electrodes = 120
    num_samples = len(sample_electrode["data"])

    itemsize = np.array([0.0], dtype=np.float32).nbytes
    n_items = int(chunks // itemsize)  # num chunked samples per chan
    total_n = num_electrodes*num_samples  # num samples total

    pbar = tqdm(
        total=total_n * itemsize, file=sys.stdout, unit_scale=1, unit='bytes')

    mmap_array = open_memmap(
        out_filename, mode='w+', dtype=np.float32, shape=(num_samples, num_electrodes))

    electrode_ids = ["electrode_" + str(electrode_id) for electrode_id in mea120]
    for k, electrode_id in enumerate(electrode_ids):
        signal = mcs_data[electrode_id]["data"]
        n = len(signal)
        i = 0
        while i * n_items < n:
            items = np.array(
                signal[i * n_items:min((i + 1) * n_items, num_samples)],
                dtype=np.float32)[:]
            mmap_array[i * n_items:i * n_items + len(items), k] = items
            pbar.update(len(items) * itemsize)
            i += 1
        pbar.close()


def convert_mcsh5_to_np(in_filename, out_filename, key=mea120, channel_order="key", chunks=1e9):
    """
        Converts an H5 file (generated from the "new rig" in Uris basement) to a new numpy file
        which Spyking Circus can use for spike sorting

        EXAMPLE:
            h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")
            circus_raw_filename = base_filename.format(wave_clus="", electrode="", ext="npy")
            dump_electrode_data_circus(h5_filename, circus_raw_filename)
    """
    f = h5py.File(in_filename, 'r')
    mcs_data = f["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]

    channel_info = f["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["InfoChannel"]
    channel_ids = [0] * len(channel_info)
    for i, channel in enumerate(channel_info):
        channel_ids[i] = channel['Label'].decode('utf-8')

    sample_electrode = mcs_data[0]
    num_electrodes = 120
    num_samples = len(sample_electrode)

    itemsize = np.array([0.0], dtype=np.float32).nbytes
    n_items = int(chunks // itemsize)  # num chunked samples per chan
    total_n = num_electrodes*num_samples  # num samples total

    pbar = tqdm(
        total=total_n * itemsize, file=sys.stdout, unit_scale=1, unit='bytes')

    mmap_array = open_memmap(
        out_filename, mode='w+', dtype=np.float32, shape=(num_samples, num_electrodes))

    for k, electrode in enumerate(key):
        channel_index = channel_ids.index(electrode)
        # fs = (10. ** 6 * Hz) / float(channel_info[channel_index]['Tick'])
        offset = channel_info[channel_index]['ADZero']
        scale = float(channel_info[channel_index]['ConversionFactor']) * 10 ** float(
            channel_info[channel_index]['Exponent'])
        signal = mcs_data[channel_index]
        signal = (np.array(signal) - offset) * scale
        signal = signal * 2000000
        n = len(signal)
        i = 0
        while i * n_items < n:
            items = np.array(
                signal[i * n_items:min((i + 1) * n_items, num_samples)],
                dtype=np.float32)[:]
            mmap_array[i * n_items:i * n_items + len(items), k] = items
            pbar.update(len(items) * itemsize)
            i += 1
        pbar.close()

def dump_old_electrode_data_circus(in_filename, out_filename, chunks=1e9):
    """
        Converts an H5 file (generated from the CPL's 'old' 60 electrode MEAs - e.g. Shane's data) to a new numpy file
        which Spyking Circus can use for spike sorting

        EXAMPLE:
            h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")
            circus_raw_filename = base_filename.format(wave_clus="", electrode="", ext="npy")
            dump_electrode_data_circus(h5_filename, circus_raw_filename)
    """
    in_file = NixIO(in_filename)
    block = in_file.read_all_blocks()[0]
    signals = block.segments[2].analogsignals
    names = [signal.annotations['nix_name'] for signal in signals]

    indices = sorted(range(len(signals)), key=lambda x: names[x])
    signals = [signals[i] for i in indices]
    names = [names[i] for i in indices]

    itemsize = np.array([0.0], dtype=np.float32).nbytes

    n = len(signals[0])  # num samples per channel
    n_items = int(chunks // itemsize)  # num chunked samples per chan
    total_n = sum(len(value) for value in signals)  # num samples total
    pbar = tqdm(
        total=total_n * itemsize, file=sys.stdout, unit_scale=1, unit='bytes')

    mmap_array = open_memmap(
        out_filename, mode='w+', dtype=np.float32, shape=(n, len(signals)))

    for k, signal in enumerate(signals):
        i = 0
        n = len(signal)

        while i * n_items < n:
            items = np.array(
                signal[i * n_items:min((i + 1) * n_items, n)],
                dtype=np.float32)[:, 0]
            mmap_array[i * n_items:i * n_items + len(items), k] = items
            pbar.update(len(items) * itemsize)
            i += 1
    pbar.close()
    print('Channel order is: {}'.format(names))


def get_spikes_by_cluster(gui_base, cluster_num, fs, t_stop):
    period = (1./fs).rescale(ms)
    times = np.load(gui_base.format(file="spike_times", ext="npy"))
    clusters = np.load(gui_base.format(file="spike_clusters", ext="npy"))
    cluster_indices = np.where(clusters == cluster_num)
    spikes = times[cluster_indices]
    spikes_converted = [period * x for x in spikes]
    spikes_converted = np.array(spikes_converted)
    spikes_converted.reshape(spikes_converted.shape[0], )
    spikes_converted = SpikeTrain(spikes_converted*ms, t_stop)
    return spikes_converted


def get_circus_auto_result_DF(result_base, gui_base, fs=10000*Hz, base=None):

    #Grab t_stop and fs from h5 metadata, if possible
    if base is not None:
        h5_filename = base.format(wave_clus="", electrode="", ext="h5")
        raw_signal = get_electrode_data(h5_filename, "12") #arbitrary electrode selected to grab metadata
        fs = raw_signal.sampling_rate
    period = (1./fs).rescale(ms)

    results_filename = result_base.format(file='result')
    results_file = h5py.File(results_filename, 'r')
    clusters_filename = result_base.format(file='clusters')
    clusters_file = h5py.File(clusters_filename, 'r')

    electrodes = np.array(clusters_file.get('electrodes'))
    channel_map = np.load(gui_base.format(file="channel_map", ext="npy"))
    electrodes = [channel_map[electrode] for electrode in electrodes]
    circus_df = pd.DataFrame({"ID": range(len(electrodes)), "Electrode": electrodes, "Data": np.nan})
    circus_df = circus_df.astype(object)
    pbar = tqdm(total=len(circus_df["ID"]), file=sys.stdout, desc="Getting SpykingCircus results")
    for id_ in circus_df["ID"]:
        temp_id = "temp_" + str(id_)
        id_st = list(results_file["spiketimes"][temp_id])
        id_st = [period * sample for sample in id_st]
        id_st = np.array(id_st)
        id_st.reshape(id_st.shape[0],)
        circus_df.loc[id_, 'Data'] = id_st
        pbar.update(1)
    pbar.close()
    return circus_df


def get_circus_manual_result_DF(gui_base, get_electrodes=True, get_groups=False, fs=20000*Hz):
    """Can get the electrode each neuron came from"""
    period = (1./fs)#.rescale(ms)
    times = np.load(gui_base.format(file="spike_times", ext="npy"))
    clusters = np.load(gui_base.format(file="spike_clusters", ext="npy"))
    cluster_ids = np.unique(clusters)

    if get_electrodes:
        clusters_h5 = gui_base.split('.')[0] + ".clusters.hdf5"
        clusters_h5 = h5py.File(clusters_h5, 'r')
        electrodes = list(clusters_h5.get('electrodes'))

    circus_df = pd.DataFrame({"ID": cluster_ids, "Electrode": np.nan, "Electrode #": np.nan, "Group":np.nan, "Data": np.nan})
    circus_df = circus_df.astype(object)
    pbar = tqdm(circus_df.iterrows(), total=circus_df.shape[0], file=sys.stdout, desc="Getting SpykingCircus results")

    i = 0
    for index, cluster in pbar:
        id_ = cluster["ID"]
        cluster_indices = np.where(clusters == id_)
        id_st = times[cluster_indices]
        id_st = [period * x for x in id_st]
        id_st = np.array(id_st)
        id_st.reshape(id_st.shape[0], )
        cluster["Data"] = id_st

        if get_electrodes:
            channel_map = np.load(gui_base.format(file="channel_map", ext="npy"))
            # if id_ in range(0, len(electrodes)):
            #     electrode_index = electrodes[id_]
            #     cluster["Electrode"] = channel_map[electrode_index]
            electrode = electrodes[i]
            cluster["Electrode #"] = electrode
            cluster["Electrode"] = mea120[electrode]
        i += 1
    pbar.close()

    if get_groups:
        csv_file = open(gui_base.format(file="cluster_group", ext="tsv"))
        cluster_groups = pd.read_csv(csv_file, sep='\t')['group']
        circus_df['Group'] = cluster_groups
    return circus_df


def filter_out_DF_by_group(circus_df, group_bad_list=['noise']):
    filtered_circus_df = circus_df[~circus_df.Group.isin(group_bad_list)]
    print("Removed clusters in group: " + str(group_bad_list))
    return filtered_circus_df


def filter_DF_by_group(circus_df, group_good_list=['good']):
    filtered_circus_df = circus_df[circus_df.Group.isin(group_good_list)]
    print("Filtered SpykingCircus Dataframe for clusters in group: " + str(group_good_list))
    return filtered_circus_df


def fix_electrode_ids(circus_df, electrode_map=old_map):
    new_electrodes = list(map(electrode_map.get, circus_df["Electrode"]))
    fixed_circus_df = circus_df
    fixed_circus_df["Electrode"] = new_electrodes
    print("Electrode IDs fixed!")
    return fixed_circus_df


def filter_DF(circus_df, electrode_list=None, cluster_list=None):
    if electrode_list is not None:
        filtered_circus_df = circus_df[circus_df.Electrode.isin(electrode_list)]
        print("Filtered SpykingCircus Dataframe for electrodes: " + str(electrode_list))
    if cluster_list is not None:
        filtered_circus_df = circus_df[circus_df.ID.isin(cluster_list)]
        print("Filtered SpykingCircus Dataframe for clusters: " + str(cluster_list))
    return filtered_circus_df


def filter_out_DF(circus_df, electrode_bad_list=None, cluster_bad_list=None):
    if electrode_bad_list is not None:
        filtered_circus_df = circus_df[~circus_df.Electrode.isin(electrode_bad_list)]
        print("Removed electrodes: " + str(electrode_bad_list))
    if cluster_bad_list is not None:
        filtered_circus_df = circus_df[~circus_df.ID.isin(cluster_bad_list)]
        print("Removed clusters: " + str(cluster_bad_list))
    return filtered_circus_df


def lump_spikes_from_DF(circus_DF, base=None, t_stop=None):
    if base is None and t_stop is None:
        raise ValueError(
            "Please supply a value for either the base or t_stop parameter")
    if base is not None:
        h5_filename = base.format(wave_clus="", electrode="", ext="h5")
        raw_signal = get_electrode_data(h5_filename, "12") #arbitrary electrode selected to grab metadata
        t_stop = raw_signal.t_stop
    all_sts = circus_DF["Data"]
    all_sts_array = all_sts.as_matrix()
    all_sts_array = np.concatenate(all_sts_array, axis=0)
    all_sts = SpikeTrain(all_sts_array*ms, t_stop=t_stop)
    return all_sts


#convert_mcsh5_to_np(r'C:\Users\Michael\Analysis\myRecordings_extra\21-06-17\', r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-28\\2021-05-28T11-31-01McsRecording.npy', key=mea120, channel_order="key", chunks=1e9)



if "___main___":
    """Convert data for spike sorting"""
    # import os
    # fdir = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\\'
    # for file in os.listdir(fdir):
    #     # if 'Recording.h5' in file:
    #     if '2021-11-17T15-52-00McsRecording.h5' in file:
    #         convert_mcsh5_to_np(
    #             fdir+file,
    #             fdir+'Analysis\spyking-circus\\'+file.replace('h5','npy'), key=mea120,
    #             channel_order="key", chunks=1e9)

    """get spyking-circus df"""
    ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\\'
    rec_fname = '2021-11-17T11-38-14McsRecording'
    circus_df = get_circus_manual_result_DF(ffolder+'Analysis\\spyking-circus\\'+rec_fname+'\\'+rec_fname+'times.GUI\\{file}.{ext}', get_electrodes=True, get_groups=True, fs=20000*Hz)
    circus_df.to_pickle(ffolder+'Analysis\spyking-circus\\'+rec_fname+'.pkl')
    circus_df.to_csv(ffolder+'Analysis\spyking-circus\\'+rec_fname+'.csv')