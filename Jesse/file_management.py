import elephant
import neo
from scipy import signal
from quantities import ms, Hz, uV, s, V
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neo.io.nixio import NixIO
from neo.core import AnalogSignal, SpikeTrain, IrregularlySampledSignal
import h5py
from itertools import groupby
from ceed.analysis import CeedDataReader
from ephys_analysis.ceed_scripts.ceed_stimulus import get_all_exps, write_exp_df_to_excel
import sys
import pandas as pd


def fix_h5_alignment_odor_series(h5_file, exp_df, drop_exps=None, wait_time=1*s, stage="Odor series", fs=20000*Hz,
                                 substage_durations=[5, 25, 5, 25, 5, 25, 5],
                                 intensities=[34.6875, 0, 34.6875, 0, 69.375, 0, 69.375]):
    """
        Aligns CEED data with MCS data from the merged h5 file when the merge script fails to align properly.

        EXAMPLE:
            h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")

    """

    period = (1/fs).rescale(s).item()
    wait_time = wait_time.rescale(s).item()

    f = h5py.File(h5_file, 'r')
    dig_io = f["data"]["mcs_data"]["data_arrays"]["digital_io"]["data"].value

    stim = np.where(dig_io != 0)[0]  # When (in samples) the projector is projecting some light pattern

    stim_delta_samples = np.diff(stim)
    non_contig = np.where(stim_delta_samples > 1)  # When (in samples) the projected pattern changes, by index of stim/stim_delta

    stim = stim[non_contig].tolist()
    stim_times = [x * period for x in stim]
    stim_times = np.array(stim_times)
    stim_delta_times = np.diff(stim_times)

    deltas_over_wait = np.where(stim_delta_times > wait_time)
    stim = np.array(stim)
    times_over_wait = stim_times[deltas_over_wait]

    stage_df = exp_df[exp_df['stage']==stage]
    stage_exps = stage_df['experiment'].unique().tolist()

    stage_exps = [int(stage_exp) for stage_exp in stage_exps]
    stage_exps.sort()
    stage_exps = [str(stage_exp) for stage_exp in stage_exps]

    if len(stage_exps) != len(times_over_wait):
        times_over_wait = times_over_wait[(len(times_over_wait)-len(stage_exps)):]


    for i, exp in enumerate(stage_exps):
        i, exp = int(i), int(exp)
        no_times = exp_df.index[(exp_df['experiment'] == exp) & (pd.isna(exp_df['t_start']))].tolist()
        print(no_times)
        if len(no_times) == 0:
            continue
        else:
            t_start = times_over_wait[i] * s
            substage_counter = 0
            for no_time in no_times:
                exp_df.loc[no_time, 't_start'] = t_start
                exp_df.loc[no_time, 't_stop'] = t_start + substage_durations[substage_counter] * s
                exp_df.loc[no_time, 'intensity'] = intensities[substage_counter]
                t_start = t_start + substage_durations[substage_counter] * s
                substage_counter += 1
    return exp_df


def fix_h5_alignment_single_stimuli(h5_file, exp_df, drop_exps=None, wait_time=1*s, stage="Odor A, strong", fs=20000*Hz,
                                    stimulus_duration=5*s, intensity=69.375):
    """
        Aligns CEED data with MCS data from the merged h5 file when the merge script fails to align properly.

        EXAMPLE:
            h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")

    """
    if drop_exps is not None:
        for exp in drop_exps:
            exp_df = exp_df.drop(exp)

    period = (1/fs).rescale(s).item()
    wait_time = wait_time.rescale(s).item()

    f = h5py.File(h5_file, 'r')
    dig_io = f["data"]["mcs_data"]["data_arrays"]["digital_io"]["data"].value

    stim = np.where(dig_io != 0)[0]  # When (in samples) the projector is projecting some light pattern

    stim_delta_samples = np.diff(stim)
    non_contig = np.where(stim_delta_samples > 1)  # When (in samples) the projected pattern changes, by index of stim/stim_delta

    stim = stim[non_contig].tolist()
    stim_times = [x * period for x in stim]
    stim_times = np.array(stim_times)
    stim_delta_times = np.diff(stim_times)

    deltas_over_wait = np.where(stim_delta_times > wait_time)
    times_over_wait = stim_times[deltas_over_wait]
    stage_df = exp_df[exp_df['stage']==stage]
    stage_exps = stage_df['experiment'].unique().tolist()

    if len(stage_exps) == 0:
        raise Exception("There are no experiments found with the given stage!")
    if len(stage_exps) != len(times_over_wait):
        raise Exception("The given stage has been found " + str(len(stage_exps)) + " times in the metadata, but has " +
                        "been detected " + str(len(times_over_wait)) + " times in the recording!")

    for i, exp in enumerate(stage_exps):
        i, exp = int(i), int(exp)
        no_times = exp_df.index[(exp_df['experiment'] == exp) & (pd.isna(exp_df['t_start']))].tolist()

        if len(no_times) == 0:
            continue
        else:
            t_start = times_over_wait[i] * s
            for no_time in no_times:
                exp_df.loc[no_time, 't_start'] = t_start
                exp_df.loc[no_time, 't_stop'] = t_start + stimulus_duration
                exp_df.loc[no_time, 'intensity'] = intensity
    return exp_df



