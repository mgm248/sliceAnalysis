import os

from quantities import ms, Hz, uV, s
import numpy as np
import matplotlib.pyplot as plt
import neo
from neo.io.nixio import NixIO
from neo.core import AnalogSignal, SpikeTrain, IrregularlySampledSignal
from ceed.analysis import CeedDataReader
from ceed.function import CeedFuncRef
from ceed.stage import CeedStageRef
import pandas as pd
from itertools import compress
from openpyxl import load_workbook

"""(I think) this gets the time course of a given stimulus """
def get_stimulus_signal(reader: CeedDataReader, exp, shape="enclosed", led="blue", returnas="percent"):

    if shape=='Shape A':
        shape='Shape-3'
    elif shape=='Shape B':
        shape = 'Shape-2'
    reader.load_application_data()
    reader.load_experiment(exp)
    shape_intensities = np.array(reader.shapes_intensity[shape])
    alignment = np.array(reader.electrode_intensity_alignment)

    if led == "red":
        index = 0
    if led == "green":
        index = 1
    if led == "blue":
        index = 2
    intensity = shape_intensities[:-1, index]

    if returnas == "percent":
        intensity = intensity * 100
    if returnas == "norm":
        max = np.max(intensity)
        intensity = intensity / max

    fs = reader.electrodes_metadata['A4']['sampling_frequency']*Hz  # arbitrary electrode
    period = 1./fs
    times = [x * period for x in alignment]
    times = np.array(times)
    intensity = intensity[0:times.shape[0]]
    stimulus = IrregularlySampledSignal(times, intensity, units='percent', time_units='s')
    return stimulus


def get_all_exps_AB(file):
    reader = CeedDataReader(file)
    reader.open_h5()
    reader.load_application_data()

    exps = reader.experiments_in_file

    reader.load_mcs_data()
    fs = reader.electrodes_metadata['A4']['sampling_frequency'] * Hz
    period = (1. / fs).rescale(s)
    led = ['red', 'green', 'blue']


    columns = ['experiment', 'stage', 'substage', 'duration', 'intensity A', 'intensity B', 'color', 't_start', 't_stop',
               'signal A', 'signal B']
    stim_df = pd.DataFrame(index=range(1000), columns=columns)

    i = 0
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment is not None:
            alignment = np.array(reader.electrode_intensity_alignment)
            times = [round(x * period, 5) for x in alignment]
            # frame_rate = reader.view_controller.effective_frame_rate
            # data = reader.stage_factory.get_all_shape_values(
            #     frame_rate, reader.experiment_stage_name,
            #     pre_compute=reader.view_controller.pre_compute_stages
            # )
            # print(data)
            # break
            stage_dict = reader.stage_factory.stage_names
            stage = stage_dict[reader.experiment_stage_name]
            substages = stage.stages
            signal_a_overall = get_stimulus_signal(reader, exp, 'Shape A', led="blue", returnas="percent")
            signal_b_overall = get_stimulus_signal(reader, exp, 'Shape B', led="blue", returnas="percent")
            if len(substages) == 0:
                stim_df.loc[i, 'experiment'] = str(exp)
                stim_df.loc[i, 'stage'] = reader.experiment_stage_name
                stim_df.loc[i, 'substage'] = "None"
                if len(stage.shapes) == 0:
                    print("Skipping experiment #" + str(exp) + "; no shapes were found within the stage.")
                    continue
                patterns = [x.name for x in stage.shapes]
                stim_df.loc[i, 'pattern'] = tuple(patterns)

                if type(stage.functions[0]) == CeedFuncRef:
                    functions = [x.func.name for x in stage.functions]
                else:
                    functions = [x.name for x in stage.functions]
                stim_df.loc[i, 'function'] = functions

                color_mask = [stage.color_r, stage.color_g, stage.color_b]
                colors = list(compress(led, color_mask))
                stim_df.loc[i, 'color'] = colors


                if len(alignment.shape) == 0:
                    stim_df.loc[i, 't_start'] = np.nan
                    stim_df.loc[i, 't_stop'] = np.nan
                    stim_df.loc[i, 'signal'] = [np.nan]
                    stim_df.loc[i, 'intensity A'] = np.nan
                    stim_df.loc[i, 'intensity B'] = np.nan
                else:
                    stim_df.loc[i, 't_start'] = times[0]
                    stim_df.loc[i, 't_stop'] = times[-1]
                    # signal = get_stimulus_signal(reader, exp, patterns[0], led=colors[0], returnas="percent")
                    # stim_df.loc[i, 'signal'] = signal
                    stim_df.loc[i, 'intensity A'] = max(signal_a).item()
                    stim_df.loc[i, 'intensity B'] = max(signal_b).item()
                i += 1

            else:
                if len(alignment.shape) != 0:
                    exp_timer = round(alignment[0] * period, 5)
                n_loops = 20
                idx = 0
                for l in range(0, n_loops):
                    print('Loop ' + str(l))
                    for k, sub_stage in enumerate(substages):
                        try:
                            sub_stage[0].name
                        except:
                            sub_stage = sub_stage.copy_expand_ref()
                        if (idx != 0) and (len(alignment.shape) != 0):
                            stop_index = times.index(true_t_stop)
                            exp_timer = times[stop_index]  # +1 to grab one sample after the t_stop value for prior sub_stage.
                        stim_df.loc[idx, 'experiment'] = str(exp)
                        stim_df.loc[idx, 'stage'] = reader.experiment_stage_name
                        try:
                            stim_df.loc[idx, 'substage'] = substages[k].get_state()['ref_name']
                        except:
                            stim_df.loc[idx, 'substage'] = sub_stage.name

                        patterns = [x.name for x in sub_stage.shapes]
                        # stim_df.loc[i + k, 'pattern'] = tuple(patterns)
                        # functions = [x.name for x in sub_stage.functions]
                        # stim_df.loc[i + k, 'function'] = functions


                        color_mask = [sub_stage.color_r, sub_stage.color_g, sub_stage.color_b]
                        colors = list(compress(led, color_mask))
                        stim_df.loc[idx, 'color'] = colors

                        if len(alignment.shape) == 0:
                            stim_df.loc[idx, 't_start'] = np.nan
                            stim_df.loc[idx, 't_stop'] = np.nan
                            stim_df.loc[idx, 'signal'] = [np.nan]
                            stim_df.loc[idx, 'intensity A'] = np.nan
                            stim_df.loc[idx, 'intensity B'] = np.nan

                        else:
                            try:
                                duration = sub_stage.functions[0].duration * s
                            except:
                                sub_stage = sub_stage.stages[0]
                                duration = sub_stage.functions[0].duration * s

                            timebase = sub_stage.functions[0].timebase
                            if not (timebase.denominator == 1):
                                duration = duration * (timebase.numerator / timebase.denominator)
                            if 'A delay B' in stim_df['substage'][idx]:
                                duration = duration+(6*(25/2999))*s #temp solution
                            if 'B delay A' in stim_df['substage'][idx]:
                                duration = duration + sub_stage.functions[1].duration * s  # temp solution
                            stim_df.loc[idx, 'duration'] = duration
                            t_stop = exp_timer + duration

                            signal_a = signal_a_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            signal_b = signal_b_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            true_t_stop = round(signal_a.t_stop, 5)
                            stim_df.loc[idx, 't_start'] = signal_a.t_start
                            stim_df.loc[idx, 't_stop'] = true_t_stop
                            try:
                                stim_df.loc[idx, 'signal A'] = signal_a
                                stim_df.loc[idx, 'signal B'] = signal_b
                            except:
                                stim_df.loc[idx, 'signal A'] = [signal_a]
                                stim_df.loc[idx, 'signal B'] = [signal_b]
                            stim_df.loc[idx, 'intensity A'] = max(signal_a).item()
                            stim_df.loc[idx, 'intensity B'] = max(signal_b).item()
                            idx+=1

                i += k + 1
    stim_df.dropna(inplace=True, how='all')
    return stim_df


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_all_exps_enclosed(file):
    reader = CeedDataReader(file)
    reader.open_h5()
    reader.load_application_data()

    exps = reader.experiments_in_file

    reader.load_mcs_data()
    fs = reader.electrodes_metadata['A4']['sampling_frequency'] * Hz
    period = (1. / fs).rescale(s)
    led = ['red', 'green', 'blue']


    columns = ['experiment', 'stage', 'substage', 'duration', 'intensity', 'frequency', 'color', 't_start', 't_stop',
               'signal']
    stim_df = pd.DataFrame(index=range(1000), columns=columns)

    i = 0
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment is not None:
            alignment = np.array(reader.electrode_intensity_alignment)
            times = [round(x * period, 5) for x in alignment]
            # frame_rate = reader.view_controller.effective_frame_rate
            # data = reader.stage_factory.get_all_shape_values(
            #     frame_rate, reader.experiment_stage_name,
            #     pre_compute=reader.view_controller.pre_compute_stages
            # )
            # print(data)
            # break
            stage_dict = reader.stage_factory.stage_names
            #stage = stage_dict[reader.experiment_stage_name]
            stage = reader.experiment_stage
            substages = stage.stages
            signal_overall = get_stimulus_signal(reader, exp, 'enclosed', led="blue", returnas="percent")
            if len(alignment.shape) != 0:
                exp_timer = round(alignment[0] * period, 5)
            if len(substages) == 0:
                n_loops = len(stage.functions[0].noisy_parameter_samples['f'])
                idx = 0
                sig_i = 0
                for l in range(0, n_loops):
                    if (i != 0) and (len(alignment.shape) != 0):
                        stop_index = times.index(true_t_stop)
                        # exp_timer = times[stop_index]
                    stim_df.loc[i, 'experiment'] = str(exp)
                    stim_df.loc[i, 'stage'] = reader.experiment_stage_name
                    stim_df.loc[i, 'substage'] = "None"
                    if len(stage.shapes) == 0:
                        print("Skipping experiment #" + str(exp) + "; no shapes were found within the stage.")
                        continue
                    patterns = [x.name for x in stage.shapes]
                    stim_df.loc[i, 'pattern'] = tuple(patterns)

                    if type(stage.functions[0]) == CeedFuncRef:
                        functions = [x.func.name for x in stage.functions]
                    else:
                        functions = [x.name for x in stage.functions]

                    color_mask = [stage.color_r, stage.color_g, stage.color_b]
                    colors = list(compress(led, color_mask))
                    stim_df.loc[i, 'color'] = colors


                    if len(alignment.shape) == 0:
                        stim_df.loc[i, 't_start'] = np.nan
                        stim_df.loc[i, 't_stop'] = np.nan
                        stim_df.loc[i, 'signal'] = [np.nan]
                        stim_df.loc[i, 'intensity'] = np.nan
                    else:
                        duration = stage.functions[0].duration*s
                        #duration = duration + (1 * (25 / 2999)) * s  # temp solution
                        stim_df.loc[i, 'duration'] = duration
                        stim_df.loc[i, 'frequency'] = stage.functions[0].noisy_parameter_samples['f'][i]
                        found0 = False
                        while not found0:
                            sig_i+=1
                            if signal_overall[sig_i] == 0:
                                found0=True

                        #t_stop = exp_timer + duration + stage.functions[1].copy_expand_ref().duration*s
                        t_stop = signal_overall.times[sig_i]
                        signal = signal_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                        true_t_stop = round(signal.t_stop, 5)

                        stim_df.loc[i, 't_start'] = signal.t_start
                        stim_df.loc[i, 't_stop'] = exp_timer+duration
                        try:
                            stim_df.loc[i, 'signal'] = signal
                        except:
                            stim_df.loc[i, 'signal'] = [signal]
                        stim_df.loc[i, 'intensity'] = max(signal).item()

                        foundnon0 = False
                        while not foundnon0:
                            if sig_i > signal_overall.shape[0]-1:
                                stim_df.dropna(inplace=True, how='all')
                                return stim_df
                            if not signal_overall[sig_i] == 0:
                                foundnon0=True
                                sig_i-=1
                            else:
                                sig_i += 1
                        exp_timer = signal_overall.times[sig_i]

                    i += 1

            else:
                if len(alignment.shape) != 0:
                    exp_timer = round(alignment[0] * period, 5)
                n_loops = 10
                idx = 0
                for l in range(0, n_loops):
                    print('Loop ' + str(l))
                    for k, sub_stage in enumerate(substages):
                        try:
                            sub_stage[0].name
                        except:
                            sub_stage = sub_stage.copy_expand_ref()
                        if (idx != 0) and (len(alignment.shape) != 0):
                            stop_index = times.index(true_t_stop)
                            exp_timer = times[stop_index]  # +1 to grab one sample after the t_stop value for prior sub_stage.
                        stim_df.loc[idx, 'experiment'] = str(exp)
                        stim_df.loc[idx, 'stage'] = reader.experiment_stage_name
                        try:
                            stim_df.loc[idx, 'substage'] = substages[k].get_state()['ref_name']
                        except:
                            stim_df.loc[idx, 'substage'] = sub_stage.name

                        patterns = [x.name for x in sub_stage.shapes]
                        # stim_df.loc[i + k, 'pattern'] = tuple(patterns)
                        # functions = [x.name for x in sub_stage.functions]
                        # stim_df.loc[i + k, 'function'] = functions


                        color_mask = [sub_stage.color_r, sub_stage.color_g, sub_stage.color_b]
                        colors = list(compress(led, color_mask))
                        stim_df.loc[idx, 'color'] = colors

                        if len(alignment.shape) == 0:
                            stim_df.loc[idx, 't_start'] = np.nan
                            stim_df.loc[idx, 't_stop'] = np.nan
                            stim_df.loc[idx, 'signal'] = [np.nan]
                            stim_df.loc[idx, 'intensity'] = np.nan

                        else:
                            try:
                                duration = sub_stage.functions[0].duration * s
                            except:
                                sub_stage = sub_stage.stages[0]
                                duration = sub_stage.functions[0].duration * s

                            timebase = sub_stage.functions[0].timebase
                            if not (timebase.denominator == 1):
                                duration = duration * (timebase.numerator / timebase.denominator)
                            stim_df.loc[idx, 'duration'] = duration
                            t_stop = exp_timer + duration

                            signal_a = signal_a_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            true_t_stop = round(signal_a.t_stop, 5)
                            stim_df.loc[idx, 't_start'] = signal.t_start
                            stim_df.loc[idx, 't_stop'] = true_t_stop
                            try:
                                stim_df.loc[idx, 'signal'] = signal
                            except:
                                stim_df.loc[idx, 'signal'] = [signal]
                            stim_df.loc[idx, 'intensity'] = max(signal).item()
                            idx+=1

                i += k + 1
    stim_df.dropna(inplace=True, how='all')
    return stim_df

def write_exp_df_to_excel(exp_df, excel, sheet):
    book = load_workbook(excel)
    writer = pd.ExcelWriter(excel)
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    exp_df.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, header=True, index=True)
    writer.save()


def read_exp_df_from_excel(excel, sheet):
    excel_file = pd.ExcelFile(excel, engine='openpyxl')
    stim_df = excel_file.parse(sheet)
    return stim_df


def zero_runs(a):
    """
    Returns a Nx2 dimensional array, with each row containing the start and stop indices of contiguous zeros in the
    original array, a.
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

#TODO: Ask Jesse what this does?
def divy_exp_series(exp_df, exps, sub_exps=10):

    new_exp_df = exp_df
    for exp in exps:
        if not isinstance(exp, str):
            exp = str(exp)
        sub_exp_df = pd.DataFrame(index=range(sub_exps), columns=exp_df.columns) #['experiment', 't_start', 't_stop', 'pattern', 'intensity', 'color', 'signal']
        sub_exp_df.loc[:, 'pattern'] = exp_df[exp_df['experiment']==exp]["pattern"].values[0] * sub_exps

        signal = exp_df[exp_df['experiment']==exp]["signal"].values
        signal = signal[0] #IrregularlySampledSignal object
        signal_aslist = signal.reshape(signal.shape[0]).tolist()
        zero_periods = zero_runs(signal_aslist)
        if zero_periods.shape[0] == sub_exps:
            zero_periods = np.insert(zero_periods, 0, [-1, 0], axis=0)

        if zero_periods.shape[0] != sub_exps+1:
            raise Exception("sub_exps it not equal to the number of intertrial intervals in the original data"
                            + str(zero_periods.shape[0]), str(zero_periods))

        for sub_exp in range(0, sub_exps):
            start_index = zero_periods[sub_exp, 1]
            stop_index = zero_periods[sub_exp+1, 0]
            sub_signal = signal[start_index:stop_index]

            sub_exp_df.loc[sub_exp, "t_start"] = sub_signal.t_start
            sub_exp_df.loc[sub_exp, "t_stop"] = sub_signal.t_stop
            sub_exp_df.loc[sub_exp, "intensity"] = round(max(sub_signal).item(), 2)
            sub_exp_df.loc[sub_exp, "experiment"] = exp + "-" + str(sub_exp)
            sub_exp_df.loc[sub_exp, "signal"] = sub_signal

        new_exp_df = new_exp_df.append(sub_exp_df)
        new_exp_df = new_exp_df[new_exp_df.experiment != exp]
    return new_exp_df

def extract_habituation_evts(df):
    #second event will always have 15s delay, first event will depend on the length of the previous stage
    df_start = df[df['start_loop']=='start_loop']
    df_end = df[df['start_loop']=='end_loop']
    evts = []
    for i, row in df_start.iterrows():
        if i == 0:
            evts.append(
                {'substage': 'ITI infinite s (first event)', 't_start': row['t_start']})
            stage_length = df_end.iloc[i]['t_start'] - row['t_start']
            ITI = round(stage_length - 16, 1)
            evts.append(
                {'substage': 'ITI ' + str(ITI) + 's', 't_start': row['t_start'] + ITI + .5})
            evts.append({'substage': 'ITI 15s',
                         't_start': row['t_start'] + ITI + 16})
        else:
            #get length of stage
            try:
                if not isinstance(df_end.iloc[i]['t_start'], str):  # if it is a string, out of range
                    stage_length = df_end.iloc[i]['t_start'] - row['t_start']
                    ITI = round(stage_length - 16, 1)
                evts.append(
                    {'substage': 'ITI ' + str(ITI) + 's', 't_start': row['t_start']+ITI+.5})
                evts.append({'substage': 'ITI 15s',
                             't_start': row['t_start'] + ITI + 16})  # there will always be an event 15s later
            except:
                print('huh')
        print(i)

    return pd.DataFrame(evts)

def get_exp_eventdata(reader, select_dataframe):
    """
    Get experiment dataframe using event data

    select_dataframe:: 'combine' -- picks the experiment with the most events
        or 'biggest' -- returns all the experiments in one dataframe
    """
    dfs = []
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment is not None:
            d = {}
            original_root = reader.stage_factory.stage_names[reader.experiment_stage_name]
            for exp_stage, orig_stage in zip(reader.experiment_stage.get_stages(), original_root.get_stages(True)):
                orig_stage = orig_stage.stage if isinstance(s, CeedStageRef) else orig_stage
                d[exp_stage.ceed_id] = exp_stage, orig_stage.name
                for f in exp_stage.functions:
                    for ff in f.get_funcs():
                        d[ff.ceed_id] = ff, ff.name

            if d[0][1] == 'Whole experiment' or d[0][1] == 'Whole experiment-2' or d[0][1]=='Stage-2' or d[0][1]=='Stage-3':
                print('Getting event data for ' + str(d[0][1]))
                flat_events = []
                i=0
                while i < len(reader.event_data):
                    ev = reader.event_data[i]
                    try:
                        if len(d[ev[1]][0].stages) > -1:
                            flat_events.append([ev[0], ev[1], d[ev[1]][1], ev[2]])
                        i+=1
                    except:
                        i+=1

                # flat_events = [[ev[0], ev[1], d[ev[1]][1], ev[2]] + list(ev[3]) for ev in reader.event_data]

                loop_events = [line for line in flat_events if line[3] == 'start_loop']
                df_start = pd.DataFrame(loop_events, columns=['Frame','ceed ID','substage','start_loop'])
                loop_events = [line for line in flat_events if line[3] == 'end_loop']
                df_end = pd.DataFrame(loop_events, columns=['Frame','ceed ID','substage','start_loop'])
                df = df_start.append(df_end)

                """Drop subevents of substages with multiple stages"""
                larger_events = ['A weak B medium cos', 'A weak B medium sq','A strong B medium cos', 'A strong B medium sq',
                                 'A medium B weak cos', 'A medium B weak sq', 'A medium B strong cos', 'A medium B strong sq',
                                 'A strong B weak sq', 'A weak B strong sq', 'A strong B weak ramp', 'A weak B strong ramp',
                                 'A delay B', 'B delay A']
                to_drop = []
                for index, row in df.iterrows():
                    if row[2] in larger_events:
                        to_drop.append(index+1)
                        to_drop.append(index+2)

                df = df.drop(labels=to_drop, axis=0)
                # df = df.drop(labels=['start_loop'], axis=1)
                df = df.sort_values('Frame', axis=0, ascending=True)
                aligned_times = []
                for index, row in df.iterrows():
                    try:
                        aligned_times.append(reader.electrode_intensity_alignment_gpu_rate[row[0]] / 20000)
                    except:
                        aligned_times.append('out of range')
                df['t_start'] = aligned_times
                if df.shape[0] > 1:
                    df = extract_habituation_evts(df)
                    df = df.sort_values('t_start', axis=0, ascending=True)
                    dfs.append(df)

                # dfs.append(df)

    # stage_names = [s.stage.name if isinstance(s, CeedStageRef) else s.name for s in
    #                original_root.stages]

    # #get stage name using stage ID
    # stage_index = stage_names.index('B strong sq')
    # ceed_id = reader.experiment_stage.stages[stage_index].ceed_id
    #
    # #filter for strong, find when first event started
    # strong = df[df[2] == 'B strong sq']
    # strong_loop_0 = strong[strong[3] == 'loop_start']
    # strong = strong[strong[5] == 0]
    # i = strong.iloc[0][0]
    if select_dataframe == 'biggest':
        max_length = 0
        for df in dfs:
            if df.shape[0] > max_length:
                max_length = df.shape[0]
                biggest_df = df
        return biggest_df
    elif select_dataframe == 'combine':
        print('adding str NE to all events in second experiment')
        for di, df in enumerate(dfs):
            if di == 0:
                overall_df = df
            elif di == 1:
                df['substage'] = 'NE ' + df['substage']
                overall_df = overall_df.append(df)
            elif di == 2:
                df['substage'] = 'Washout ' + df['substage']
                overall_df = overall_df.append(df)
        return overall_df



if __name__ == "__main__":

    #
    # date = "10-30-20"
    # slice = 3
    #
    # base_filename = r'D:\{date}\{file}.{ext}'
    # h5_file = date + "___slice" + str(slice) + "_merged"
    # h5_file = base_filename.format(date=date, file=h5_file, ext="h5")
    import pprint
    ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\\'
    for fname in os.listdir(ffolder):
        #if 'merged' in fname: # and 'ramp' in fname:
        if '_merged' in fname:
            ceed_data = ffolder + fname
            Fs = 20000
            reader = CeedDataReader(ceed_data)
            # open the data file
            reader.open_h5()

            exp_df = get_exp_eventdata(reader, select_dataframe='combine')  # Grab all experiment information
            exp_df.to_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl')

    # electrode = 'A4'
    # offset, scale = reader.get_electrode_offset_scale(electrode)
    # fs = reader.electrodes_metadata[electrode]['sampling_frequency'] * Hz
    # period = 1. / fs
    # raw_data = (np.array(reader.electrodes_data[electrode]) - offset) * scale
    # raw_data = raw_data * 2000000
    # raw_signal = neo.core.AnalogSignal(raw_data, sampling_rate=fs, units='uV')
    # # Plot data from electrode of interest
    # plt.plot(raw_signal.times, raw_signal, 'k', lw=0.5)
    #
    # # Grab stimulus signal for given experiment, and plot it alongside electrode data
    # exp = 1
    # stim = get_stimulus_signal(reader, exp, shape="enclosed")
    # plt.plot(stim.times, stim, 'b', lw=2)
    # plt.show()

    # exp_excel = base_filename.format(date=date, file="experiment_df", ext="xlsx")  # exp_excel is your excel filename
    # exp_excel = 'D:\myRecordings\\experiment_df.xlsx'
    # sheet = "slice" + str(slice)
    # write_exp_df_to_excel(exp_df, exp_excel, sheet)  # write experiment information for given slice to exp_excel

    # Load experiment stimulus information from excel file
    # exp_df = read_exp_df_from_excel(exp_excel, sheet)
    # exp_df = exp_df[exp_df['experiment'] == exp]
    # stim_start = exp_df['t_start'].values[0]
    # stim_stop = exp_df['t_stop'].values[0]
    # pattern = exp_df['pattern'].values[0]
    # intensity = exp_df['intensity'].values[0]
    # function = exp_df['function'].values[0]
    #
    # print("Experiment " + str(exp) + ", consisting of " + str(pattern) + ", was presented as a " + function +
    #       " with intensity = " + str(intensity) + ", from " + str(stim_start) + " to " + str(stim_stop))
