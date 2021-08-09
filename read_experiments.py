from ceed.analysis import CeedDataReader
import mne
import numpy as np

def read_experiment_fullCosine(reader, Fs=20000):
    init = True
    usefirstonly = True
    for exp in range(0,len(reader.experiments_in_file)):
        reader.load_experiment(exp)
        print(reader.experiment_stage_name)
        # if reader.experiment_stage_name == 'Stage ABC full cosine':  # Read in times where A peaks (first stim of set)
        if reader.electrode_intensity_alignment is not None:
            if usefirstonly is True and reader.experiment_stage_name != 'eyfp':

                # peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] > .99)
                # i = 0
                # while i < peak_idxs[0].shape[0]:
                #     idx = peak_idxs[0][i]
                #     if init:
                #         my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs, #always use first
                #                                     duration=1,
                #                                     description='Experiment ' + reader.experiment_stage_name + ' Shape' + shape)
                #         init = False
                #     else:
                #         my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs, #always use first
                #                                     duration=1,
                #                                     description='Experiment ' + reader.experiment_stage_name + ' Shape' + shape)
                #     i += 6 #skipping the number of other peaks in the same stimulus


                shapes2plot = 'Shape-10'
                if reader.electrode_intensity_alignment is not None:
                    # find the peak of the stimulus
                    for shape in reader.shapes_intensity.keys():
                        if shape == shapes2plot or shapes2plot is None:
                            peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                            peak_idxs_b = np.where(reader.shapes_intensity['Shape-11'][:, 2] == 1)

                            i = 0
                            while i < peak_idxs[0].shape[0]:
                                idx = peak_idxs[0][i]
                                if idx==peak_idxs_b[0][i]:
                                    condition='0s'
                                if (idx - peak_idxs_b[0][i]) == -10:
                                    condition='.083s'
                                if (idx - peak_idxs_b[0][i]) == -1:
                                    condition='.0083s'
                                if not idx >= reader.electrode_intensity_alignment.shape[0]:
                                    t_start = reader.electrode_intensity_alignment[idx] / Fs
                                    duration = 5
                                    if init:
                                        my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs, #always use first
                                                                    duration=5,
                                                                    description='Experiment ' + reader.experiment_stage_name + 'Condition '+condition)
                                        init = False
                                    else:
                                        my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs, #always use first
                                                                    duration=5,
                                                                    description='Experiment ' + reader.experiment_stage_name + 'Condition '+condition)

                                i += 15  # skipping the number of other peaks in the same stimulus


            else:
                for shape in reader.shapes_intensity.keys():
                    peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                    for i in range(0, peak_idxs[0].shape[0]):
                        idx = peak_idxs[0][i]
                        if init:
                            my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs,
                                                       duration=.2,
                                                       description='Experiment ' + reader.experiment_stage_name + ' Shape' + shape)
                            init = False
                        else:
                            my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[idx] / Fs,
                                                        duration=.2,
                                                        description='Experiment ' + reader.experiment_stage_name + ' Shape' + shape)
    return my_annot



def read_experiment_halfcos_timedelays(reader, Fs=20000):
    init = True
    usefirstonly = True
    for exp in range(0,len(reader.experiments_in_file)): #TEMP
        reader.load_experiment(exp)
        print(reader.experiment_stage_name)
        if reader.electrode_intensity_alignment is not None:
            shapes2plot = 'Shape-11'
            # find the peak of the stimulus
            for shape in reader.shapes_intensity.keys():
                if shape == shapes2plot or shapes2plot is None:
                    peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                    peak_idxs_b = np.where(reader.shapes_intensity['Shape-12'][:, 2] == 1)

                    i = 0
                    for i in range(0, peak_idxs[0].shape[0]):
                        idx = peak_idxs[0][i]
                        condition = str(peak_idxs_b[0][i] - idx)
                        if not idx >= reader.electrode_intensity_alignment.shape[0]:
                            duration = 166.66
                            if init:
                                my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'A')
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs_b[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'B')
                                init = False
                            else:
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'A')
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs_b[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'B')
    return my_annot


def read_experiment_normalization(reader, Fs=20000, last_exp_only=True):
    init = True
    usefirstonly = True
    if last_exp_only:
        exp_loop = reader.experiments_in_file[-1]
    else:
        exp_loop = range(0,len(reader.experiments_in_file))
    for exp in exp_loop:
        reader.load_experiment(exp)
        print(reader.experiment_stage_name)
        if reader.electrode_intensity_alignment is not None:
            shapes2plot = 'Shape-5'
            # find the peak of the stimulus
            for shape in reader.shapes_intensity.keys():
                if shape == shapes2plot or shapes2plot is None:
                    peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                    peak_idxs_b = np.where(reader.shapes_intensity['Shape-4'][:, 2] == 1)
                    i = 0
                    for i in range(0, peak_idxs[0].shape[0]):
                        idx = peak_idxs[0][i]
                        condition = str(peak_idxs_b[0][i] - idx)
                        """just threw in some code from my attempts to use the ceed substages"""
                        stage_dict = reader.stage_factory.stage_names
                        stage = stage_dict[reader.experiment_stage_name]
                        substages = stage.stages
                        substages[0].get_state()
                        substages[0].copy_expand_ref()
                        if not idx >= reader.electrode_intensity_alignment.shape[0]:
                            duration = 166.66
                            if init:
                                my_annot = mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'A')
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs_b[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'B')
                                init = False
                            else:
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'A')
                                my_annot += mne.Annotations(onset=reader.electrode_intensity_alignment[peak_idxs_b[0][i]] / Fs,
                                                           # always use first
                                                           duration=duration,
                                                           description='Experiment: ' + reader.experiment_stage_name + '; Condition: ' + condition + '; Shape ' + 'B')
    return my_annot

def convert_expdf_toMNE(exp_df, Fs=20000):
    init = False
    for i, row in exp_df.iterrows():
        if not init:
            my_annot = mne.Annotations(onset=row['t_start'], duration=1,
                            description=row['substage'])
            init = True
        else:
            my_annot += mne.Annotations(onset=row['t_start'], duration=1,
                            description=row['substage'])

    return my_annot