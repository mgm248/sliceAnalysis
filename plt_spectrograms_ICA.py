from spectrogram import plot_spectrogram
from ceed.analysis import CeedDataReader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import scipy

def add_stim_to_spectrogram(reader):
    # get experiment data, add patches to plot
    for exp in range(0, len(reader.experiments_in_file)):
        reader.load_experiment(exp)
        if reader.experiment_stage_name == 'Stage ABC' or reader.experiment_stage_name == 'Stage ABC full Cosine':
            shapes2plot = 'B'
        else:
            shapes2plot = None
        if reader.electrode_intensity_alignment is not None:
            # find the peak of the stimulus
            for shape in reader.shapes_intensity.keys():
                if shape == shapes2plot or shapes2plot is None:
                    peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                    for i in range(0, peak_idxs[0].shape[0]):
                        idx = peak_idxs[0][i]
                        if not idx >= reader.electrode_intensity_alignment.shape[0]:
                            ax = plt.gca()
                            t_start = reader.electrode_intensity_alignment[idx] / 30000
                            duration = .2
                            rect = matplotlib.patches.Rectangle((t_start, 0), duration, Fs / 2, linewidth=1,
                                                                edgecolor='k',
                                                                facecolor='none')
                            ax.add_patch(rect)
                            bbox_props = dict(boxstyle="round", fc='white', lw=0.5)
                            ax.text(t_start, 225, shape, ha="left", va="top", rotation=0, size=8, bbox=bbox_props)

Fs = 1000

saveloc = 'D:\myRecordings\\1_22_20\Figures\Spectrograms\slice2_1_22_21_merged\ICA\\'
data = scipy.io.loadmat('C:\\Users\\Michael\Documents\MATLAB\ICA analysis\\activations.mat')
for elec in range(0,data['activations'].shape[0]):
    plt.figure(figsize=(18.0, 10.0))
    plot_spectrogram(data['activations'][elec,:], Fs)
    # add_stim_to_spectrogram(reader)
    plt.title(elec)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.show()
    plt.savefig(saveloc + 'activation' + str(elec) + '.png')
    plt.close()





