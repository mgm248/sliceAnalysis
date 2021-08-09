import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


def plot_stim_on_spectrogram(exp_df, exps=None, ax=None, rfs=1000, label_by='substage', lw=1):

    if exps is not None:
        exp_df = exp_df[exp_df['experiment'].isin(exps)]

    pbar = tqdm(exp_df.iterrows(), total=exp_df.shape[0], desc="Processing experiment info.", file=sys.stdout)
    for index, exp in pbar:
        t_start = exp['t_start']
        t_stop = exp['t_stop']
        if type(t_start) is str:
            t_start = np.float(t_start.strip('s'))
            t_stop = np.float(t_stop.strip('s'))
        if (np.isnan(t_start)) or (np.isnan(t_stop)):
            print("Skipping experiment #" + str(exp) + " (occurred prior to MEA recording).")
            continue
        duration = t_stop - t_start
        if ax is None:
            ax = plt.gca()
        rect = matplotlib.patches.Rectangle((t_start, 0), duration, int(rfs)/2, linewidth=lw, edgecolor='k',
                                            facecolor='none')
        ax.add_patch(rect)
        if label_by is not None:
            label = exp[label_by]
            bbox_props = dict(boxstyle="round", fc='white', lw=0.5)
            ax.text(t_start, 225, label, ha="left", va="top", rotation=0, size=8, bbox=bbox_props)

