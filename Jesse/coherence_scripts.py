from elephant.spectral import welch_cohere
import neo
import math
from scipy import signal
from scipy.signal import hilbert
from quantities import ms, Hz, uV, s
from ephys_analysis.lfp_processing.filters import butter_lowpass_filter
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ceed.analysis import CeedDataReader
import sys
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook

new = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9',
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


def find_coherence(h5, ref_electrode, coh_excel, times, freq=20*Hz, freq_res=1*Hz, channels=new, rfs=512*Hz,
                   mag_sq=False):

    columns = ["reference", "electrode", "t_start", "t_stop", "frequency", "coherence", "phase"]
    coherence_df = pd.DataFrame(index=range(len(channels)), columns=columns)

    reader = CeedDataReader(h5)
    reader.open_h5()
    reader.load_mcs_data()

    offset, scale = reader.get_electrode_offset_scale(ref_electrode)
    fs = reader.electrodes_metadata[ref_electrode]['sampling_frequency'] * Hz

    raw_data = (np.array(reader.electrodes_data[ref_electrode]) - offset) * scale
    raw_data = raw_data * 2000000
    raw_signal = neo.core.AnalogSignal(raw_data, sampling_rate=fs, units='uV')

    num_samples = int(raw_signal.duration.rescale(s) * rfs / Hz)
    resampled_signal = butter_lowpass_filter(raw_signal, rfs / 2, fs)
    resampled_signal = signal.resample(resampled_signal, num_samples)
    resampled_signal = neo.core.AnalogSignal(resampled_signal, units=uV, sampling_rate=rfs, t_start=0*s)
    resampled_signal = resampled_signal.time_slice(times[0], times[1])
    print("Loaded reference electrode...")

    j = 0
    for channel in tqdm(channels):
        channel_data = (np.array(reader.electrodes_data[channel]) - offset) * scale
        channel_data = channel_data * 2000000
        channel_signal = neo.core.AnalogSignal(channel_data, sampling_rate=fs, units='uV')
        channel_resampled_signal = butter_lowpass_filter(channel_signal, rfs / 2, fs)
        channel_resampled_signal = signal.resample(channel_resampled_signal, num_samples)
        channel_resampled_signal = neo.core.AnalogSignal(channel_resampled_signal, units=uV, sampling_rate=rfs, t_start=0 * s)
        channel_resampled_signal = channel_resampled_signal.time_slice(times[0], times[1])

        print("Loaded electrode " + channel + "...")

        # this line is what calls the code from the elephant package
        freqs, coherence, phase_lags = welch_cohere(resampled_signal, channel_resampled_signal, freq_res=freq_res,
                                                    window='hamming', detrend='constant', scaling='spectrum')

        # in case freq_res != 1, this finds the frequency bin closest to the freq parameter
        freq_approx = min(freqs.tolist(), key=lambda x: abs(x - freq))
        freq_index = np.where(freqs == freq_approx)

        if not mag_sq:
            coherence = np.sqrt(coherence)
        peak_freq_coherence = coherence[freq_index]
        peak_freq_phase = phase_lags[freq_index]
        coherence_df.iloc[j, :5] = ref_electrode, channel, times[0], times[1], freq
        coherence_df.iloc[j, 5] = peak_freq_coherence[0][0]
        coherence_df.iloc[j, 6] = peak_freq_phase[0][0]
        j += 1

    book = load_workbook(coh_excel)
    writer = pd.ExcelWriter(coh_excel, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    str_freq = str(int(freq.item()))
    coherence_df.to_excel(writer, sheet_name="ref" + ref_electrode + "_" + str_freq + "hz",
                          startrow=0, startcol=0, header=True, index=False)
    writer.save()


def quiver_plot_from_excel(coh_excel, ref_electrode, freq, blackout_corners=True):

    str_freq = str(int(freq.item()))
    sheet = "ref" + ref_electrode + "_" + str_freq + "hz"

    X = np.arange(1, 13)
    Y = np.arange(1, 13)
    X_components = np.zeros([12, 12])
    Y_components = np.zeros([12, 12])

    excel_file = pd.ExcelFile(coh_excel)
    coh_df = excel_file.parse(sheet)
    pbar = tqdm(coh_df.iterrows(), total=coh_df.shape[0], desc="Processing " + sheet, file=sys.stdout)

    for index, row in pbar:
        channel = str(row['electrode'])
        channel_letter = channel[0]
        channel_x = (ord(channel_letter) - 64) - 1  # ord()-64 to convert letter to #, - 1 for zero-indexing
        channel_y = 12 - int(channel[1:])

        # reconstruct vector from phase and coherence
        coherence = np.float(row['coherence'])
        phase = np.float(row['phase'][:-3])
        u = coherence * math.cos(phase)
        v = coherence * math.sin(phase)
        X_components[channel_y, channel_x] = u
        Y_components[channel_y, channel_x] = v

    fig, ax = plt.subplots(1)
    plt.quiver(X, Y, X_components, Y_components, angles='xy', scale_units='xy', scale=1,
               headlength=4, headaxislength=4, headwidth=2, alpha=1)

    plt.xlabel('Column')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'])
    plt.ylabel('Row')
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1'])

    if blackout_corners:
        blackout = [(1, 1), (1, 2), (1, 3), (1, 10), (1, 11), (1, 12), (2, 1), (2, 2), (2, 11), (2, 12), (3, 1), (3, 12),
                    (10, 1), (10, 12), (11, 1), (11, 2), (11, 11), (11, 12), (12, 1), (12,2), (12,3), (12,10), (12,11),
                    (12, 12)]
        for coord in blackout:
            coord = coord[0]-0.5, coord[1]-0.5
            rect = plt.Rectangle(coord, 1, 1, facecolor='k')
            ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":

    date = "2-24-20"
    slice = '2'

    base_filename = r'D:\{date}\{file}.{ext}'
    h5_file = date + "___slice" + str(slice) + "_merged"
    h5_file = base_filename.format(date=date, file=h5_file, ext="h5")  # path for the merged h5 file

    excel = base_filename.format(date=date, file="coherence", ext="xlsx")  # path for the excel file which the results will be written to
    sheet = "slice" + str(slice)

    find_coherence(h5_file, "A4", excel, [10*s, 20*s], channels=new[0:8])
    quiver_plot_from_excel(excel, "A4", 33*Hz)
