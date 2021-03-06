import logging
from ceed.analysis.merge_data import CeedMCSDataMerger
fdir = r'C:\Users\Michael\Analysis\myRecordings_extra\21-11-17\\'
ceed_file = fdir+'slice5.h5'
mcs_file = fdir+'2021-11-17T15-52-00McsRecording.h5'
output_file = ceed_file.replace('.h5','_merged.h5')
notes = ''
notes_filename = None
debug = False

merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

merger.read_mcs_data()
merger.read_ceed_data()
merger.parse_mcs_data()

alignment = {}
for experiment in merger.get_experiment_numbers():
    merger.read_ceed_experiment_data(experiment)
    merger.parse_ceed_experiment_data()
    try:
        align = alignment[experiment] = merger.get_alignment(ignore_additional_ceed_frames=True)
        # print experiment summary, see method for column meaning
        print(merger.get_skipped_frames_summary(align, experiment))
    except Exception as e:
        print(
            "Couldn't align MCS and ceed data for experiment "
            "{} ({})".format(experiment, e))
        if debug:
            logging.exception(e)

merger.merge_data(
    output_file, alignment, notes=notes, notes_filename=notes_filename)
