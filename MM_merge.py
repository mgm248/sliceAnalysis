from ceed.analysis.merge_data import CeedMCSDataMerger
import os
import logging

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\\'
recordings = []
from datetime import time
file_num = 1
for fname in os.listdir(ffolder): #collect all recordings, list in order
    if 'McsRecording' in fname:
        # Could use to sort the times, but I think they'll always be listed in the correct order...
        # recTime = time(hour=int(fname[11:13]), minute=int(fname[14:16]), second=int(fname[14:16]))
        # recordings.append(fname)
        ceed_file = ffolder + 'slice' + str(file_num) + '.h5'
        mcs_file = ffolder + fname
        output_file = ffolder + 'slice' + str(file_num) + '_merged.h5'
        file_num+=1
        debug = False

        merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

        merger.read_mcs_data()
        merger.read_ceed_data()
        merger.parse_mcs_data()

        alignment = {}
        try:
            for experiment in merger.get_experiment_numbers([]):
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

            merger.merge_data(output_file, alignment)

        except:
            print('Failed to merge slice' + str(file_num-1))

