from ceed.analysis.merge_data import CeedMCSDataMerger
import os
import logging

def merge(ffolder, fname, file_num):
    ceed_file = ffolder + 'slice' + str(file_num) + '.h5'
    mcs_file = ffolder + fname
    output_file = ffolder + 'slice' + str(file_num) + '_merged.h5'
    file_num += 1
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
        print('Failed to merge slice' + str(file_num - 1))

if __name__ == '__main__':
    ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-08-12\\'
    # recordings = []
    # """Merge one file at a time"""
    # file_num = 1
    # for fname in os.listdir(ffolder):  # collect all recordings, list in order
    #     if 'McsRecording' in fname:
    #         merge(ffolder, fname, file_num)
    #         file_num += 1

    """Merge all files in parallel"""
    file_num = 1
    fnames = []
    file_nums = []
    ffolders = []
    for fname in os.listdir(ffolder):  # collect all recordings, list in order
        if 'McsRecording' in fname:
            fnames.append(fname)
            file_nums.append(file_num)
            ffolders.append(ffolder)
            file_num+=1
    from joblib import Parallel, delayed
    # Parallel(n_jobs=len(fnames), backend="multiprocessing")(delayed(merge)(ffolder=ffolder, fname=fname, file_num=file_num) for ffolder in ffolders for fname in fnames for file_num in file_nums)
    Parallel(n_jobs=len(fnames), backend="multiprocessing")(delayed(merge)(ffolder=ffolders[i], fname=fnames[i], file_num=file_nums[i]) for i in range(0, len(fnames)))