#!/usr/local/bin/python
from rfipip import rfiObs, rfiDatabase
import numpy as np
from scipy import ndimage
from skimage import filters
from scipy.ndimage import measurements
import h5py
import argparse
from statistics import median
import timeit
import pandas as pd

parser = argparse.ArgumentParser(description='Find known RFI events in filterbank file.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path',
                    dest='path',
                    type=str, action='store',
                    default='',
                    help='Path to the fil file')
args = parser.parse_args()
path = ''
if args.path != '':
    path = args.path
if path == '':
    raise RuntimeError('Could not find a path to use.')

# time it
# tic = timeit.default_timer()

fil_rfiObs = rfiObs.RfiObservation(path=path,
                                   fil_file=True)
header = fil_rfiObs.file.header
tobs = header.tobs
nchans = header.nchans
# TODO download from google docs
# read RFI database
csv_path = '/home/RFI_Spectrum_Database.csv'
rfiDb = rfiDatabase.RfiDatabase()
rfiDb.write_dict([csv_path])
bands = rfiDb.dictionary.keys()

# save number of blocks vector
# create vector with start time
# this need some logic dependent on file size
# TODO
vec_length = 100
start_vector = np.linspace(0,
                           tobs,
                           num=vec_length,
                           endpoint=False,
                           retstep=True)

duration = start_vector[1]
# first find median for mask value
obs_val = []
for sv in range(vec_length):
    start_time = start_vector[0][sv]
    block, _ = fil_rfiObs.read_time_freq(start_time,
                                         duration)
    val = filters.threshold_yen(block)
    obs_val.append(val)

val = median(obs_val)
training_set = pd.DataFrame(columns=('event',
                                     'c_freq',
                                     'bw',
                                     't_start',
                                     'duration',
                                     'culprit',
                                     'description',
                                     'band'))
temp_set = training_set
for sv in range(vec_length):
    start_time = start_vector[0][sv]
    block, num_sam = fil_rfiObs.read_time_freq(start_time,
                                               duration)
    t_vector = fil_rfiObs.time
    mask = block < val
    op_struck = np.ones((2, 2))
    open_img = ndimage.binary_opening(mask,
                                      structure=op_struck)
    cl_struck = np.ones((1, 1))
    close_img = ndimage.binary_closing(open_img,
                                       structure=cl_struck)
    close_img_inv = np.invert(close_img)
    labeled_array, num_features = measurements.label(close_img_inv)
    start_sample = long(start_time / header.tsamp)
    feature_range = np.linspace(1, num_features, num=num_features)
    tf_extent = np.empty([num_features, 5],
                         dtype='int')
    # tf_extent = [event number,
    #             start freq,
    #             freq channels,
    #             start time,
    #             time samples]
    for ev in feature_range:
        x, y = np.where(labeled_array == ev)
        tf_e = [ev,
                x[0],
                x.max() - x[0],
                y[0],
                y.max() - y[0]]
        tf_extent[int(ev - 1), :] = tf_e
    norms = np.apply_along_axis(np.linalg.norm, 0, tf_extent[:, 4])
    time_occ = tf_extent[:, 4] / norms
    freq_occ = tf_extent[:, 2]

    pd_idx = 0
    t_df = fil_rfiObs.time[1] - fil_rfiObs.time[0]
    for ev in range(tf_extent.shape[0]):
        # logic for centre freq
        # check if the event occupies more than one channel
        # yes
        if tf_extent[ev][2] > 0:
            # freq channels times BW
            temp_bw = tf_extent[ev][2] * header.foff
            # freq of middle channel
            peak_freq = fil_rfiObs.freqs[tf_extent[ev][1]] + temp_bw / 2.0
            # duration
            temp_dur = tf_extent[ev][4] * t_df
        # no
        else:
            peak_freq = fil_rfiObs.freqs[tf_extent[ev][1]]
            temp_bw = header.foff
            temp_dur = t_df
        ev_f = peak_freq
        temp_t = fil_rfiObs.time[tf_extent[ev][3]]
        # find freq range
        for key in bands:
            top = float(key.split('-')[1])
            bottom = float(key.split('-')[0])
            band = rfiDb.dictionary[key]['band']
            if bottom <= ev_f <= top:  # freq is in range
                # frequencies not empty
                if rfiDb.dictionary[key]['frequencies']:
                    found_culprit = False
                    for av_f in rfiDb.dictionary[key]['frequencies'].keys():
                        #                     c_range = 0.1 # MHz
                        c_range = 1  # MHz
                        # check if there is a culprit close by
                        if (float(av_f) - c_range) <= ev_f <= (float(av_f) + c_range):
                            label = 1
                            temp_set.loc[pd_idx] = [tf_extent[ev][0],
                                                    peak_freq,
                                                    temp_bw,
                                                    temp_t,
                                                    temp_dur,
                                                    label,
                                                    rfiDb.dictionary[key]['frequencies'][av_f],
                                                    band]
                            pd_idx += 1
                            found_culprit = True
                    # no culprit write just band
                    if not found_culprit:
                        label = 0
                        temp_set.loc[pd_idx] = [tf_extent[ev][0],
                                                peak_freq,
                                                temp_bw,
                                                temp_t,
                                                temp_dur,
                                                label,
                                                'Unknown',
                                                band]
                        pd_idx += 1

                # frequencies empty
                else:
                    label = 0
                    temp_set.loc[pd_idx] = [tf_extent[ev][0],
                                            peak_freq,
                                            temp_bw,
                                            temp_t,
                                            temp_dur,
                                            label,
                                            'Unknown',
                                            band]
                    pd_idx += 1
        training_set.append(temp_set)

training_set.to_csv('training_set.csv')

# toc = timeit.default_timer()
# print('elapsed time in minutes %f' % (toc - tic)/60.0)
