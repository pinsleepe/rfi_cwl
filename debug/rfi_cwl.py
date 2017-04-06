#!/usr/local/bin/python
from rfipip import rfiObs, rfiDatabase
import numpy as np
from scipy import ndimage
from skimage import filters
from scipy.ndimage import measurements
import h5py
import argparse

parser = argparse.ArgumentParser(description='Plot outgoing p0 dsim data.',
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

fil_rfiObs = rfiObs.RfiObservation(path=path,
                                   fil_file=True)
header = fil_rfiObs.file.header
tobs = header.tobs
nchans = header.nchans

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
h5_name = 'scp' + '.h5'
with h5py.File(h5_name, 'w') as hf:
    # TODO time vector
    # save time vector
    t_vector = hf.create_dataset("t_vector",
                                 (vec_length, 1),
                                 maxshape=(vec_length, None))
    # save freq vector
    hf.create_dataset("f_vector",
                      data=fil_rfiObs.freqs)
    # next two  datasets are extendable
    # initialise first
    # save dc per channel
    dc_stats = hf.create_dataset("dc_stats",
                                 (nchans, vec_length),
                                 maxshape=(nchans, vec_length))
    # save time/freq extent
    tf_info = hf.create_dataset("tf_extent",
                                (1, 5),
                                maxshape=(None, 5))
    # save vector with num_features
    nf_vector = hf.create_dataset("nf_vector",
                                  (vec_length, 1),
                                  maxshape=(vec_length, 1))
    duration = start_vector[1]
    curr_tf = 0  # current tf
    for sv in range(vec_length):
        print('running block: %d' % sv)
        block, num_sam = fil_rfiObs.read_time_freq(start_vector[0][sv],
                                                   duration)
        if sv == 0:
            t_vector.resize((vec_length,
                             int(num_sam)))
        t_vector[sv, :] = fil_rfiObs.time
        val = filters.threshold_yen(block)
        mask = block < val
        # TODO different mask types for different events?
        op_struck = np.ones((5, 5))
        open_img = ndimage.binary_opening(mask,
                                          structure=op_struck)
        cl_struck = np.ones((1, 1))
        close_img = ndimage.binary_closing(open_img,
                                           structure=cl_struck)
        close_img_inv = np.invert(close_img)
        labeled_array, num_features = measurements.label(close_img_inv)
        nf_vector[sv] = num_features
        new_m = close_img_inv.astype(int)
        sum_mask = new_m.sum(axis=1)
        # append this sum mask and save as h5 file for analysis
        dc_stats[:, sv] = sum_mask
        tf_extent = np.empty([num_features, 5],
                             dtype='int')
        # TODO this loop is bottleneck
        for ev in np.linspace(1, num_features,
                              num=num_features):
            x, y = np.where(labeled_array == ev)
            tf_e = [ev, x.min(), x.max(), y.min(), y.max()]
            tf_extent[int(ev - 1), :] = tf_e
        tf_info.resize((curr_tf + num_features, 5))
        tf_info[curr_tf:curr_tf + num_features, :] = tf_extent
        curr_tf += num_features
