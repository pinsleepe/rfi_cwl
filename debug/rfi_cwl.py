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

# create vector with start time
vec_length = 100
start_vector = np.linspace(0, tobs, num=vec_length, endpoint=False, retstep=True)
duration = start_vector[1]
full_stat = np.empty([nchans, vec_length], dtype='int')
# for sv in range(vec_length):
for sv in range(vec_length):
    block, _ = fil_rfiObs.read_time_freq(start_vector[0][sv], duration)
    val = filters.threshold_yen(block)
    mask = block < val

    op_struck = np.ones((5, 5))
    open_img = ndimage.binary_opening(mask,
                                      structure=op_struck)
    cl_struck = np.ones((1, 1))
    close_img = ndimage.binary_closing(open_img,
                                       structure=cl_struck)

    close_img_inv = np.invert(close_img)
    labeled_array, num_features = measurements.label(close_img_inv)
    new_m = close_img_inv.astype(int)
    sum_mask = new_m.sum(axis=1)

    # append this sum mask and save as h5 file for analysis
    full_stat[:, sv] = sum_mask

h5_name = header.source_name + '.h5'
with h5py.File(h5_name, 'w') as hf:
    hf.create_dataset("stats", data=full_stat)
    hf.create_dataset("vector", data=start_vector[0])
