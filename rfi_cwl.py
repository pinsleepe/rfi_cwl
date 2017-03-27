import sys
sys.path.append('/home/psr/docker_scratch/src')

import rfiObs, rfiDatabase
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt, mpld3
import matplotlib.mlab as mlab
import numpy as np

fil_path = '/home/psr/hackathon'
def allFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

fil_list = allFiles(fil_path)

fil_rfiObs1 = rfiObs.RfiObservation(path=join(fil_path,
                                              fil_list[1]),
                                    fil_file=True)
header = fil_rfiObs1.file.header
start_time = header.tobs//2  # in seconds
duration = 1  # in seconds
block, nsamples = fil_rfiObs1.read_time_freq(start_time, duration)


start_sample = long(start_time / fil_rfiObs1.file.header.tsamp)

# print type(block[0, 0]), block[:10, :10]
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.imshow(block,
           extent=[start_sample * fil_rfiObs1.file.header.tsamp,
                   (start_sample + np.shape(block)[1]) * fil_rfiObs1.file.header.tsamp,
                   np.shape(block)[0], 0],
           cmap='viridis',
           aspect='auto')
ax.set_aspect("auto")
ax.set_xlabel('observation time (secs)')
ax.set_ylabel('freq channel')
ax.set_title(header.source_name)
ax.set_aspect('auto')
mpld3.save_html(fig, 'rfi.html')
