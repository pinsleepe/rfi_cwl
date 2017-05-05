#!/usr/local/bin/python
# import matplotlib
# matplotlib.use('Agg')
from rfipip import rfiObs, rfiDatabase, rfiEvent, rfiUtils
import numpy as np
# from scipy import ndimage
# from skimage import filters
from scipy.ndimage import measurements
# import h5py
# import argparse
# from statistics import median
# import timeit
import pandas as pd
# import datetime
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description='Find known RFI events in filterbank file.',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--path',
#                     dest='path',
#                     type=str, action='store',
#                     default='',
#                     help='Path to the fil file')
# args = parser.parse_args()
# path = ''
# if args.path != '':
#     path = args.path
# if path == '':
#     raise RuntimeError('Could not find a path to use.')

# time it
# tic = timeit.default_timer()

fil_rfiObs = rfiObs.RfiObservation(path=path,
                                   fil_file=True)
header = fil_rfiObs.file.header
nchans = header.nchans
# TODO download from google docs
# read RFI database
# csv_path = '/home/RFI_Spectrum_Database.csv'
rfiDb = rfiDatabase.RfiDatabase()
rfiDb.write_dict([csv_path])
bands = rfiDb.dictionary.keys()
int_bands = [rfiDb.dictionary[k]['band'] for k in rfiDb.dictionary.keys()]
int_dict = dict(zip(int_bands, rfiDb.dictionary.keys()))

# save number of blocks vector
# create vector with start time
# this need some logic dependent on file size
# TODO
vec_length = 100
# find median value for threshold
val = fil_rfiObs.rfi_median(vec_length)
# val = rfi_median(vec_length)

training_set = pd.DataFrame(columns=('event',
                                     'c_freq',
                                     'bw',
                                     't_start',
                                     'duration',
                                     'culprit',
                                     'description',
                                     'band'))
start_vector, duration = fil_rfiObs.time_vector(vec_length)

temp_set = training_set
corrupted_samples = 0
for sv in range(vec_length):
    block, num_sam = fil_rfiObs.read_time_freq(start_vector[sv], duration)
    mask = block < val
    open_img = rfiUtils.open_blob(mask)
    close_img = rfiUtils.close_blob(open_img)
    labeled_array, num_features = measurements.label(close_img)

    # for total corrupted data
    # needs separate function
    corrupt_block = np.count_nonzero(close_img == 1)
    corrupted_samples += corrupt_block

    feature_range = np.linspace(1, num_features, num=num_features)
    # create events here
    # initialise events
    rfi_evs = [rfiEvent.RfiEvent(ev, labeled_array) for ev in feature_range]
    # assign bw
    foff = header.foff
    freqs = fil_rfiObs.freqs
    t_df = fil_rfiObs.time[1] - fil_rfiObs.time[0]
    temp_t = fil_rfiObs.time
    [ev.finetune_attr(foff, freqs, t_df, temp_t) for ev in rfi_evs]
    [ev.find_bands(int_dict) for ev in rfi_evs]
    [ev.find_culprit(rfiDb.dictionary, int_dict) for ev in rfi_evs]
    print(sv)

# write CSV
training_set.to_csv('training_set.csv')


# sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended
# write report
def percentage(part, whole):
    return 100 * float(part)/float(whole)

file_sam = vec_length * num_sam * nchans
rfi_sam = percentage(corrupted_samples, file_sam)

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('rfi_report.pdf') as pdf:
    plt.rc('text', usetex=False)
    plt.figure(figsize=(5, 5))
    labels = 'Good', 'RFI'
    rfi_perc = rfi_sam * 100
    good_perc = 100 - rfi_perc
    sizes = [good_perc, rfi_perc]
    explode = (0, 0.1)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,
            labels=labels,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)
    ax1.axis('equal')
    plt.title('Cleanliness of the observation')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=False)
    plt.figure(figsize=(5, 5))
    training_set.groupby('culprit')['event'].nunique().plot.pie(autopct='%1.1f%%',
                                                        labels=['unknown',
                                                                'known'])
    plt.ylabel('')
    plt.title('Culprit classyfication')
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(5, 5))
    training_set.loc[lambda df: df.culprit > 0, :].groupby('description')['event'].nunique().plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title('Known culprit occurences')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(5, 5))
    training_set.loc[lambda df: df.culprit > 0, :].groupby('description')['duration'].sum().plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title('Known culprit time occupancy')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'RFI Report'
    d['ModDate'] = datetime.datetime.today()

# toc = timeit.default_timer()
# print('elapsed time in minutes %f' % (toc - tic)/60.0)
