#!/usr/local/bin/python
import matplotlib
matplotlib.use('Agg')
from rfipip import rfiObs, rfiUtils, rfiReport
import timeit
import numpy as np
import pandas as pd
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import argparse

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

fil_rfiObs = rfiObs.RfiObservation(path=path,
                                   fil_file=True)
# TODO download from google docs
# read RFI database
csv_path = '/home/RFI_Spectrum_Database.csv'
int_dict = fil_rfiObs.read_database(csv_path)

# save number of blocks vector
# create vector with start time
# this need some logic dependent on file size
# TODO
vec_length = 100
# find median value for threshold
fil_rfiObs.rfi_median(vec_length)

fil_rfiObs.obs_events(vec_length, int_dict)
data = fil_rfiObs.write2csv(return_h5=True)

# sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended
# write report
report_name = 'rfi_report.pdf'
rfi_pdf = rfiReport.RfiReport(report_name)
rfi_sam = fil_rfiObs.percentage_rfi(vec_length)
rfi_pdf.write_report(data, rfi_sam)
