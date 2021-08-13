#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
deltaf_f.py: Python script that processes fiber photometry data. It truncates
the signal, filters it, and saves the data/figure.
"""


__author__ = "DM Brady"
__datewritten__ = "09 May 2018"
__lastmodified__ = "11 May 2018"

import sys
from imaging_analysis.event_processing import LoadEventParams, ProcessEvents, ProcessTrials, GroupTrialsByEpoch
from imaging_analysis.segment_processing import TruncateSegments, AppendDataframesToSegment
from imaging_analysis.utils import ReadNeoPickledObj, ReadNeoTdt, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.signal_processing import ProcessSignalData
from imaging_analysis.plotting import PlotDeltaFOverF
#######################################################################
# VARIABLES TO ALTER
# For truncating the recordings/events
truncate_begin = 10 # how many seconds to remove from the beginning of the recording
truncate_end = 0 # how many seconds to remove from the end of the recording
# For filtering and calculating deltaf/f
lowpass_filter = 40.0 # Will remove frequencies above this number
signal_channel = 'LMag 1' # Name of our signal channel
reference_channel = 'LMag 2' # Name of our reference channel
#deltaf_ch_name = 'DeltaF_F' # New name for our processed signal channel
deltaf_options = {} # Any parameters you want to pass when calculating deltaf/f
# Save processed trials as pickle object? Honestly, its faster just to run 
# the processing again
load_pickle_object = False
save_pickle = False 
pickle_name = 'processed.pkl'

# PLOTTING parameters
save_plot = True 
plot_title = 'DeltaF_F'
color = 'b'
alpha = 0.1
plot_events = [] # times when you want a vertical line in the plot (trial start, stimulus on, etc.)
sem = True #plot standard error of the mean. if false, plots standard deviation

# Set these to numbers if you want to control axis size
xmin = None
xmax = None
ymin = None 
ymax = None

# smoothing window in msecs (None if you don't want to use it.)
smoothing_window = 500

##########################################################################
# Checks if a directory path to the data is provided, if not, will
# use what is specified in except
try:
    dpath = sys.argv[1]
except IndexError:
    dpath = '/Users/DB/Development/Monkey_frog/data/TDT-LockinRX8-22Oct2014_20-4-15_DT1_012518'

# Tries to load a processed pickle object, othewise reads the Tdt folder,
# processes the data and writes a pickle object
try:
    # Attempting to load pickled object
    if load_pickle_object:
        PrintNoNewLine('Trying to load processed pickled object...')
        seglist = ReadNeoPickledObj(path=dpath, name=pickle_name, 
                                return_block=False)
        print('Done!')
    else:
        raise IOError
except IOError:
    # Reads data from Tdt folder
    PrintNoNewLine('\nCannot find processed pkl object, reading TDT folder instead...')
    block = ReadNeoTdt(path=dpath, return_block=True)
    seglist = block.segments
    print('Done!')

    # Trunactes first/last seconds of recording
    PrintNoNewLine('Truncating signals and events...')
    seglist = TruncateSegments(seglist, start=truncate_begin, end=truncate_end, clip_same=True)
    print('Done!')

    # Iterates through each segment in seglist. Right now, there is only one segment
    for segment in seglist:
        # Extracts the sampling rate from the signal channel
        sampling_rate = [x for x in segment.analogsignals if x.name == signal_channel][0].sampling_rate
        # Appends an analog signal object that is delta F/F. The name of the channel is
        # specified by deltaf_ch_name above. It is calculated using the function
        # NormalizeSignal in signal_processing.py. As of right now it:
        # 1) Lowpass filters signal and reference (default cutoff = 40 Hz, order = 5)
        # 2) Calculates deltaf/f for signal and reference (default is f - median(f) / median(f))
        # 3) Detrends deltaf/f using a savgol filter (default window_lenght = 3001, poly order = 1)
        # 4) Subtracts reference from signal
        # NormalizeSignal has a ton of options, you can pass in paramters using
        # the deltaf_options dictionary above. For example, if you want it to be mean centered
        # and not run the savgol_filter, set deltaf_options = {'mode': 'mean', 'detrend': False}
        PrintNoNewLine('\nCalculating delta_f/f...')
        ProcessSignalData(seg=segment, sig_ch=signal_channel, ref_ch=reference_channel,
                        name='DeltaF_F', fs=sampling_rate, highcut=lowpass_filter, **deltaf_options)
        print('Done!')
        PrintNoNewLine('Plotting data...')
        signal = [x for x in segment.analogsignals if x.name == 'DeltaF_F'][0]
        PlotDeltaFOverF(signal, save=save_plot, title=plot_title, color=color, 
            alpha=alpha, dpath=dpath, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
            smoothing_window=smoothing_window, sampling_frequency=sampling_rate,
            save_csv=True)
        print('Done!')

