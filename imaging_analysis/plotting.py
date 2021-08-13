#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plotting.py: Python script that contains functions for plotting data.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "07 Mar 2018"

import matplotlib.pyplot as plt
import seaborn as sns
import os
from imaging_analysis.signal_processing import SmoothSignalWithPeriod
import pandas as pd
import types

def PlotAverageSignal(traces, reference=None, mode='raw', events=[0, 5], sem=True, save=True, 
        title=None, color='b', alpha=0.1, dpath='', xmin=None, xmax=None, plot_ref=True,
        ymin=None, ymax=None, smoothing_window=None, sampling_frequency=None):
    """Given a dataframe of traces (mode='raw') or an average trace (mode='avg'). 
    It will draw the average trace +/- the sem (if sem=True) or the sd (sem=False).
    events is a list of times when there should be vertical lines (trial start, 
    stimulus onset, end, etc.)"""
    plt.figure()
    if mode == 'raw':
        avg = traces.mean(axis=1)
        if sem:
            error = traces.sem(axis=1)
        else:
            error = traces.sd(axis=1)
    elif mode == 'avg':
        avg = traces['avg']
        if sem:
            error = traces['se']
        else:
            error = traces['sd']

    if not isinstance(reference, type(None)):
        if mode == 'raw':
            avg_ref = reference.mean(axis=1)
            if sem:
                error_ref = reference.sem(axis=1)
            else:
                error_ref = reference.sd(axis=1)
        elif mode == 'avg':
            avg_ref = reference['avg']
            if sem:
                error_ref = reference['se']
            else:
                error_ref = reference['sd']      

    if smoothing_window is not None:
        avg = SmoothSignalWithPeriod(x=avg.values, sampling_rate=sampling_frequency, 
            ms_bin=smoothing_window, window='flat')
        error = SmoothSignalWithPeriod(x=error.values, sampling_rate=sampling_frequency, 
            ms_bin=smoothing_window, window='flat')

    plt.plot(traces.index, avg, color=color, label='Signal')
    plt.fill_between(traces.index, avg-error, avg+error, color=color, alpha=alpha)

    if plot_ref:
        plt.plot(reference.index, avg_ref, color='r', label='Signal')
        plt.fill_between(reference.index, avg_ref-error_ref, avg_ref+error_ref, color='r', alpha=alpha)
    
    if len(events) > 0:
        [plt.axvline(event, color='k', linestyle='--') for event in events]
    
    if xmin:
        plt.xlim(xmin=xmin)
    if xmax:
        plt.xlim(xmax=xmax)
    if ymin:
        plt.ylim(ymin=ymin)
    if ymax:
        plt.ylim(ymax=ymax)

    if title is not None:
        plt.title(title)

    plt.legend(['signal', 'reference'])

    if save:
        if len(dpath) == 0:
            dpath = os.getcwd()
        dpath = dpath + os.sep + title
        plt.savefig(dpath + '.pdf')
    plt.close()

def PlotRaster():
    test['grouping'] = (test.index.to_series()/191.).astype(int)
    t = test.groupby('grouping').mean()
    t = t.set_index('index')
    sns.heatmap(t.T)

def PlotDeltaFOverF(analogsignal, save=True, title=None, color='b', alpha=0.1, 
    dpath='', xmin=None, xmax=None, ymin=None, ymax=None, smoothing_window=None, 
    sampling_frequency=None, save_csv=True):
    plt.figure()

    mag = analogsignal.magnitude[:, 0]
    times = analogsignal.times

    if smoothing_window is not None:
        mag = SmoothSignalWithPeriod(x=mag, sampling_rate=sampling_frequency, 
            ms_bin=smoothing_window, window='flat')


    plt.plot(times, mag, color=color)
    
    if xmin:
        plt.xlim(xmin=xmin)
    if xmax:
        plt.xlim(xmax=xmax)
    if ymin:
        plt.ylim(ymin=ymin)
    if ymax:
        plt.ylim(ymax=ymax)

    if title is not None:
        plt.title(title)

    if save:
        if len(dpath) == 0:
            dpath = os.getcwd()
        dpath = dpath + os.sep + title
        plt.savefig(dpath + '.pdf')
    plt.close()

    if save_csv:
        to_csv = pd.DataFrame(data=mag, index=times, columns=['DeltaF_F'])
        to_csv.index.name = 'Time'
        to_csv.to_csv(dpath + '.csv')
