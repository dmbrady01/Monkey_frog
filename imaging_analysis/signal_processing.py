#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
signal_processing.py: Python script that contains functions for signal
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "01 Mar 2018"


import scipy.signal as ssp
import numpy as np


def NormalizeSignal(signal=None, reference=None, framelen=3001, order=1, return_filt=False):
    """Given a signal and a reference, it returns the signal - savgol_filter(reference).
    If only a signal is given, it returns the signal - savgol_filter(signal). 
    If return_filt = True, then we just return the filtered signal or reference."""
    # We determine which axis the signal is recorded (column or row vector)
    if signal.shape[0] > signal.shape[1]:
        axis = 0
    else:
        axis = 1
    # We filter the signal with a Savtizky-Golay filter
    if reference:
        filtered_signal = ssp.savgol_filter(reference, framelen, order, axis=axis)
    else:
        filtered_signal = ssp.savgol_filter(signal, framelen, order, axis=axis)
    # Returns the signal - filtered signal (or just the filtered signal)
    if return_filt:
        return filtered_signal
    else:
        return signal - filtered_signal


def SubtractNoise(signal=None, reference=None, framelen=3001, order=1):
    """Given a signal and reference, it subtracts the filtered reference from
    the signal. Both signals are median subtracted/scaled first."""
    # First we center and scale the signal/reference by the median
    median_signal = (signal - np.median(signal))/np.median(signal)
    median_reference = (reference - np.median(reference))/np.median(reference)
    # Then we filter the reference with a Savtizky-Golay filter
    normalized_sig = NormalizeSignal(median_signal, median_reference, framelen, order)
    # We return the median_subtracted_signal - the filtered_reference
    return median_signal - filt_reference




def NormalizeSignalData(seg=None, signal=['LMag 1'], reference=['LMag 2']):
    pass