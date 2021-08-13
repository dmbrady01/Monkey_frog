#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
signal_processing.py: Python script that contains functions for signal
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "09 Mar 2018"


import scipy.signal as ssp
import scipy.optimize as op
import numpy as np
import neo
import quantities as pq
import copy as cp
import types
import pandas as pd
from neo.core import AnalogSignal
from copy import deepcopy

def TruncateSignal(signal, start=0, end=0):
    """Given an AnalogSignal object, will remove the first 'start' seconds and the 
    last 'end' seconds. Please note that this is different logic
    than TruncateEvent! TruncateSignals trims start/end seconds from the ends no matter
    the length. TruncateEvent start/end truncates at those specific timestamps. It
    is not relative."""
    # Makes sure signal is an AnalogSignal object
    if not isinstance(signal, neo.core.AnalogSignal):
        raise TypeError('%s must be an AnalogSignal object.' % signal)
    # converts start and end to times
    start = start * pq.s 
    end = end * pq.s
    # Truncate signal
    truncated_signal = signal.time_slice(signal.t_start + start, signal.t_stop - end)

    return truncated_signal

def TruncateSignals(signal_list, start=0, end=0):
    """Given a list of AnalogSignal objects, will iterate through each one and remove
    values before 'start' and after 'end'. Start and end must be in seconds."""
    # Makes sure a list is passed
    if not isinstance(signal_list, list):
        raise TypeError('%s must be a list' % signal_list)
    # Iterate through each item with TrunacteSignal
    truncated_list = [TruncateSignal(sig, start=start, end=end) for sig in signal_list]
    
    return truncated_list

def ButterFilterDesign(lowcut=None, highcut=None, fs=381.469726562, order=5, 
                    btype='lowpass'):
    """Constructs the numerator and denominator for the low/high/bandpass/bandstop
    filter. low and high are the cutoff frequencies, fs is the sampling frequency, 
    order is the desired filter order, and btype designates what type of filter will
    ultimately be constructed (lowpass, highpass, bandpass, bandstop)."""
    # calculate nyquist frequency (half the sampling frequency)
    nyquist = fs * 0.5
    # try constructing normalized lowcut
    try:
        norm_low = lowcut / nyquist
    except TypeError:
        pass
    # try constructing normalized highcut
    try:
        norm_high = highcut / nyquist
        # Makes sure highcut is lower than nyquist frequency
        if norm_high > 1:
            raise ValueError('%s is larger than the nyquist frequency. Choose a smaller highcut (at most half the sampling rate).' % highcut)
    except TypeError:
        pass
    # Construct param to be bassed into butter
    if btype == 'lowpass':
        params = norm_high
    elif btype == 'highpass':
        params = norm_low
    elif (btype == 'bandpass') or (btype == 'bandstop'):
        params = [norm_low, norm_high]
    else:
        raise ValueError("%s must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'" 
            % btype)
    # Return butter
    return ssp.butter(order, params, btype=btype, analog=False)

def FilterSignal(signal, lowcut=None, highcut=None, fs=381, order=5, 
        btype='lowpass', axis=0, window_length=3001, savgol_order=1):
    """Given some signal data, uses ButterFilterDesign or savgol_filter to 
    construct a filter and applies it to signal.
    signal: data signal
    lowcut: lower bound for filter
    highcut: upper bound for filter
    fs: sampling frequnecy
    order: filter order
    btype: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'savgol'
    axis: axis of matrix to apply filter
    offset: constant added to entire signal at end (to prevent divide by 0 errors
    later)"""
    # If btype is savgol, performs a Savitzky-Golay filter
    if btype == 'savgol':
        return ssp.savgol_filter(signal, window_length, savgol_order, axis=axis)
    # Otherwise performs filfilt (backwards and forwards filtering)
    else:
        b, a = ButterFilterDesign(lowcut=lowcut, highcut=highcut, fs=fs, 
                                order=order, btype=btype)
        return ssp.filtfilt(b, a, signal, axis=axis)

def DeltaFOverF(signal, reference=None, period=None, mode='median', offset=0):
    """Calcualte DeltaF/F for a signal. Units are %. 
    
    Offset will first add offset to signal (and reference if provided) 
    to prevent divide by 0 errors. Note that this will change the output of
    deltaf/f (10-9)/9 is not the same as (1010-1009)/1009. So use small values.

    There are several modes:
    'median': (signal - median(signal))/median(signal)

    'mean': (signal - mean(signal))/mean(signal)
    
    'reference': (signal - reference_signal)/reference_signal
    Note that reference must be a same length array as signal
    
    'period_mean': ref = mean(signal[period[0]:period[1]] 
                   (signal - ref)/ref
    
    'period_median': same as 'period_mean' but uses median instead
    Note that period must be a list or tuple of beginning and end timstamps"""
    signal = signal + offset
    if mode == 'median':
        return (signal - np.median(signal))/np.median(signal) * 100.0
    elif mode == 'mean':
        return (signal - np.mean(signal))/np.mean(signal) * 100.0
    elif mode == 'reference':
        reference = reference + offset
        return (signal - reference)/reference * 100.0
    elif mode == 'period_mean':
        reference = signal[period[0]:period[1]]
        return (signal - np.mean(reference))/np.mean(reference) * 100.0
    elif mode == 'period_median':
        reference = signal[period[0]:period[1]]
        return (signal - np.median(reference))/np.median(reference) * 100.0
    elif mode == 'z_score_period':
        signal = ZScoreCalculator(signal, baseline_start=period[0], baseline_end=period[1])
        return signal
    elif mode == 'z_score_rolling':
        shape = signal.shape
        series_signal = pd.Series(signal.flatten().copy())
        moving_mean = series_signal.rolling(period).mean()
        moving_std = series_signal.rolling(period).std()
        z_score = (series_signal - moving_mean).divide(moving_std)
        return z_score.values.reshape(shape)
    elif mode == 'z_score':
        return signal
    elif mode == 'no_deltaf_or_zscore':
        return signal
    else:
        raise ValueError('%s is not an accepted mode for calculating deltaf' 
            % mode)

def ZScoreCalculator(signal, baseline_start=None, baseline_end=None):
    mean_baseline = np.mean(signal[baseline_start:baseline_end], axis=0)
    std_baseline = np.std(signal[baseline_start:baseline_end], axis=0)
    z_score = (signal - mean_baseline) / std_baseline
    return z_score

def SmoothSignal(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("SmoothSignal only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+ window +'(window_len)')

    y = np.convolve(w/w.sum(), s , mode='valid')
    y = y[(int(window_len/2)-1):-int(window_len/2)]
    return y[:x.shape[0]]

def SmoothSignalWithPeriod(x, sampling_rate=None, ms_bin=None, window='flat'):
    "Given a signal and sampling period, will smooth to the nearest ms bin."
    # Convert ms to seconds
    seconds = ms_bin/1000.

    num_bins = int(seconds * sampling_rate)
    if num_bins // 2 == 0:
        num_bins += 1

    if isinstance(x, np.ndarray):
        return SmoothSignal(x, window_len=num_bins, window=window)
    elif isinstance(x, pd.core.series.Series):
        values = SmoothSignal(x.values, window_len=num_bins, window=window)
        series = pd.Series(values, index=x.index)
        series.index.name = x.index.name
        return series

def PolyfitWindow(reference, signal=None, window_length=3001, return_projection=False):
    """
    Applies a polynomial fit in a sliding window to an input signal. This has the same functionality as a 
    Savitzky-Golay filter. 
    """

    if not isinstance(signal, type(None)):
        x = reference
        y = signal
    else:
        x = np.arange(reference.shape[0])
        y = reference


    if window_length is not None:
        shape = y.shape
        num_samples = len(x)
        idx = np.arange(window_length)
        x_out = np.zeros(num_samples)
        steps = np.arange(0, num_samples, window_length)

        x = x.flatten()
        y = y.flatten()

        for step in steps:
            x_frame = x[step:step+window_length]
            y_frame = y[step:step+window_length]
            p = np.polyfit(x_frame, y_frame, deg=1)
            x_out[step:step+window_length] = np.polyval(p, x_frame)

        if return_projection:
            return x_out.reshape(shape)
        else:
            return (y - x_out).reshape(shape)
    else:
        fits = np.array([np.polyfit(x[:, i], y[:, i],1) for i in range(y.shape[1])])
        Y_fit_all = np.array([np.polyval(fits[i], x[:,i]) for i in np.arange(x.shape[1])]).T
        Y_df_all = y - Y_fit_all
        return Y_df_all



def ExponentialFitWindow(reference, signal=None, window_length=3001, return_projection=False):
    """Tries to fit an exponential decay across the window_length"""
    if not isinstance(signal, type(None)):
        x = reference
        y = signal
    else:
        x = np.arange(reference.shape[0])
        y = reference

    shape = y.shape
    num_samples = len(x)
    idx = np.arange(window_length)
    x_out = np.zeros(num_samples)
    steps = np.arange(0, num_samples, window_length)

    x = x.flatten()
    y = y.flatten()

    def model_func(t, A, K, C):
        return A * np.exp(-K * t) + C

    def fit_exp_nonlinear(t, y):
        opt_parms, parm_cov = op.curve_fit(model_func, t, y, maxfev=100000)
        A, K, C = opt_parms
        return A, K, C

    for step in steps:
        x_frame = x[step:step+window_length]
        y_frame = y[step:step+window_length]
        A, K, C = fit_exp_nonlinear(x_frame, y_frame)
        x_out[step:step+window_length] = model_func(x_frame, A, K, C)

    if return_projection:
        return x_out.reshape(shape)
    else:
        return (y - x_out).reshape(shape)

def NormalizeSignal(signal=None, reference=None, **kwargs):
    """The current method for correcting a signal. These are the steps:
    1) Lowpass filter signal and reference
    2) Calculate deltaf/f for signal and reference
    3) Detrend deltaf/f using a savgol filter (if detrend is True)
    4) Subtract reference from signal

    There are no code tests for this since it is likely to change when you come
    up with a better strategy."""
    # Default options
    options = {
        'signal_btype': 'lowpass', 
        'signal_highcut': 40.0, 
        'signal_lowcut': None,
        'signal_order': 5,
        'signal_window_length': 3001,
        'signal_savgol_order': 1,
        'reference_btype': 'lowpass', 
        'reference_highcut': 40.0, 
        'reference_lowcut': None,
        'reference_order': 5,
        'reference_window_length': 3001,
        'reference_savgol_order': 1,
        'detrend': 'no_detrend',
        'second_detrend': 'none',
        'detrend_from_reference': True,
        'subtract': False,
        'mode': 'no_deltaf_or_zscore',
        'period': 3001,
        'offset': 0,
        'axis': 0,
        'fs': 381
        }
    # Update based on kwargs
    options.update(kwargs)

    all_signals = {}
    ##### Filter the signals
    # Pass signal and reference through filters
    filt_signal = FilterSignal(signal, lowcut=options['signal_lowcut'], 
        highcut=options['signal_highcut'], fs=options['fs'], order=options['signal_order'], 
        btype=options['signal_btype'], axis=options['axis'])

    # for reference
    if not isinstance(reference, type(None)):
        filt_ref = FilterSignal(reference, lowcut=options['reference_lowcut'], 
            highcut=options['reference_highcut'], fs=options['fs'], order=options['reference_order'], 
            btype=options['reference_btype'], axis=options['axis'])
    else:
        filt_ref = None
        

    # # Scale the signals to have std of 1
    # if filt_signal.std() < 10.:
    #     factor = 1. /filt_signal.std()
    #     filt_signal = filt_signal * factor
    #     filt_ref = filt_ref * factor
    all_signals['filtered_signal'] = filt_signal
    all_signals['filtered_reference'] = filt_ref
    # Determine how to detrend
    if not options['detrend_from_reference']:
        detrend_ref = filt_signal
        detrend_sig = None
    else:
        detrend_ref = filt_ref
        detrend_sig = filt_signal
    ##### Calculate deltaf/f
    if 'z_score' in options['mode']:
        # If doing z-score, first detrend, then filter
        
        ####### Detrend the signal
        if options['detrend'] == 'savgol':
            # for signal
            trend_sig = FilterSignal(filt_signal, fs=options['fs'], btype='savgol', 
                axis=options['axis'], window_length=options['signal_window_length'], 
                savgol_order=options['signal_savgol_order'])
            filt_signal = filt_signal - trend_sig
            # for reference
            if not isinstance(reference, type(None)):
                trend_ref = FilterSignal(filt_ref, fs=options['fs'], btype='savgol', 
                    axis=options['axis'], window_length=options['reference_window_length'], 
                    savgol_order=options['reference_savgol_order'])
                filt_ref = filt_ref - trend_ref
        elif options['detrend'] == 'linear':
            filt_signal = PolyfitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = PolyfitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
        elif options['detrend'] == 'decay':
            filt_signal = ExponentialFitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = ExponentialFitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
        elif options['detrend'] == 'savgol_from_reference':
            filt_ref = filt_ref - np.median(filt_ref)
            filt_signal = filt_signal - np.median(filt_signal)
            trend_ref = FilterSignal(filt_ref, fs=options['fs'], btype='savgol', 
                axis=options['axis'], window_length=options['reference_window_length'], 
                savgol_order=options['reference_savgol_order'])
            filt_signal = filt_signal - trend_ref
            filt_ref = filt_ref - trend_ref

        # Second detrending
        if options['second_detrend'] == 'linear':
            filt_signal = PolyfitWindow(filt_signal, signal=None, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = PolyfitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
        elif options['second_detrend'] == 'decay':
            filt_signal = ExponentialFitWindow(filt_signal, signal=None, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = ExponentialFitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
        
        all_signals['detrended_signal'] = filt_signal 
        all_signals['detrended_reference'] = filt_ref

        ################ Subtract signals
        if options['subtract']:
            filt_signal = filt_signal - filt_ref
        else:
            filt_signal = filt_signal
        all_signals['subtracted_signal'] = filt_signal
        
        ############## Calculate z-score
        filt_signal = DeltaFOverF(filt_signal, reference=filt_ref, 
            mode=options['mode'], period=options['period'], offset=options['offset'])
        if not isinstance(reference, type(None)):
            filt_ref = DeltaFOverF(filt_ref, reference=filt_ref, 
                mode=options['mode'], period=options['period'], offset=options['offset'])
        else:
            filt_ref = 0
        all_signals['zscore_signal'] = filt_signal
        all_signals['zscore_reference'] = filt_ref

        final_signal = filt_signal
        final_reference = filt_ref
    
    else:
        ################ Calculate deltaf/f
        filt_signal = DeltaFOverF(filt_signal, reference=filt_ref, 
            mode=options['mode'], period=options['period'], offset=options['offset'])
        if not isinstance(reference, type(None)):
            filt_ref = DeltaFOverF(filt_ref, reference=filt_ref, 
                mode=options['mode'], period=options['period'], offset=options['offset'])
        else:
            filt_ref = 0
        all_signals['deltaf_signal'] = filt_signal
        all_signals['deltaf_reference'] = filt_ref
        
        ####### Detrend the signal
        if options['detrend'] == 'savgol':
            # for signal
            trend_sig = FilterSignal(filt_signal, fs=options['fs'], btype='savgol', 
                axis=options['axis'], window_length=options['signal_window_length'], 
                savgol_order=options['signal_savgol_order'])
            filt_signal = filt_signal - trend_sig
            # for reference
            if not isinstance(reference, type(None)):
                trend_ref = FilterSignal(filt_ref, fs=options['fs'], btype='savgol', 
                    axis=options['axis'], window_length=options['reference_window_length'], 
                    savgol_order=options['reference_savgol_order'])
                filt_ref = filt_ref - trend_ref
        elif options['detrend'] == 'linear':
            filt_signal = PolyfitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = PolyfitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
            # trend_ref = FilterSignal(filt_ref, fs=options['fs'], btype='savgol', 
            #     axis=options['axis'], window_length=options['window_length'], 
            #     savgol_order=options['savgol_order'])
            # filt_ref = filt_ref - trend_ref
        elif options['detrend'] == 'decay':
            filt_signal = ExponentialFitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            filt_ref = ExponentialFitWindow(filt_ref, signal=None,
                window_length=options['reference_window_length'], return_projection=False)
        all_signals['detrended_signal'] = filt_signal 
        all_signals['detrended_reference'] = filt_ref

        ## Subtract signals
        if options['subtract']:
            filt_signal = filt_signal - filt_ref
        else:
            filt_signal = filt_signal
        all_signals['subtracted_signal'] = filt_signal

        final_signal = filt_signal
        final_reference = filt_ref

    # Return signal with reference subtracted out
    return final_signal, final_reference, all_signals

def ProcessSignalData(seg=None, sig_ch='LMag 1', ref_ch='LMag 2', 
    name='DeltaF_F_or_Z_score', **kwargs):
    """Given a segment object, it will extract the analog signal channels specified as
    signal (sig_ch) and reference (ref_ch), and will perform NormalizeSignal on them.
    Will append the new signal to the segment object as 'name'."""
    # Gets any keyword arguments
    options = kwargs
    # Check that segment object was passed
    if not isinstance(seg, neo.core.Segment):
        raise TypeError('%s must be a Segment object' % seg)
    # Retrieves signal and reference
    if sig_ch:
        try:
            signal = [x for x in seg.analogsignals if x.name == sig_ch][0]
        except IndexError:
            raise ValueError('There is no signal channel named %s' % sig_ch)

    if ref_ch:
        try:
            reference = [x for x in seg.analogsignals if x.name == ref_ch][0]
        except IndexError:
            raise ValueError('There is no reference channel named %s' % ref_ch)
    else:
        reference = None
    # Build a new AnalogSignal based on the other ones
    new_signal, ref_signal, all_signals = NormalizeSignal(signal=signal, reference=reference, **options)
    units = pq.percent # new units are in %
    t_start = signal.t_start # has the same start time as all other analog signal objects in segment
    fs = signal.sampling_rate # has the same sampling rate as all other analog signal objects in segment
    
    ###### Add deltaf_f channel ##################
    # Creates new AnalogSignal object
    deltaf_f = AnalogSignal(new_signal, units=units, t_start=t_start, sampling_rate=fs, name='DeltaF_F_or_Z_score')
    deltaf_f_ref = AnalogSignal(ref_signal, units=units, t_start=t_start, sampling_rate=fs, name='DeltaF_F_or_Z_score_reference')
    # Adds processed signal back to segment
    seg.analogsignals.append(deltaf_f)
    seg.analogsignals.append(deltaf_f_ref)

    ############ Add detrended signal #######################
    # Creates new AnalogSignal object
    detrended = AnalogSignal(all_signals['detrended_signal'], units=units, t_start=t_start, sampling_rate=fs, name='Detrended')
    seg.analogsignals.append(detrended)
    
    ######### Add filtered signal ########################
    # Creates new AnalogSignal object
    filtered = AnalogSignal(all_signals['filtered_signal'], units=units, t_start=t_start, sampling_rate=fs, name='Filtered_signal')
    filtered_ref = AnalogSignal(all_signals['filtered_reference'], units=units, t_start=t_start, sampling_rate=fs, name='Filtered_reference')
    seg.analogsignals.append(filtered)
    seg.analogsignals.append(filtered_ref)
    
    ############ Add subtracted signal #######################
    # Creates new AnalogSignal object
    subtracted = AnalogSignal(all_signals['subtracted_signal'], units=units, t_start=t_start, sampling_rate=fs, name='Subtracted')
    seg.analogsignals.append(subtracted)

    return all_signals

def SingleStepProcessSignalData(data=None, process_type='filter', input_sig_ch='LMag 1', 
    input_ref_ch='LMag 2', datatype='segment', **kwargs):

    options = {
        'signal_btype': 'lowpass', 
        'signal_highcut': 40.0, 
        'signal_lowcut': None,
        'signal_order': 5,
        'signal_window_length': 3001,
        'signal_savgol_order': 1,
        'reference_btype': 'lowpass', 
        'reference_highcut': 40.0, 
        'reference_lowcut': None,
        'reference_order': 5,
        'reference_window_length': 3001,
        'reference_savgol_order': 1,
        'detrend': 'no_detrend',
        'detrend_from_reference': True,
        'subtract': False,
        'mode': 'no_deltaf_or_zscore',
        'period': 3001,
        'offset': 0,
        'axis': 0,
        'fs': 381
        }
    # Update based on kwargs
    options.update(kwargs)

    options = deepcopy(options)

    if datatype == 'segment':
        # Retrieves signal and reference
        if input_sig_ch:
            try:
                signal = [x for x in data.analogsignals if x.name == input_sig_ch][-1]
            except IndexError:
                raise ValueError('There is no input signal channel named %s' % input_sig_ch)

        if input_ref_ch:
            try:
                reference = [x for x in data.analogsignals if x.name == input_ref_ch][-1]
            except IndexError:
                raise ValueError('There is no input reference channel named %s' % input_ref_ch)
        else:
            reference = None

        units = pq.V # new units are in %
        t_start = signal.t_start # has the same start time as all other analog signal objects in segment
        fs = signal.sampling_rate # has the same sampling rate as all other analog signal objects in segment
        signal = signal.magnitude
        reference = reference.magnitude
    else:
        signal = data[input_sig_ch]
        if input_ref_ch is not None:
            reference = data[input_ref_ch].values
        else:
            reference = None
        index = signal.index
        if (process_type == 'measure') and ('period' in options['mode']):
            measure_period = np.where((signal.index >= options['period'][0]) & (signal.index <= options['period'][1]))[0]
            options['period'][0] = measure_period[0]
            options['period'][1] = measure_period[-1]
        columns = signal.columns
        signal = signal.values

    if process_type == 'filter':
        ##### Filter the signals
        # Pass signal and reference through filters
        signal = FilterSignal(signal, lowcut=options['signal_lowcut'], 
            highcut=options['signal_highcut'], fs=options['fs'], order=options['signal_order'], 
            btype=options['signal_btype'], axis=options['axis'])
        new_sig_ch_name = 'filtered_signal'
        if not isinstance(reference, type(None)):
            reference = FilterSignal(reference, lowcut=options['reference_lowcut'], 
                highcut=options['reference_highcut'], fs=options['fs'], order=options['reference_order'], 
                btype=options['reference_btype'], axis=options['axis'])
            new_ref_ch_name = 'filtered_reference'
        else:
            new_ref_ch_name = None

    elif process_type == 'detrend':
        if not options['detrend_from_reference']:
            detrend_ref = signal
            detrend_sig = None
        else:
            detrend_ref = reference
            detrend_sig = signal
        
        ####### Detrend the signal
        if options['detrend'] == 'savgol':
            # for signal
            trend_sig = FilterSignal(signal, fs=options['fs'], btype='savgol', 
                axis=options['axis'], window_length=options['signal_window_length'], 
                savgol_order=options['signal_savgol_order'])
            signal = signal - trend_sig 
            # for reference
            if not isinstance(reference, type(None)):
                trend_ref = FilterSignal(reference, fs=options['fs'], btype='savgol', 
                    axis=options['axis'], window_length=options['reference_window_length'], 
                    savgol_order=options['reference_savgol_order'])
                reference = reference - trend_ref 
        
        elif options['detrend'] == 'linear':
            signal = PolyfitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            # reference = PolyfitWindow(reference, signal=None,
            #     window_length=options['reference_window_length'], return_projection=False)
        
        elif options['detrend'] == 'decay':
            signal = ExponentialFitWindow(detrend_ref, signal=detrend_sig, 
                window_length=options['signal_window_length'], return_projection=False)
            # filt_signal = PolyfitWindow(filt_signal, signal=None, 
            #     window_length=options['signal_window_length'], return_projection=False)
            # reference = ExponentialFitWindow(reference, signal=None,
            #     window_length=options['reference_window_length'], return_projection=False)
        
        elif options['detrend'] == 'savgol_from_reference':
            reference = reference - np.median(reference) 
            signal = signal - np.median(signal) 
            trend_ref = FilterSignal(reference, fs=options['fs'], btype='savgol', 
                axis=options['axis'], window_length=options['reference_window_length'], 
                savgol_order=options['reference_savgol_order'])
            signal = signal - trend_ref 
            reference = reference - trend_ref 

        new_sig_ch_name = 'detrended_signal'
        if not isinstance(reference, type(None)):
            new_ref_ch_name = 'detrended_reference'
        else:
            new_ref_ch_name = None

    # elif process_type == 'subtract':
    #     signal = signal - reference
    #     new_sig_ch_name = 'subtracted_signal'
    #     new_ref_ch_name = None

    elif process_type == 'measure':
        units = pq.percent
        signal = DeltaFOverF(signal, reference=reference, 
            mode=options['mode'], period=options['period'], offset=options['offset'])
        new_sig_ch_name = 'measure_signal'
        if not isinstance(reference, type(None)):
            reference = DeltaFOverF(reference, reference=reference, 
                mode=options['mode'], period=options['period'], offset=options['offset'])
            new_ref_ch_name = 'measure_reference'
        else:
            new_ref_ch_name = None

    if datatype == 'segment':   
        # Add processed to channels
        new_sig_ch = AnalogSignal(signal, units=units, t_start=t_start, sampling_rate=fs, name=new_sig_ch_name)
        data.analogsignals.append(new_sig_ch)

        if new_ref_ch_name:
            new_ref_ch = AnalogSignal(reference, units=units, t_start=t_start, sampling_rate=fs, name=new_ref_ch_name)
            data.analogsignals.append(new_ref_ch)
    else:
        signal = pd.DataFrame(signal, index=index, columns=columns)
        if not isinstance(reference, type(None)):
            reference = pd.DataFrame(reference, index=index, columns=columns)

    if not isinstance(reference, type(None)):
        return signal, reference
    else:
        return signal, None

def Downsample(signal, window, index_col='index'):
    signal.reset_index(inplace=True)
    sample = (signal.index.to_series() / window).astype(int)
    signal = signal.groupby(sample).mean()
    signal = signal.set_index(index_col)
    return signal 

def GetImagingDataIndices(start=None, end=None, sampling_rate=1):
    return (int(start*sampling_rate), int(end*sampling_rate))

