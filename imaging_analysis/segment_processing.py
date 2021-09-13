#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segment_processing.py: Python script that contains functions for segment
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "11 Jun 2018"


import quantities as pq
import neo
import pandas as pd
import os
import numpy as np
from warnings import warn
from imaging_analysis.signal_processing import ZScoreCalculator

def TruncateSegment(segment, start=0, end=0, clip_same=True, evt_start=None, evt_end=None):
    """Given a Segment object, will remove the first 'start' seconds and the 
    last 'end' seconds for all events and analog signals in that segment. If not
    clip_same, specify evt_start and evt_end if you want events to be trimmed
    differently."""
    from imaging_analysis.event_processing import TruncateEvents
    from imaging_analysis.signal_processing import TruncateSignals
    # Makes sure signal is an AnalogSignal object
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a Segment object.' % segment)
    # Truncate signals first
    segment.analogsignals = TruncateSignals(segment.analogsignals, 
        start=start, end=end)
    # if clip_same=True, it clips events before/after newly trimmed 
    # analog signal, otherwise you can specify.
    if clip_same:
        segment.events = TruncateEvents(segment.events, 
            start=segment.analogsignals[0].t_start.magnitude, 
            end=segment.analogsignals[0].t_stop.magnitude)
    else:
        segment.events = TruncateEvents(segment.events, start=evt_start, end=evt_end)

    return segment

def TruncateSegments(list_of_segments, start=0, end=0, clip_same=True, evt_start=None, evt_end=None):
    """Given a list of Segment object, will run TruncateSegment on each one."""
    # Make sure list is passed
    if not isinstance(list_of_segments, list):
        raise TypeError('%s must be a list of segment objects' % list_of_segments)
    # Run TruncateSegments
    truncated_list = [TruncateSegment(x, start=start, end=end, clip_same=clip_same, 
        evt_start=evt_start, evt_end=evt_end) for x in list_of_segments]
    return truncated_list

def AppendDataframesToSegment(segment, dataframe, names):
    """Given a segment object, will check to see if it has the correct attribute.
    If not, will create 'dataframes' attribute and append dataframe to it or cycle
    through a list of dataframes and append them all."""
    # Checks segment and dataframe types
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a segment object' % segment)
    # Check dataframe is dataframe or list of dataframes
    if isinstance(dataframe, pd.core.frame.DataFrame):
        dataframe = [dataframe]
    elif isinstance(dataframe, list) and \
        all(isinstance(x, pd.core.frame.DataFrame) for x in dataframe):
        dataframe = dataframe 
    else:
        raise TypeError('%s must be a dataframe or list of dataframes object' % dataframe)
    # Check names is string or list of strings
    if isinstance(names, str):
        names = [names]
    elif isinstance(names, list) and all(isinstance(x, str) for x in names):
        names = names
    else:
        raise TypeError('%s must be a string or list' % names)
    # checks if attribute exists, if not creates it
    if not hasattr(segment, 'dataframes'):
        segment.dataframes = {}
    # Adds dataframe to segment object
    for name, df in zip(names, dataframe):
        segment.dataframes.update({name: df})

def AppendDictToSegment(segment, dicts):
    """Given a segment object, will check to see if it has the correct attribute.
    If not, will create 'analyzed' attribute and append dictionary to it or cycle
    through a list of dictionaries and append them all."""
    # Checks segment and dataframe types
    if not isinstance(segment, neo.core.Segment):
        raise TypeError('%s must be a segment object' % segment)
    if isinstance(dicts, dict):
        dicts = [dicts]
    elif isinstance(dicts, list) and all(isinstance(x, dict) for x in dicts):
        dicts = dicts
    else:
        raise TypeError('%s must be a dictionary or list of dictionaries' % dicts)
    # checks if attribute exists, if not creates it
    if not hasattr(segment, 'analyzed'):
        segment.analyzed = {}
    # Adds dataframe to segment object
    for d in dicts:
        segment.analyzed.update(d)

def AlignEventsAndSignals(seg=None, epoch_name=None, analog_ch_name=None, 
        event_ch_name=None, event=None, event_type='type', prewindow=0, 
        postwindow=0, window_type='event', clip=False, name=None, to_csv=False,
        dpath=''):
    """Takes a segment object and spits out four dataframes:
    1) 'all_traces' - all analog traces for a specified epoch and window 
    centered at an event
    2) 'all_events' all events for a specified epoch and window centered at an event
    2) 'average_trace' average analog trace and sem for a specified epoch and window
    3) 'point_estimate' average analog value of the average analog trace and sem for a 
    specified epoch and window

    Takes the following arguments:
    1) seg: segment object to be processed. Must have processed analog signals,
    events, epochs, and dataframes
    2) epoch_name: the type of epoch to analyze (e.g. 'omission', 'correct_correct')
    3) analog_ch_name: name of the analog channel to analyze (e.g. 'deltaf_f')
    4) event: the event to align by (e.g. 'result', 'iti_start', 'start')
    5) event_class: 'label' means the exact name of an event ('iti_start',
    'tray_activated'), 'type' means the type of event specified in event_params.json
    ('results', 'start', 'end')
    6) prewindow: time in seconds to begin collecting before event or trial
    7) postwindow: time in seconds to end collecting after event or trial
    8) window_type: 'event' means the pre/post windows are only around the alignment
    event. 'trial' means the pre/post windows are collected around the entire trial
    9) clip: only relevant for window_type='trial', clip=True will only keep values
    within the specified window (even though trials may be different lengths), so
    there will be NaN values for shorter trials. clip=False will use the longest
    trial as a default and just collect the values. clip=False is more relevant
    for plotting.
    10) name: name of collection of analyzed dataframes
    11) to_csv: whether dataframes should be written to csv or only appended
    to segment.analyze

    Example:
    AlignEventsAndSignals(segment, epoch_name='correct_incorrect', 
        analog_ch_name='deltaf_f', event='start', event_type='type', 
        prewindow=2, postwindow=8, window_type='event')

    Will find all 'correct_incorrect' trials, look for the first event type called
    'start' in each trial, and collect 2 seconds before and 8 seconds after that
    event.

    Example 2:
    AlignEventsAndSignals(segment, epoch_name='incorrect', analog_ch_name='deltaf_f',
        event='tray_activated', event_type='label', prewindow=1, postwindow=1, 
        window_type='trial', clip=True)

    Will find all 'incorrect' trials, and take 1 seconds before the start of each
    trial to 1 second after each trial. The data will be aligned at tray_activated.
    Since clip=True, the dataframe will be as long as the longest trial, and so shorter
    trials will have NaN values outside of their pre/post window.
    """
    # Check segment object is passed
    if not isinstance(seg, neo.core.Segment):
        raise TypeError('seg variable must be a segment object')

    # Extract analog signal object
    try:
        signal = [x for x in seg.analogsignals if x.name == analog_ch_name][-1]
    except:
        raise ValueError("""%s not in segment object. Did you not run 
            ProcessSignalData or misspell the analog_ch_name?"""
            % analog_ch_name)

    # Extract events object
    try:
        events = [x for x in seg.events if x.name == event_ch_name][0]
    except:
        raise ValueError("""%s not in segment object. Did you not run 
            ProcessEvents or misspell the event_ch_name?"""
            % event_ch_name)

    # Extract trials dataframe
    try:
        trials = seg.dataframes['trials']
    except:
        raise ValueError("""There is no trials dataframe in the segment object.
            Did you run ProcessTrials?""")

    # Extract epoch object
    if epoch_name == 'all':
        epoch_name = trials.results.unique()
    elif isinstance(epoch_name, str):
        epoch_name = [epoch_name]

    try:
        epochs = [x for x in seg.epochs if x.name in epoch_name]
        epoch_mask = np.concatenate([epoch.times for epoch in epochs]) * pq.s
    except:
        raise ValueError("""%s not in segment object. Did you not run 
            GroupTrialsByEpoch or misspell the epoch_name?"""
            % epoch_name)

    # converts pre and post windows to seconds
    prewindow = prewindow * pq.s 
    postwindow = postwindow * pq.s

    # Makes sure event_type is correct
    if event_type == 'label':
        event_mask = 'event'
    elif event_type == 'type':
        event_mask = 'event_type'
    else:
        raise ValueError("""event_type must be either 'label' for specific events 
            (i.e. tray_activated) or 'type' for classes of events (i.e. results)""")

    # Makes sure event is in event or event_type column
    if not event in trials.loc[:, event_mask].values:
        raise ValueError("""%s is not found in trials dataframe. 
            Is event_type correct?""" % event)

    # Gets trial indices for that epoch
    trial_indices = trials.loc[trials.time.isin(epoch_mask), 'trial_idx'].unique()
    
    # Time when event of interest occured
    trial_start = np.concatenate([epoch.times for epoch in epochs]) * pq.s
    trial_end = trial_start + np.concatenate([epoch.durations for epoch in epochs]) * pq.s
    event_time = trials.loc[(trials[event_mask] == event) & \
        (trials.trial_idx.isin(trial_indices)), 'time'].unique() * pq.s

    # Make trials column names
    # Starts at 1 and increments
    trial_names = ['trial' + str(ind + 1) for ind in range(trial_indices.shape[0])]
    # Starts at the actual trial number
    #trial_names = ['trial' + str(ind) for ind in trial_indices]

    # Get time window around event
    if window_type == 'event':
        windows = np.array([event_time - prewindow, event_time + postwindow]).T * pq.s
        longest_pre_window = prewindow 
        longest_post_window = postwindow
    # Get time window around each trial
    elif window_type == 'trial':
        # Finds the trial with the longest window between pre trial and event
        longest_pre_window = np.max(event_time - (trial_start - prewindow))
        # Finds the trial with the longest window between event and post trial
        longest_post_window = np.max((trial_end + postwindow) - event_time)
        
        if clip is True:
            # if clipped then we only take values around each trial
            windows = np.array([trial_start - prewindow, 
                trial_end - postwindow]).T * pq.s
        else:
            # if not clipped then we take values around each trial according to
            # the longest pre/post windows possible
            windows = np.array([event_time - longest_pre_window, 
                event_time + longest_post_window]).T * pq.s
    
    # Calculates how many rows dataframe will be (must be event)
    pre_len = int(np.ceil((longest_pre_window.magnitude - 0)/signal.sampling_period))
    post_len = int(np.ceil((longest_post_window.magnitude - 0)/signal.sampling_period))
    # Create index starting at prewindow and ending at postwindow
    start_index = -1 * pre_len * signal.sampling_period.magnitude
    end_index = post_len * signal.sampling_period.magnitude
    len_index = pre_len + post_len + 1
    index = np.linspace(start_index, end_index, len_index)
    # creates analog signal and event dataframes
    signal_df = pd.DataFrame(np.nan, index=index, columns=trial_names)
    event_df = pd.DataFrame(np.nan, index=index, columns=trial_names)
    # list of bad trials
    bad_trials = []
    # Goes through each trial to get relevant time stamps and resets them
    for trial in range(trial_indices.shape[0]):
        # Get the timestamp of the aligning event in that trial
        centering_event = event_time[trial]
        # Get the start/end of the window
        window_start, window_end = windows[trial]
        # check if window_start/end times are in data
        if window_start < signal.times[0]:
            warn("""\nYour prewindow is too long. There is not enough signal data 
                in your first trial. Throwing first trial out. If you want to keep 
                first trial data, choose a shorter prewindow and run again.""")
            bad_trials.append(trial)
        elif window_end > signal.times[-1]:
            warn("""\nYour postwindow is too long. There is not enough signal data 
                in your last trial. Throwing last trial out. If you want to keep 
                last trial data, choose a shorter postwindow and run again.""")
            bad_trials.append(trial)           
        else:
            sig_values = signal.time_slice(window_start, window_end).magnitude 
            # get corresponding time stamps for the time signal
            sig_times = signal.time_slice(window_start, window_end).times - centering_event
            # because the sampling frequency might not exactly align with the event
            # windows (event happens at 2 seconds, but sampling is at 1.997 and 2.001 
            # seconds) we have to align them
            # make a dataframe with signal magnitude and signal timestamps
            sig_df = pd.DataFrame(sig_values, index=sig_times)
            # do a fuzzy merge that aligns event time index with sampling freq index
            aligned_sig = pd.merge_asof(signal_df, sig_df, left_index=True, 
                right_index=True).iloc[:, -1]
            # Assign signal to signal dataframe
            signal_df.iloc[:, trial] = aligned_sig 
            # get event times
            evt_times = events.time_slice(window_start, window_end).times
            if len(evt_times) > 0:
                # get labels for those events
                evt_labels = events.labels[np.isin(events.times, evt_times)]
                # Find the closest sampled index to event (since sampling frequency
                # might not match when event occurs)
                evt_indices = [(np.abs(index - x.magnitude)).argmin() for x in 
                    (evt_times - centering_event)]
                event_df.iloc[evt_indices, trial] = evt_labels
    # Get rid of bad trials
    bad_trial_mask = ~pd.Index(list(range(signal_df.columns.shape[0]))).isin(bad_trials)
    signal_df = signal_df.iloc[:, bad_trial_mask]
    event_df = event_df.iloc[:, bad_trial_mask]
    # Do forward fill and backward fill for stray NaN values if clip=False
    # These values come from the events not being totally inline with the
    # sampling frequency
    if clip is False:
        signal_df = signal_df.ffill().bfill()

    # # Calculate average signal
    # avg_df = pd.DataFrame()
    # avg_df['avg'] = signal_df.mean(axis=1)
    # avg_df['sd'] = signal_df.std(axis=1)
    # avg_df['se'] = signal_df.sem(axis=1)

    # # Calculate average of average (point estimate)
    # pe_df = pd.DataFrame(np.nan, columns=['avg', 'sd', 'se'], index=[0])
    # pe_df.loc[0, 'avg'] = avg_df.avg.mean()
    # pe_df.loc[0, 'sd'] = avg_df.sd.mean()
    # pe_df.loc[0, 'se'] = avg_df.se.sem()

    # Creates a dictionary with name or constructes name
    if not name:
        name = '_'.join([analog_ch_name, event_ch_name, event, event_type, 
            epoch_name, str(prewindow), str(postwindow), window_type, str(clip)])


    final_dict = {name: {
        'all_traces': signal_df,
        'all_events': event_df
    }}

    # final_dict = {
    #     epoch_name: 
    #         {
    #             analog_ch_name: 
    #                 {
    #                     'all_traces': signal_df,
    #                     'all_events': event_df
    #                 }
    #         }

    # }

    # final_dict = {name: {
    #     'all_traces': signal_df,
    #     'all_events': event_df,
    #     'average_trace': avg_df,
    #     'point_estimate': pe_df
    # }}

    AppendDictToSegment(seg, final_dict)

    if to_csv:
        dpath = dpath + os.sep + name
        signal_df.to_csv(dpath + '_all_traces.csv')
        event_df.to_csv(dpath + '_all_events.csv')
        # avg_df.to_csv(dpath + '_average_trace.csv')
        # pe_df.to_csv(dpath + '_point_estimate.csv')

    return event_df







