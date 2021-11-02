#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
event_processing.py: Python script that contains functions for event and epoch
processing.
"""


__author__ = "DM Brady"
__datewritten__ = "01Mar2018"
__lastmodified__ = "09Aug2018"

import quantities as pq
from collections import OrderedDict
import numbers
import neo
from neo.core import Event, Epoch
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
import json
import itertools
import re
from imaging_analysis import utils

def LoadEventParams(dpath=None, evtdict=None, mode='TTL', offset_events=None):
    """Checks that loaded event parameters (either through a directory path or
    from direct input (dpath vs evtdict)). Returns three dataframes and three lists
    in this order: startoftrial, endoftrial, epochs, event_type, plot, results
    Dataframes:
    1) 'event_type' that convert ch codes to event
    2) 'plot' that maps event to specific plotting styles
    3) 'results' that maps event to its type (start trial, result, end trial, etc.)
    Lists:
    1) 'endoftrial': event types that signal the end of a trial (default type = results)
    2) 'startoftrial': event types that signal the start of a trial (default type = start)
    3) 'epoch': event types that signal different epochs (default type = results)
    """
    # Load event params
    if dpath:
        if not os.path.exists(dpath):
            raise IOError('%s cannot be found. Please check that it exists' % dpath)
        else:
            evtdict = json.load(open(dpath, 'r'), object_pairs_hook=OrderedDict)
    else:
        if not isinstance(evtdict, dict):
            raise TypeError('%s must be a dictionary' % evtdict)
    # Constructs endoftrial list
    endoftrial = evtdict['endoftrial']
    # Constructs startoftrial list
    startoftrial = evtdict['startoftrial']
    # Constructs epoch list
    epochs = evtdict['epochs']
    # Constructs code dataframe
    if mode == 'TTL':
        code_event_pairs = [(x, y['code']) for x, y in list(evtdict['events'].items())]
        channels = evtdict['channels']
        events, codes = list(zip(*code_event_pairs))
        event_type = pd.DataFrame(data=list(codes), index=list(events), columns=channels)
        event_type.index.name = 'event'
        event_type.name = 'eventframe'
    elif mode == 'manual':
        event_type = pd.DataFrame()
    else:
        raise ValueError('mode must be "TTL" or "manual"')
    # # Constructs plotting dataframe
    # plot_event_pairs = [(x, y['plot']) for x, y in evtdict['events'].items()]
    # plot = pd.DataFrame(plot_event_pairs, columns=['event', 'plot']).set_index('event')
    # plot.name = 'plotframe'
    # Constructs results dataframe
    results_event_pairs = [(x, y['type']) for x, y in list(evtdict['events'].items())]
    results = pd.DataFrame(results_event_pairs, columns=['event', 'type']).set_index('event')
    results.name = 'resultsframe'
    # returns dataframe
    return startoftrial, endoftrial, epochs, event_type, results

def ReadManualExcelFile(excel_file):
    if isinstance(excel_file, pd.core.frame.DataFrame):
        return excel_file
    else:
        if '.csv' in excel_file:
            df = pd.read_csv(excel_file)
        else:
            df = pd.read_excel(excel_file)
        return df

def FormatManualExcelFile(excel_file, event_col='Bout type', start_col='Bout start', 
    end_col='Bout end', resample=False, resample_window=1, tolerance=0.00001):
    """Given a manual excel file, it string formats Bout types to be consistent."""
    # Read excel file
    df = ReadManualExcelFile(excel_file)

    # Check proper columns in excel file
    if event_col not in df.columns:
        raise ValueError('%s is not a column in your manual excel file. Check spelling!' 
            % event_col)

    if start_col not in df.columns:
        raise ValueError('%s is not a column in your manual excel file. Check spelling!' 
            % start_col)

    if end_col not in df.columns:
        raise ValueError('%s is not a column in your manual excel file. Check spelling!' 
            % end_col)

    # Check start and end col are filled out
    if (not is_numeric_dtype(df[start_col])) or (df[start_col].isnull().sum() > 0):
        raise ValueError('Your %s contains non-numerical or missing data. Please check it!' 
            % start_col)

    if (not is_numeric_dtype(df[end_col])) or (df[end_col].isnull().sum() > 0):
        raise ValueError('Your %s contains non-numerical or missing data. Please check it!' 
            % end_col)
    # Convert event col to be standard
    df[event_col] = df[event_col].str.strip().str.lower().apply(
        lambda x: re.sub(r"\s+", "_", x)).apply(lambda x: re.sub(r"/", "_", x))

    df[end_col] = df[end_col] - 1e-9

    if any(np.diff(InterleaveEvents(df[start_col], df[end_col])) <= -1*tolerance):
        raise ValueError('Some of your bouts overlap in time in your manual excel file. Please fix bout start/ends!')

    if resample:
        start = np.floor(df[start_col].iloc[0])
        end = np.ceil(df[end_col].iloc[-1])
        df = df.set_index(start_col).reindex(np.arange(start, end, resample_window)).ffill()

    return df

def GenerateManualEventParamsJson(dataframe, event_col='Bout type', 
        name='imaging_analysis/manual_event_params.json'):
    """From an excel file, generates a param file similar to TTL one""" 
    dataframe = FormatManualExcelFile(dataframe)
    to_json = dict()
    to_json["endoftrial"] = ["end"]
    to_json["startoftrial"] = ["start"]
    to_json["epochs"] = ["start"]
    to_json['events'] = dict()
    to_json['events']['end'] = {'type': 'end', 'plot': '-b'}
    for event in dataframe[event_col].unique():
        to_json['events'][event] = {'type': 'start', 'plot': '-k'}
    with open(name, 'w') as fp:
        json.dump(to_json, fp)

def TruncateEvent(event, start=None, end=None):
    """Given an Event object, will remove events before 'start' and after 
    'end'. Start and end must be in seconds. Please note that this is different logic
    than TruncateSignal! TruncateSignals trims start/end seconds from the ends no matter
    the length. TruncateEvent start/end truncates at those specific timestamps. It
    is not relative."""
    # Makes sure event is an Event object
    if not isinstance(event, neo.core.Event):
        raise TypeError('%s must be an Event object.' % event)
    # converts start and end to times
    if start is None:
        start = event.min()
    else:
        start = start * pq.s

    if end is None:
        end = event.max()
    else:
        end = end * pq.s

    truncated_event = event.time_slice(start, end)

    return truncated_event

def TruncateEvents(event_list, start=None, end=None):
    """Given a list of Event objects, will iterate through each one and remove
    events before 'start' and after 'end'. Start and end must be in seconds."""
    # Makes sure a list is passed
    if not isinstance(event_list, list):
        raise TypeError('%s must be a list' % event_list)
    # Iterate through each item with TrunacteEvent
    truncated_list = [TruncateEvent(evt, start=start, end=end) for evt in event_list]
    return truncated_list

def ExtractEventsToList(seg=None, evtframe=None, time_name='times', ch_name='ch'):
    """Given a segment object and an eventframe, extracts event times
    for channels specified in evtframe and returns a list of dictionaries.
    Each dictionary has the event times (time_name) and a number representing
    that channel (ch_name)"""
    # Makes sure segment object is passed
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s needs to be a neo segment object' % seg)
    # Makes sure pandas dataframe object is passed
    if not isinstance(evtframe, pd.core.frame.DataFrame):
        raise TypeError('%s needs to be a pandas DataFrame object' % evtframe)
    # Iterates through the event object in seg and pulls out the relevent ones
    eventlist = list()
    for event_obj in seg.events:
        # Checks if event object is type we care about (DIn1, DIn2, etc.)
        if event_obj.name in evtframe.columns:
            # idx to represent channel
            idx = evtframe.columns.get_loc(event_obj.name)
            # Adds dictionary to eventlist containing event times and ch
            eventlist.append({time_name: event_obj.times, ch_name: idx})
    # Makes sure eventlist has something in it
    if len(eventlist) == 0:
        raise ValueError("""Segment object does not contain event channels that
            match those specified in eventframe. Check segment object or
            event params.json""")
    # Makes sure that eventlist length matches evtframe column length
    elif len(eventlist) != evtframe.columns.shape[0]:
        raise ValueError("""Evtframe has too many or too few channels compared to
            the segment.events list. Either segment.events has duplicate entries or
            event params.json is not correct.""")
    else:
        return eventlist

def ProcessEventList(eventlist=None, tolerance=None, evtframe=None, 
    time_name='times', ch_name='ch'):
    """Given an event list of dictionaries (with times in an array associated
    with time_name and channel names associated with ch_name), performs the 
    following algorithm with a while loop:
    1) Copies event dictionary in eventlist to a new list if event dictionary times
    are not empty
    2) Takes the earliest event across the remaining event dictionaries
    3) Creates a temporary list where each element represents whether a channel
    fired at the event time in 2. Default is that the channel did not fire.
    4) Iterates through each channel to see if the earliest event in that channel
    is within the tolerance of the event timestamp from 2).
    5) If that channels event is found to be within that tolerance, it sets that
    channels position in temporary list to 1 (meaning it co-occured).
    6) Events that are determined to co-occur are purged from their respective
    event dictionary.
    7) The co-occurence list is matched against the evtframe to determine the 
    event label.
    8) An event_times list and an event_labels list are updated
    9) When the while loop is finished, event_times and event_labels are returned.
    """
    # Makes sure that tolerance is a number
    if not isinstance(tolerance, numbers.Number):
        raise TypeError('%s needs to be a number' % tolerance)
    # Converts tolerance to have units
    tolerance = tolerance * pq.s
    # Checks that eventlist is a list
    if not isinstance(eventlist, list):
        raise TypeError('%s needs to be a list')
    # Checks time_name and ch_name are keys in each eventlist dict
    time_name_check = all(time_name in list(d.keys()) for d in eventlist)
    ch_name_check = all(ch_name in list(d.keys()) for d in eventlist)
    if not (time_name_check and ch_name_check):
        raise ValueError('%s and %s must be keys in dictionaries in eventlist' 
            % (time_name, ch_name))
    event_times = list() # Will store every event timestamp
    event_labels = list() # Will store every event label
    # Systematically goes through each channel in eventlist. Takes the
    # first event from each channel. Out of those it takes the earliest one.
    # If the other events are within the tolerance of that earliest event,
    # it is marked as co-occuring. All co-occuring events are then deleted from
    # their respective channels. This is done until eventlist event times are
    # empty
    # While loops works until every event is deleted from all channels in
    # event list
    while any(event_array[time_name].size for event_array in eventlist):
        # Pulls out channels that still have events in them
        evtlist_non_empty = [x for x in eventlist if x[time_name].size]
        # Gets the first event for each channel in evtlist_non_empty
        first_elems = [x[time_name][0] for x in evtlist_non_empty]
        # Selects the earliest out of all the channels
        current_earliest = np.amin(first_elems) * pq.s 
        # Sets a list where each element represents whether a channel
        # fired an event or not. Default is that it did not
        current_event_list = np.zeros(evtframe.columns.shape[0])
        # Goes through each channel and sees if the first time stamp 
        # is within the tolerance of the earliest timestamp
        # out of all channels (current_earliest). If so, it sets that channels
        # position in current_event_list to 1 and that timestamp is deleted
        # from eventlist
        for event_ch in evtlist_non_empty:
            # Calculates if event is within tolerance
            if event_ch[time_name][0] - current_earliest <= tolerance:
                # Sets position in current_evet_list to 1
                current_event_list[event_ch[ch_name]] = 1
                # Deletes that event from the channel
                event_ch[time_name] = np.delete(event_ch[time_name], 0) * pq.s
        # Finds the correct label by matching current_event_list (which
        # could look like [0, 1, 1] for example) with the corresponding
        # dataframe row
        try:
            label = evtframe.index[evtframe.apply(lambda x: 
                all(x == current_event_list), axis=1)][0]
            event_labels.append(label)
            # Adds the timestamp to event_times
            event_times.append(current_earliest)
            # Resets eventlist to reflect event deletion
        except IndexError:
            pass
        eventlist = evtlist_non_empty
    return event_times, event_labels

def InterleaveEvents(list1, list2):
    return np.array(list(itertools.chain(*list(zip(list1, list2)))))

def SpoofEvents(dataframe, event_col='Bout type', start_col='Bout start', end_col='Bout end', offset_events=None):
    """Given a dataframe of events to spoof, adds event labels with times"""
    # Read file
    dataframe = FormatManualExcelFile(dataframe)
    offset = 0
    if isinstance(offset_events, float):
        offset = offset_events
    elif isinstance(offset_events, str):
        offset = GetImagingDataTTL(offset_events)
    # Create start events
    start_events = dataframe[event_col].values
    start_times = dataframe[start_col].values + offset
    end_times = dataframe[end_col].values + offset
    end_events = np.repeat('end', dataframe[end_col].shape[0])
    eventtimes = InterleaveEvents(start_times, end_times)
    eventlabels = InterleaveEvents(start_events, end_events)
    return eventtimes, eventlabels

def ProcessEvents(seg=None, tolerance=None, evtframe=None, name='Events', mode='TTL', 
        manualframe=None, event_col='Bout type', start_col='Bout start', end_col='Bout end', offset_events=None):
    """Takes a segment object, tolerance, and event dataframe"""
    # Makes sure that seg is a segment object
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s needs to be a neo segment object' % seg)
    # Checks if events have already been processed
    if name in [event.name for event in seg.events]:
        print("Events array has already been processed")
    else:
        if mode == 'TTL':
            # Iterates through the event object in seg and pulls out the relevent ones
            eventlist = ExtractEventsToList(seg=seg, evtframe=evtframe)
            eventtimes, eventlabels = ProcessEventList(eventlist=eventlist, 
                tolerance=tolerance, evtframe=evtframe, time_name='times', ch_name='ch')
        elif mode == 'manual':
            eventtimes, eventlabels = SpoofEvents(manualframe, event_col=event_col, 
                start_col=start_col, end_col=end_col, offset_events=offset_events)
        # Creates an Event object
        results = Event(times=np.array(eventtimes) * pq.s,
            labels=np.array(eventlabels, dtype='S'), name=name)
        # Appends event objec to segment object
        seg.events.append(results)

def ResultOfTrial(listtocheck=None, noresults='NONE', multipleresults='MULTIPLE',
        appendmultiple=False):
    """Checks the result of a trial. Assumes that a trial can only have one
    results. Trials with no results will be labeled noresults. If multiple 
    results are found, results are set to MULTIPLE. If appendmultiple, then multiple 
    results will also have the result list added.
    Examples:
    1) ['omission'] -> 'omission'
    2) [] -> 'NONE'
    3) ['omission', 'premature'] -> 'MULTIPLE'
    4) ['omission', 'premature'] and appendmultiple=True -> 'MULTIPLE_omission_premature'"""
    if not isinstance(listtocheck, list):
        raise TypeError("%s must be a list" % listtocheck)
    if len(listtocheck) == 1:
        return listtocheck[0]
    elif len(listtocheck) == 0:
        return noresults
    else:
        if appendmultiple:
            return multipleresults + '_' + '_'.join(listtocheck)
        else:
            return multipleresults

def ProcessTrials(seg=None, name='Events', startoftrial=None, epochs=None, 
        typedf=None, appendmultiple=False, firsttrial=True, returndf=True):
    """Takes a segment object, the name of Event channel to process, a list
    of events that deem start of trial, and the results/type dataframe.
    Returns a dataframe of event times, event names, trial number, and trial 
    outcome. If firsttrial, will only pass events from the first trial or later.
    If returndf=True, will return a dataframe. Either way, it will append the
    dataframe"""
    # Make sure startoftrial is passed
    if not isinstance(startoftrial, list):
        raise TypeError('%s must be a list' % startoftrial)
    # Make sure epochs is passed
    if not isinstance(epochs, list):
        raise TypeError('%s must be a list' % startoftrial)
    # Make sure typedf is a dataframe
    if not isinstance(typedf, pd.core.frame.DataFrame):
        raise TypeError('%s must be a pandas dataframe.' % typedf)
    # Makes sure segment object is passed
    if not isinstance(seg, neo.core.segment.Segment):
        raise TypeError('%s must be a segment object' % seg)
    # Gets processed event object
    try:
        event_obj = [x for x in seg.events if x.name == name][0]
    except IndexError:
        raise IndexError("""%s does not have an events object named %s. Make sure 
            to run ProcessEvents first!""" % (seg, name))
    ## Get relevent event types
    # get events that signify start of trial
    start_events = typedf.loc[typedf.type.isin(startoftrial)].index
    # get events that signify different epochs
    epoch_events = typedf.loc[typedf.type.isin(epochs)].index
    # Transforms seg.events object into a dataframe with times and labels
    # as columns
    labels = pd.Series(event_obj.labels.astype(str), name='event')
    times = pd.Series(event_obj.times, name='time')
    trial_df = pd.concat([times, labels], axis=1)
    # Adds trial index column to dataframe
    trial_df['trial_idx'] = 0
    # Marks start of trial
    trial_df.loc[trial_df.event.isin(start_events), 'trial_idx'] = 1
    # Uses cumulative sum to determine trial number
    trial_df['trial_idx'] = trial_df['trial_idx'].cumsum()
    # Get results from each trial
    results_by_trial = trial_df.groupby('trial_idx').agg({'event': 
        lambda x: [evt for evt in list(x) if evt in epoch_events]})
    # Result processing
    results_by_trial['results'] = results_by_trial.event.apply(lambda x: 
        ResultOfTrial(x, appendmultiple=appendmultiple))
    # Merged dataframe
    trials = pd.merge(trial_df, results_by_trial.drop('event', axis=1), 
        how='left', left_on='trial_idx', right_index=True)
    # New column with previous result too (example: 'omission_correct')
    trials['with_previous_results'] = np.nan
    for idx in trials.trial_idx[trials.trial_idx > 1]:
        current_results = trials.loc[trials.trial_idx == idx, 'results'].values[0]
        previous_results = trials.loc[trials.trial_idx == idx - 1, 'results'].values[0]
        combined = previous_results + '_' + current_results
        trials.loc[trials.trial_idx == idx, 'with_previous_results'] = combined
    # Attach event type to trials dataframe
    trials['event_type'] = trials.event.apply(lambda x: typedf.loc[x, 'type'])
    # Only returns events that started with first trial
    if firsttrial:
        trials = trials.loc[trials.trial_idx >= 1, :]
    # Add name to dataframe
    trials.name = 'trials'
    # Add to segment object
    from imaging_analysis.segment_processing import AppendDataframesToSegment
    AppendDataframesToSegment(seg, trials, ['trials'])
    # return dataframe if asked
    if returndf:
        return trials

def CalculateStartsAndDurations(trials=None, epoch_column=None, 
        startoftrial=None, endoftrial=None, endeventmissing='next'):
    """Given a trialframe and epoch_column, calculates start and durations."""
    # Makes sure trials is a dataframe
    if not isinstance(trials, pd.core.frame.DataFrame):
        raise TypeError('%s must be a dataframe' % trials)
    # Makes sure startoftrial is a list
    if not isinstance(startoftrial, list):
        raise TypeError('%s must be a list' % startoftrial)
    # Makes sure endoftrial is a list
    if not isinstance(endoftrial, list):
        raise TypeError('%s must be a list' % endoftrial)
    # Make sure endeventmissing is either last or next
    if endeventmissing not in ['last', 'next']:
        raise ValueError("endeventmissing must be 'last' or 'next'")
    # Gets a set of the epochs
    epochs = trials.loc[trials[epoch_column].notnull(), epoch_column].unique()
    # to return (list of tuples)
    final_list = []
    for epoch in epochs:
        # Makes a dataframe of events only concerning a specific epoch
        epochframe = trials.loc[trials[epoch_column] == epoch, :]
        start_times = []
        durations = []
        # Goes through each trial in that epoch to figure out start times and
        # durations
        for trial_num in epochframe.trial_idx.unique():
            # Makes a dataframe of events for a specific trial
            trialframe = epochframe.loc[epochframe.trial_idx == trial_num]
            # Mask for start events
            startmask = trialframe.event_type.isin(startoftrial)
            # Gets the earliest timestamp for a start event
            starttime = trialframe.loc[startmask, 'time'].min()
            # Mask for end events
            endmask = trialframe.event_type.isin(endoftrial)
            # Checks if there are end events in the trial. If so, takes the
            # earliest one as the end of the trial
            if trialframe.loc[endmask].shape[0]:
                endtime = trialframe.loc[endmask, 'time'].min()
            # Otherwise gets the last event in the trial if 'last' mode is chosen
            elif endeventmissing == 'last':
                endtime = trialframe.time.max()
            # Or gets the first event of the next trial if 'next' mode is chosen
            elif endeventmissing == 'next':
                endtime = trials.loc[trials.trial_idx == trial_num + 1, 'time'].min()
                # For the last trial, trial ends with the last timestamp
                if np.isnan(endtime):
                    endtime = trials.time.max()
            duration = endtime - starttime
            durations.append(duration)
            start_times.append(starttime)
        final_list.append((start_times, durations, epoch))
    return final_list

def GroupTrialsByEpoch(seg=None, trials=None, startoftrial=None, 
        endoftrial=None, endeventmissing='next'):
    """Given a segment object and a trials dataframe, will go through 
    each epoch type and collect when the trial started and stopped.
    Started is by the start events in startoftrial, stopped is by
    event in endoftrial. If endoftrial is missing, stopped can be determined
    by two modes:
    endeventmissing = 'next': end of trial is the start of the next one
    (last trial ends on its last event)
    endeventmissing = 'last': end of trial is the last event of that trial"""
    # Makes sure seg is a segment object
    if not isinstance(seg, neo.core.Segment):
        raise TypeError('%s must be a segment object' % seg)
    # Assign trials for segment object if trials is not given
    if trials is None:
        trials = seg.dataframes['trials']
    # Calculate epochs for results column
    epoch_list = CalculateStartsAndDurations(trials=trials, epoch_column='results', startoftrial=startoftrial, endoftrial=endoftrial, 
        endeventmissing=endeventmissing)
    # Adds durations and start times to create an Epoch
    for epoch in epoch_list:
        seg.epochs.append(Epoch(times=np.array(epoch[0]) * pq.s,
            durations=np.array(epoch[1]) * pq.s, name=epoch[2]))
    # Now calculates epochs with previous trial results
    # Gets a set of the epochs
    prev_epoch_list = CalculateStartsAndDurations(trials=trials, 
        epoch_column='with_previous_results', startoftrial=startoftrial, 
        endoftrial=endoftrial, endeventmissing=endeventmissing)
    # Adds durations and start times to create an Epoch
    for epoch in prev_epoch_list:
        seg.epochs.append(Epoch(times=np.array(epoch[0]) * pq.s,
            durations=np.array(epoch[1]) * pq.s, name=epoch[2]))

def GetImagingDataTTL(fp_datapath, time_idx=0, event_name='PC0/', event_idx=1):
    block = utils.ReadNeoTdt(path=fp_datapath)
    seglist = block.segments
    seg = seglist[0]
    try:
        event = [x for x in seg.events if event_name in x.name][0]
    except:
        event = seg.events[event_idx]
    return event.times[time_idx].magnitude