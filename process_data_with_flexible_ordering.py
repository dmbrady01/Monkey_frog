#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_data_V2.py: Python script that processes fiber photometry data. It truncates
the signal, filters it, labels events, processes trials, and groups trials by epoch.
The main difference in V2 is that you can specify the order of processing explicitly
for both before and after aligning with events.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "06 Feb 2019"

import sys
from imaging_analysis.event_processing import LoadEventParams, ProcessEvents, ProcessTrials, GroupTrialsByEpoch, GenerateManualEventParamsJson, GetImagingDataTTL
from imaging_analysis.segment_processing import TruncateSegments, AppendDataframesToSegment, AlignEventsAndSignals
from imaging_analysis.utils import ReadNeoPickledObj, ReadNeoTdt, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.signal_processing import SingleStepProcessSignalData, DeltaFOverF, PolyfitWindow, SmoothSignalWithPeriod, ZScoreCalculator, SmoothSignalWithPeriod, Downsample, GetImagingDataIndices
import numpy as np
from neo.core import Epoch, Event
import quantities as pq
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import json
import click
import os

@click.command()
@click.option(
    '-f',
    '--file',
    default="params_flexible.json",
    type=str,
    show_default=True,
    help=("Parameter file with channel info, events of interest, etc."))
def main(file: str) -> None:
    """Runs flexible imaging analysis based on inputs from a parameter file"""
    sns.set_style('darkgrid')
    ############## PART 1 Preprocess data ##########################
    ##################### Kazu/Mike Section ###############################
    print(f"LOADING PARAMETERS FROM {file}")
    params = json.load(open(file, 'r'))
    mode = params.get("mode")
    dpaths = params.get("dpaths")
    baseline_window_unaligned = params.get("baseline_window_unaligned")
    offsets_list = params.get("offsets_list")
    offset_events = params.get("offset_events", [])
    before_alignment = params.get("before_alignment")
    signal_channel = params.get("signal_channel")
    reference_channel = params.get("reference_channel")
    analysis_blocks = params.get("analysis_blocks")
    path_to_ttl_event_params = params.get("path_to_ttl_event_params")
    path_to_social_excel = params.get("path_to_social_excel")
    ####################### PREPROCESSING DATA ###############################
    print(('\n\n\n\nRUNNING IN MODE: %s \n\n\n' % mode))
    for dpath_ind, dpath in enumerate(dpaths):
        # Reads data from Tdt folder
        PrintNoNewLine('\nCannot find processed pkl object, reading TDT folder instead...')
        block = ReadNeoTdt(path=dpath, return_block=True)
        seglist = block.segments
        print('Done!')

        # Trunactes first/last seconds of recording
        PrintNoNewLine('Truncating signals and events...')
        seglist = TruncateSegments(seglist, start=0, end=10, clip_same=True)
        print('Done!')


        # Iterates through each segment in seglist. Right now, there is only one segment
        for segment in seglist:
            segment_name = segment.name
            # Extracts the sampling rate from the signal channel
            try:
                sampling_rate = [x for x in segment.analogsignals if x.name == signal_channel][0].sampling_rate
            except IndexError:
                raise ValueError('Could not find your channels. Make sure you have the right names!')
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
            PrintNoNewLine('\nProcessing signal before event alignment...')
            before_alignment_channels = []
            if len(before_alignment) > 0:
                for step_number, process in enumerate(before_alignment):
                    if step_number == 0:
                        input_sig_ch = signal_channel
                        input_ref_ch = reference_channel

                    if 'start_from' in list(process['options'].keys()):
                        start_from = process['options']['start_from']
                        if start_from == 'list':
                            start = offsets_list[dpath_ind]
                            end = start + process['options']['end_from']
                        else:
                            start = GetImagingDataTTL(dpath)
                            end = start + process['options']['end_from']
                        indices = GetImagingDataIndices(start, end, sampling_rate.magnitude)
                        process['options']['period'] = indices

                    signal, reference = SingleStepProcessSignalData(data=segment, process_type=process['type'], 
                        input_sig_ch=input_sig_ch, input_ref_ch=input_ref_ch, datatype='segment', **process['options'])

                    if process['type'] == 'filter':
                        input_sig_ch = 'filtered_signal' 
                        input_ref_ch = 'filtered_reference'
                    elif process['type'] == 'detrend':
                        input_sig_ch = 'detrended_signal'
                        input_ref_ch = 'detrended_reference'
                    elif process['type'] == 'subtract':
                        input_sig_ch = 'subtracted_signal'
                        input_ref_ch = None
                    elif process['type'] == 'measure':
                        input_sig_ch = 'measure_signal'
                        input_ref_ch = 'measure_reference'

                    if input_sig_ch not in before_alignment_channels:
                        before_alignment_channels.append(input_sig_ch)
                    
                    if input_ref_ch not in before_alignment_channels:
                        before_alignment_channels.append(input_ref_ch)
            # Appends an Event object that has all event timestamps and the proper label
            # (determined by the evtframe loaded earlier). Uses a tolerance (in seconds)
            # to determine if events co-occur. For example, if tolerance is 1 second
            # and ch1 fires an event, ch2 fires an event 0.5 seconds later, and ch3 fires
            # an event 3 seconds later, the output array will be [1, 1, 0] and will
            # match the label in evtframe (e.g. 'omission')
            print('Done!')
            

            if mode == 'TTL':
                # Loading event labeling/combo parameters
                path_to_event_params = path_to_ttl_event_params[dpath_ind]
            elif mode == 'manual':
                # Generates a json for reading excel file events
                path_to_event_params = 'imaging_analysis/manual_event_params.json'
                GenerateManualEventParamsJson(path_to_social_excel[dpath_ind], event_col='Bout type', 
                    name=path_to_event_params)
            # This loads our event params json
            start, end, epochs, evtframe, typeframe = LoadEventParams(dpath=path_to_event_params, 
                mode=mode)
            # Appends processed event_param.json info to segment object
            AppendDataframesToSegment(segment, [evtframe, typeframe], 
                ['eventframe', 'resultsframe'])
        
            # Processing events
            PrintNoNewLine('\nProcessing event times and labels...')
            if mode == 'manual':
                manualframe = path_to_social_excel[dpath_ind]
            else:
                manualframe = None
            if len(offset_events) == 0:
                offset_event = None
            else:
                offset_event = offset_events[dpath_ind]
            ProcessEvents(seg=segment, tolerance=.1, evtframe=evtframe, 
                name='Events', mode=mode, manualframe=manualframe, 
                event_col='Bout type', start_col='Bout start', end_col='Bout end', offset_events=offset_event)
            print('Done!')
            

            # Takes processed events and segments them by trial number. Trial start
            # is determined by events in the list 'start' from LoadEventParams. This
            # can be set in the event_params.json. Additionally, the result of the 
            # trial is set by matching the epoch type to the typeframe dataframe 
            # (also from LoadEventParams). Example of epochs are 'correct', 'omission',
            # etc. 
            # The result of this process is a dataframe with each event and their
            # timestamp in chronological order, with the trial number and trial outcome
            # appended to each event/timestamp.
            PrintNoNewLine('\nProcessing trials...')
            trials = ProcessTrials(seg=segment, name='Events', 
                startoftrial=start, epochs=epochs, typedf=typeframe, 
                appendmultiple=False)
            print('Done!')
            
            # With processed trials, we comb through each epoch ('correct', 'omission'
            # etc.) and find start/end times for each trial. Start time is determined
            # by the earliest 'start' event in a trial. Stop time is determined by
            # 1) the earliest 'end' event in a trial, 2) or the 'last' event in a trial
            # or the 3) 'next' event in the following trial.
            PrintNoNewLine('\nCalculating epoch times and durations...')
            GroupTrialsByEpoch(seg=segment, startoftrial=start, endoftrial=end, 
                endeventmissing='last')
            print('Done!')
            segment.processed = True


        ################### ALIGN DATA ##########################################
        # for segment in seglist:
            for block in analysis_blocks:
                # Extract analysis block params
                epoch_name = block['epoch_name']
                event = block['event']
                prewindow = block['prewindow']
                postwindow = block['postwindow']
                downsample = block['downsample']
                after_alignment = block['after_alignment']
                quantification = block['quantification']
                baseline_window = block['baseline_window']
                response_window = block['response_window']
                save_file_as = block['save_file_as']
                heatmap_range = block['plot_paramaters']['heatmap_range']
                smoothing_window = block['plot_paramaters']['smoothing_window']
                lookup = {}

        #############################################################################
        ######################## PROCESS SIGNALS (IF NECESSARY); PLOT; STATS ######
                # Load data

                # Checks to see if we have filtered the data before alignment
                filter_channel_names = [x for x in before_alignment_channels if 'filtered' in x]
                if len(filter_channel_names) == 0:
                    filter_channel_names = [signal_channel, reference_channel]

                PrintNoNewLine('Centering trials and analyzing for filtered signal...')
                for channel in filter_channel_names:
                    dict_name = epoch_name + '_' + channel
                    lookup[channel] = dict_name

                    results = AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name=channel, 
                        event_ch_name='Events', event=event, event_type='label', 
                        prewindow=prewindow, postwindow=postwindow, window_type='event', 
                        clip=False, name=dict_name, to_csv=False, dpath=dpath)
                print('Done!')

                filter_signal_name = [x for x in filter_channel_names if 'signal' in x][0]
                filter_reference_name = [x for x in filter_channel_names if 'reference' in x][0]
                signal = segment.analyzed[lookup[filter_signal_name]]['all_traces']
                reference = segment.analyzed[lookup[filter_reference_name]]['all_traces']

                # check to see if we need to filter after alignment
                for process in [x for x in after_alignment if x['type'] == 'filter']:
                    data = {filter_signal_name: signal, filter_reference_name: reference}
                    signal, reference = SingleStepProcessSignalData(data=data, process_type=process['type'], 
                        input_sig_ch=filter_signal_name, input_ref_ch=filter_reference_name, 
                        datatype='dataframe', **process['options'])

                filter_signal_name = 'filtered_signal'
                filter_reference_name = 'filtered_reference'
                lookup[filter_signal_name] = epoch_name + '_' + filter_signal_name
                lookup[filter_reference_name] = epoch_name + '_' + filter_reference_name
                if lookup[filter_signal_name] in list(segment.analyzed.keys()):
                    segment.analyzed[lookup[filter_signal_name]]['all_traces'] = signal.copy()
                    segment.analyzed[lookup[filter_reference_name]]['all_traces'] = reference.copy()
                else:
                    segment.analyzed[lookup[filter_signal_name]] = {'all_traces': signal.copy(), 'all_events': results}
                    segment.analyzed[lookup[filter_reference_name]] = {'all_traces': reference.copy(), 'all_events': results}

                # Get plotting read
                figure = plt.figure(figsize=(12, 12))
                figure.subplots_adjust(hspace=1.3)
                ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
                ax3 = plt.subplot2grid((6, 2), (4, 0), rowspan=2)
                ax4 = plt.subplot2grid((6, 2), (0, 1), rowspan=3)
                ax5 = plt.subplot2grid((6, 2), (3, 1), rowspan=3)

            ############################### PLOT AVERAGE EVOKED RESPONSE ######################
                PrintNoNewLine('Calculating average filtered responses for %s trials...' % epoch_name)
                signal_mean = signal.mean(axis=1)
                reference_mean = reference.mean(axis=1)

                signal_sem = signal.sem(axis=1)
                reference_sem = reference.sem(axis=1)

                signal_dc = signal_mean.mean()
                reference_dc = reference_mean.mean()

                signal_avg_response = signal_mean - signal_dc 
                reference_avg_response = reference_mean - reference_dc

                if smoothing_window is not None:
                    signal_avg_response = SmoothSignalWithPeriod(x=signal_avg_response, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                    reference_avg_response = SmoothSignalWithPeriod(x=reference_avg_response, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                    signal_sem = SmoothSignalWithPeriod(x=signal_sem, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                    reference_sem = SmoothSignalWithPeriod(x=reference_sem, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')

                curr_ax = ax1
                curr_ax.plot(signal_avg_response.index, signal_avg_response.values, color='b', linewidth=2)
                curr_ax.fill_between(signal_avg_response.index, (signal_avg_response - signal_sem).values, 
                    (signal_avg_response + signal_sem).values, color='b', alpha=0.05)

                # Plotting reference
                curr_ax.plot(reference_avg_response.index, reference_avg_response.values, color='g', linewidth=2)
                curr_ax.fill_between(reference_avg_response.index, (reference_avg_response - reference_sem).values, 
                    (reference_avg_response + reference_sem).values, color='g', alpha=0.05)

                # Plot event onset
                curr_ax.axvline(0, color='black', linestyle='--')
                curr_ax.set_ylabel('Voltage (V)')
                curr_ax.set_xlabel('Time (s)')
                curr_ax.legend(['465 nm', '405 nm', event])
                curr_ax.set_title('Average Lowpass Signal $\pm$ SEM: {} Trials'.format(signal.shape[1]))
                print('Done!')


    ################################################################################################################
            ############################# Calculate detrended signal #################################

                    # # Detrending
                    # PrintNoNewLine('Detrending signal...')
                    # fits = np.array([np.polyfit(reference.values[:, i],signal.values[:, i],1) for i in xrange(signal.shape[1])])
                    # Y_fit_all = np.array([np.polyval(fits[i], reference.values[:,i]) for i in np.arange(reference.values.shape[1])]).T
                    # Y_df_all = signal.values - Y_fit_all
                    # # detrended_signal = pd.DataFrame(Y_df_all, index=signal.index)

                # Checks to see if we have detrended the data before alignment
                detrend_channel_names = [x for x in before_alignment_channels if 'detrended' in x]
                if len(detrend_channel_names) == 0:
                    detrend_channel_names = [filter_signal_name, filter_reference_name]

                PrintNoNewLine('Centering trials and analyzing for detrended signal...')
                try:
                    for channel in detrend_channel_names:
                        dict_name = epoch_name + '_' + channel
                        lookup[channel] = dict_name

                        results = AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name=channel, 
                            event_ch_name='Events', event=event, event_type='label', 
                            prewindow=prewindow, postwindow=postwindow, window_type='event', 
                            clip=False, name=dict_name, to_csv=False, dpath=dpath)
                    print('Done!')
                except:
                    print('No detrending before alignment...')
                detrend_signal_name = [x for x in detrend_channel_names if 'signal' in x][0]
                detrend_reference_name = [x for x in detrend_channel_names if 'reference' in x][0]
                detrended_signal = segment.analyzed[lookup[detrend_signal_name]]['all_traces']
                detrended_reference = segment.analyzed[lookup[detrend_reference_name]]['all_traces']

                # check to see if we need to filter after alignment
                for process in [x for x in after_alignment if x['type'] == 'detrend']:
                    data = {detrend_signal_name: detrended_signal, detrend_reference_name: detrended_reference}
                    detrended_signal, detrended_reference = SingleStepProcessSignalData(data=data, process_type=process['type'], 
                        input_sig_ch=detrend_signal_name, input_ref_ch=detrend_reference_name, 
                        datatype='dataframe', **process['options'])

                detrend_signal_name = 'detrended_signal'
                detrend_reference_name = 'detrended_reference'
                lookup[detrend_signal_name] = epoch_name + '_' + detrend_signal_name
                lookup[detrend_reference_name] = epoch_name + '_' + detrend_reference_name
                if lookup[detrend_signal_name] in list(segment.analyzed.keys()):
                    segment.analyzed[lookup[detrend_signal_name]]['all_traces'] = detrended_signal.copy()
                    segment.analyzed[lookup[detrend_reference_name]]['all_traces'] = detrended_reference.copy()
                else:
                    segment.analyzed[lookup[detrend_signal_name]] = {'all_traces': detrended_signal.copy(), 'all_events': results}
                    segment.analyzed[lookup[detrend_reference_name]] = {'all_traces': detrended_reference.copy(), 'all_events': results}

                # if downsample > 0:
                #     detrended_signal.reset_index(inplace=True)
                #     detrended_reference.reset_index(inplace=True)
                #     sample = (detrended_signal.index.to_series() / downsample).astype(int)
                #     detrended_signal = detrended_signal.groupby(sample).mean()
                #     detrended_reference = detrended_reference.groupby(sample).mean()
                #     detrended_signal = detrended_signal.set_index('index')
                #     detrended_reference = detrended_reference.set_index('index')

            ################# PLOT DETRENDED SIGNAL ###################################

                detrended_signal_mean = detrended_signal.mean(axis=1)
                detrended_signal_sem = detrended_signal.sem(axis=1)

                if smoothing_window is not None:
                    detrended_signal_mean = SmoothSignalWithPeriod(x=detrended_signal_mean, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                    detrended_signal_sem = SmoothSignalWithPeriod(x=detrended_signal_sem, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')

                # Plotting signal
                # current axis
                curr_ax = ax2
                # # curr_ax = axs[1, 0]
                #curr_ax = plt.axes()
                if any([x['type'] == 'measure' for x in before_alignment]):
                    pass
                else:
                    z_score_window = [x['options']['period'] for x in after_alignment if x['type'] == 'measure'][0]
                    zscore_start = detrended_signal[z_score_window[0]:z_score_window[1]].index[0]
                    zscore_end = detrended_signal[z_score_window[0]:z_score_window[1]].index[-1]
                    zscore_height = detrended_signal[z_score_window[0]:z_score_window[1]].mean(axis=1).min()
                    if zscore_height < 0:
                        zscore_height = zscore_height * 1.3 
                    else:
                        zscore_height = zscore_height * 0.7

                    curr_ax.plot([zscore_start, zscore_end], [zscore_height, zscore_height], color='.1', linewidth=3)


                curr_ax.plot(detrended_signal_mean.index, detrended_signal_mean.values, color='b', linewidth=2)
                curr_ax.fill_between(detrended_signal_mean.index, (detrended_signal_mean - detrended_signal_sem).values, 
                    (detrended_signal_mean + detrended_signal_sem).values, color='b', alpha=0.05)

                # Plot event onset
                if any([x['type'] == 'measure' for x in before_alignment]):
                    pass
                else:
                    curr_ax.legend(['z-score window'])
                curr_ax.axvline(0, color='black', linestyle='--')
                curr_ax.set_ylabel('Voltage (V)')
                curr_ax.set_xlabel('Time (s)')
                curr_ax.set_title('465 nm Average Detrended Signal $\pm$ SEM')

                print('Done!')
            
            # ########### Calculate z-scores ###############################################
                measure_channel_names = [x for x in before_alignment_channels if 'measure' in x]
                if len(measure_channel_names) == 0:
                    measure_channel_names = [detrend_signal_name, detrend_reference_name]

                PrintNoNewLine('Centering trials and analyzing for z scores...')
                try:
                    for channel in measure_channel_names:
                        dict_name = epoch_name + '_' + channel
                        lookup[channel] = dict_name

                        results = AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name=channel, 
                            event_ch_name='Events', event=event, event_type='label', 
                            prewindow=prewindow, postwindow=postwindow, window_type='event', 
                            clip=False, name=dict_name, to_csv=False, dpath=dpath)
                except:
                    print('No z scores before alignment...')
                print('Done!')

                measure_signal_name = [x for x in measure_channel_names if 'signal' in x][0]
                measure_reference_name = [x for x in measure_channel_names if 'reference' in x][0]
                measure_signal = segment.analyzed[lookup[measure_signal_name]]['all_traces']
                measure_reference = segment.analyzed[lookup[measure_reference_name]]['all_traces']
                
                # check to see if we need to filter after alignment
                for process in [x for x in after_alignment if x['type'] == 'measure']:
                    data = {measure_signal_name: measure_signal, measure_reference_name: measure_reference}
                    measure_signal, measure_reference = SingleStepProcessSignalData(data=data, process_type=process['type'], 
                        input_sig_ch=measure_signal_name, input_ref_ch=measure_reference_name, 
                        datatype='dataframe', **process['options'])

                measure_signal_name = 'measure_signal'
                measure_reference_name = 'measure_reference'
                lookup[measure_signal_name] = epoch_name + '_' + measure_signal_name
                lookup[measure_reference_name] = epoch_name + '_' + measure_reference_name
                if lookup[measure_signal_name] in list(segment.analyzed.keys()):
                    segment.analyzed[lookup[measure_signal_name]]['all_traces'] = measure_signal.copy()
                    segment.analyzed[lookup[measure_reference_name]]['all_traces'] = measure_reference.copy()
                else:
                    segment.analyzed[lookup[measure_signal_name]] = {'all_traces': measure_signal.copy(), 'all_events': results}
                    segment.analyzed[lookup[measure_reference_name]] = {'all_traces': measure_reference.copy(), 'all_events': results}

                # if downsample > 0:
                #     measure_signal.reset_index(inplace=True)
                #     measure_reference.reset_index(inplace=True)
                #     sample = (measure_signal.index.to_series() / downsample).astype(int)
                #     measure_signal = measure_signal.groupby(sample).mean()
                #     measure_reference = measure_reference.groupby(sample).mean()
                #     measure_signal = measure_signal.set_index('index')
                #     measure_reference = measure_reference.set_index('index')


                zscores = measure_signal.copy()

            ############################ Make rasters #######################################
                PrintNoNewLine('Making heatmap for %s trials...' % event)
                # indice that is closest to event onset
                # curr_ax = axs[0, 1]
                curr_ax = ax4
                # curr_ax = plt.axes()
                # Plot nearest point to time zero
                zero = np.concatenate([np.where(zscores.index == np.abs(zscores.index).min())[0], 
                    np.where(zscores.index == -1*np.abs(zscores.index).min())[0]]).min()
                for_hm = zscores.T.copy()
                for_hm.reset_index(drop=True, inplace=True)
                # for_hm.index = for_hm.index + 1
                for_hm.columns = np.round(for_hm.columns, 1)
                try:
                    sns.heatmap(for_hm.iloc[::-1], center=0, robust=True, ax=curr_ax, cmap='bwr',
                        xticklabels=int(for_hm.shape[1]*.15), yticklabels=int(for_hm.shape[0]*.15), 
                        vmin=heatmap_range[0], vmax=heatmap_range[1])
                except:
                    sns.heatmap(for_hm.iloc[::-1], center=0, robust=True, ax=curr_ax, cmap='bwr', 
                        xticklabels=int(for_hm.shape[1]*.15), vmin=heatmap_range[0], vmax=heatmap_range[1])
                curr_ax.axvline(zero, linestyle='--', color='black', linewidth=2)
                curr_ax.set_ylabel('Trial');
                curr_ax.set_xlabel('Time (s)');
                if any([x['type'] == 'measure' for x in before_alignment]):
                    period = [x['options']['period'] for x in before_alignment if x['type'] == 'measure'][0]
                    sampling_per = segment.analogsignals[0].sampling_period
                    curr_ax.set_title('Z-Score Heat Map \n Baseline Window: {} to {} Seconds'.format(round(period[0]*sampling_per), round(period[1]*sampling_per)));
                else:
                    curr_ax.set_title('Z-Score Heat Map \n Baseline Window: {} to {} Seconds'.format(z_score_window[0], z_score_window[1]));
                print('Done!')
            ########################## Plot Z-score waveform ##########################
                PrintNoNewLine('Plotting Z-Score waveforms...')
                zscores_mean = zscores.mean(axis=1)

                zscores_sem = zscores.sem(axis=1)

                if smoothing_window is not None:
                    zscores_mean = SmoothSignalWithPeriod(x=zscores_mean, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                    zscores_sem = SmoothSignalWithPeriod(x=zscores_sem, 
                        sampling_rate=float(sampling_rate)/downsample, 
                        ms_bin=smoothing_window, window='flat')
                # Plotting signal
                # current axis
                # curr_ax = axs[1, 1]
                curr_ax = ax3
                #curr_ax = plt.axes()
                # Plot baseline and response

                legend = []
                if baseline_window_unaligned is False:
                    baseline_start = zscores[baseline_window[0]:baseline_window[1]].index[0]
                    baseline_end = zscores[baseline_window[0]:baseline_window[1]].index[-1]
                    baseline_height = zscores[baseline_window[0]:baseline_window[1]].mean(axis=1).min() - 0.5
                    curr_ax.plot([baseline_start, baseline_end], [baseline_height, baseline_height], color='.6', linewidth=3)
                    legend.append('baseline window')
                
                response_start = zscores[response_window[0]:response_window[1]].index[0]
                response_end = zscores[response_window[0]:response_window[1]].index[-1]
                response_height = zscores[response_window[0]:response_window[1]].mean(axis=1).max() + .5
                curr_ax.plot([response_start, response_end], [response_height, response_height], color='r', linewidth=3)
                legend.append('response window')

                curr_ax.plot(zscores_mean.index, zscores_mean.values, color='b', linewidth=2)
                curr_ax.fill_between(zscores_mean.index, (zscores_mean - zscores_sem).values, 
                    (zscores_mean + zscores_sem).values, color='b', alpha=0.05)

                # Plot event onset
                curr_ax.axvline(0, color='black', linestyle='--')

                curr_ax.set_ylabel('Z-Score')
                curr_ax.set_xlabel('Time (s)')
                curr_ax.legend(legend)
                curr_ax.set_title('465 nm Average Z-Score Signal $\pm$ SEM')
                print('Done!')
            ##################### Quantification #################################
                PrintNoNewLine('Performing statistical testing on baseline vs response periods...')
                if quantification is not None:
                    if baseline_window_unaligned:
                        baseline_window = process['options']['period']
                        baseline_window_values = [x.magnitude for x in segment.analogsignals if x.name == 'measure_signal'][0]
                    else:
                        baseline_window_values = zscores
                    # Generating summary statistics
                    if quantification == 'AUC':
                        base = np.trapz(baseline_window_values[baseline_window[0]:baseline_window[1]], axis=0)
                        resp = np.trapz(zscores[response_window[0]:response_window[1]], axis=0)
                        ylabel = 'AUC'
                    elif quantification == 'mean':
                        base = np.mean(baseline_window_values[baseline_window[0]:baseline_window[1]], axis=0)
                        resp = np.mean(zscores[response_window[0]:response_window[1]], axis=0)
                        ylabel = 'Z-Score'
                    elif quantification == 'median':
                        base = np.median(baseline_window_values[baseline_window[0]:baseline_window[1]], axis=0)
                        resp = np.median(zscores[response_window[0]:response_window[1]], axis=0)
                        ylabel = 'Z-Score'

                    if isinstance(base, pd.core.series.Series):
                        base = base.values

                    if isinstance(resp, pd.core.series.Series):
                        resp = resp.values

                    base_sem = np.mean(base)/np.sqrt(base.shape[0])
                    resp_sem = np.mean(resp)/np.sqrt(resp.shape[0])

                    # Testing for normality (D'Agostino's K-Squared Test) (N>8)
                    if base.shape[0] > 8:
                        normal_alpha = 0.05
                        base_normal = stats.normaltest(base)
                        resp_normal = stats.normaltest(resp)
                    else:
                        normal_alpha = 0.05
                        base_normal = [1, 1]
                        resp_normal = [1, 1]

                    difference_alpha = 0.05
                    if baseline_window_unaligned is True:
                        test = 'One Sample T-Test'
                        stats_results = stats.ttest_1samp(resp, base[0])
                    elif (base_normal[1] >= normal_alpha) or (resp_normal[1] >= normal_alpha):
                        test = 'Wilcoxon Signed-Rank Test'
                        stats_results = stats.wilcoxon(base, resp)
                    else:
                        test = 'Paired Sample T-Test'
                        stats_results = stats.ttest_rel(base, resp)

                    if stats_results[1] <= difference_alpha:
                        sig = '**'
                    else:
                        sig = 'ns'

                    #curr_ax = plt.axes() 
                    curr_ax = ax5
                    ind = np.arange(2)
                    labels = ['baseline', 'response']
                    bar_kwargs = {'width': 0.7,'color': ['.6', 'r'],'linewidth':2,'zorder':5}
                    err_kwargs = {'zorder':0,'fmt': 'none','linewidth':2,'ecolor':'k'}
                    curr_ax.bar(ind, [base.mean(), resp.mean()], tick_label=labels, **bar_kwargs)
                    curr_ax.errorbar(ind, [base.mean(), resp.mean()], yerr=[base_sem, resp_sem],capsize=5, **err_kwargs)
                    x1, x2 = 0, 1
                    y = np.max([base.mean(), resp.mean()]) + np.max([base_sem, resp_sem])*1.3
                    h = y * 1.5
                    col = 'k'
                    curr_ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                    curr_ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)
                    curr_ax.set_ylabel(ylabel)
                    curr_ax.set_title('Baseline vs. Response Changes in Z-Score Signal \n {} of {}s'.format(test, quantification))

                    print('Done!')
            ################# Save Stuff ##################################
                PrintNoNewLine('Saving everything...')
                save_path = os.path.join(dpath, segment_name, save_file_as)
                figure.savefig(save_path + '.png', format='png')
                # figure.savefig(save_path + '.pdf', format='pdf')
                plt.close()
                print('Done!')

                # Trial z-scores
                # Fix columns
                zscores.columns = np.arange(1, zscores.shape[1] + 1)
                zscores.columns.name = 'trial'
                # Fix rows 
                zscores.index.name = 'time'
                if downsample > 0:
                    zscores = Downsample(zscores, downsample, index_col='time')
                zscores.to_csv(save_path + '_zscores_aligned.csv')
                if quantification is not None:
                    # Trial point estimates
                    point_estimates = pd.DataFrame({'baseline': base, 'response': resp}, 
                        index=np.arange(1, resp.shape[0]+1)).ffill().bfill()
                    point_estimates.index.name = 'trial'
                    point_estimates.to_csv(save_path + '_point_estimates.csv')
                # Save meta data
                metadata = {
                    'baseline_window': baseline_window,
                    'response_window': response_window, 
                    'quantification': quantification,
                    'original_sampling_rate': float(sampling_rate),
                    'downsampled_sampling_rate': float(sampling_rate)/downsample
                }
                with open(save_path + '_metadata.json', 'w') as fp:
                    json.dump(metadata, fp)
                # Save smoothed data
                smoothed_zscore = pd.concat([zscores_mean, zscores_sem], axis=1)
                smoothed_zscore.columns = ['mean', 'sem']
                if downsample > 0:
                    smoothed_zscore = Downsample(smoothed_zscore, downsample, index_col='time')
                smoothed_zscore.to_csv(save_path + '_smoothed_zscores.csv')


        print(('Finished processing datapath: %s' % dpath))

if __name__ == '__main__':
    main()