#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_data.py: Python script that processes fiber photometry data. It truncates
the signal, filters it, labels events, processes trials, and groups trials by epoch.
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "06 Feb 2019"

import sys
from imaging_analysis.event_processing import LoadEventParams, ProcessEvents, ProcessTrials, GroupTrialsByEpoch, GenerateManualEventParamsJson
from imaging_analysis.segment_processing import TruncateSegments, AppendDataframesToSegment, AlignEventsAndSignals
from imaging_analysis.utils import ReadNeoPickledObj, ReadNeoTdt, WriteNeoPickledObj, PrintNoNewLine
from imaging_analysis.signal_processing import ProcessSignalData, DeltaFOverF, PolyfitWindow, SmoothSignalWithPeriod, ZScoreCalculator, SmoothSignalWithPeriod
import numpy as np
from neo.core import Epoch, Event
import quantities as pq
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import json

sns.set_style('darkgrid')
############## PART 1 Preprocess data ##########################

#######################################################################
############ KEVIN SECTION ##################################
### For new rig/Kevin's data
deltaf_options = {
    'detrend': 'savgol_from_reference',
    'signal_window_length': None,
    'mode': 'median', 
    'subtract': 'True',
    'period': [40, 306290]
}
z_score_before_alignment = True
signal_channel = '465A 1' # Name of our signal channel
reference_channel = '405A 1' # Name of our reference channel

mode = 'TTL'
##### WHAT ARE THE EVENTS/HOW TO INTERPRET EVENT TIMESTAMPS
path_to_ttl_event_params = [
    'imaging_analysis/ttl_event_params_new_rig.json'
]
#### WHERE IS THE DATA
dpaths = [
    '/Users/DB/Development/Monkey_frog/data/FirstFibPho-210808'
]

analysis_blocks = [
    {
        'epoch_name': ['correct', 'incorrect'],
        'event': 'iti_start',
        'prewindow': 3,
        'postwindow': 10,
        'z_score_window': [],
        'downsample': 10,
        'quantification': 'mean', # options are AUC, median, and mean
        'baseline_window': [-5, 0],
        'response_window': [0, 5],
        'save_file_as': 'correct_processed',
        'plot_paramaters': {
            'heatmap_range': [None, None],
            'smoothing_window': 200
        }
    },
    # {
    #     'epoch_name': 'correct',
    #     'event': 'iti_start',
    #     'prewindow': 10,
    #     'postwindow': 30,
    #     'z_score_window': [-10, -5],
    #     'downsample': 10,
    #     'quantification': 'AUC', # options are AUC, median, and mean
    #     'baseline_window': [-6, -3],
    #     'response_window': [0, 3],
    #     'save_file_as': 'iti_start_processed',
    #     'plot_paramaters': {
    #         'heatmap_range': [-2, 2],
    #         'smoothing_window': 1000
    #     }
    # }
]


##################### LUCY SECTION ######################
# z_score_before_alignment = True
# deltaf_options = {
#     'detrend': 'savgol_from_reference',
#     'second_detrend': 'linear',
#     'signal_window_length': 3001,
#     'mode': 'z_score_period', 
#     'period': [1000, 30000],
# } # Any parameters you want to pass when calculating deltaf/f
# signal_channel = 'LMag 1' # Name of our signal channel
# reference_channel = 'LMag 2' # Name of our reference channel

# mode = 'manual'
# path_to_social_excel = [
#     '/Users/DB/Development/Monkey_frog/data/social/FP_41718_PVGHjSI_9949_3_chunky_social.csv',
#     '/Users/DB/Development/Monkey_frog/data/social/FP_41718_PVGHjSI_9949_3_chunky_social.csv'
# ]

# dpaths = [
#     '/Users/DB/Development/Monkey_frog/data/social/TDT-LockinRX8-22Oct2014_20-4-15_DT1_04171819/',
#     '/Users/DB/Development/Monkey_frog/data/social/TDT-LockinRX8-22Oct2014_20-4-15_DT1_04171819/'
# ]


# analysis_blocks = [
#     {
#         'epoch_name': 'active',
#         'event': 'active',
#         'prewindow': 30,
#         'postwindow': 60,
#         'z_score_window': [],
#         'downsample': 10,
#         'quantification': 'mean', # options are AUC, median, and mean
#         'baseline_window': [-5, -2],
#         'response_window': [1, 4],
#         'save_file_as': 'active_processed',
#         'plot_paramaters': {
#             'heatmap_range': [None, None],
#             'smoothing_window': 500
#         }
#     }
# ]

####################### PREPROCESSING DATA ###############################
print('\n\n\n\nRUNNING IN MODE: %s \n\n\n' % mode)
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
        # Extracts the sampling rate from the signal channel
        try:
            sampling_rate = filter(lambda x: x.name == signal_channel, segment.analogsignals)[0].sampling_rate
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
        PrintNoNewLine('\nCalculating delta_f/f...')
        all_signals = ProcessSignalData(seg=segment, sig_ch=signal_channel, ref_ch=reference_channel,
                            name='DeltaF_F', fs=sampling_rate, highcut=40.0, **deltaf_options)
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
        ProcessEvents(seg=segment, tolerance=.1, evtframe=evtframe, 
            name='Events', mode=mode, manualframe=manualframe, 
            event_col='Bout type', start_col='Bout start', end_col='Bout end')
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
            z_score_window = block['z_score_window']
            quantification = block['quantification']
            baseline_window = block['baseline_window']
            response_window = block['response_window']
            save_file_as = block['save_file_as']
            heatmap_range = block['plot_paramaters']['heatmap_range']
            smoothing_window = block['plot_paramaters']['smoothing_window']

            lookup = {}
            for channel in ['Filtered_signal', 'Filtered_reference', 'Detrended', 'DeltaF_F_or_Z_score']:
                print('\nAnalyzing "{}" trials centered around "{}". Channel: "{}" \n'.format(epoch_name, event, channel))

                dict_name = "{}_{}".format(epoch_name, channel)
                lookup[channel] = dict_name 
                PrintNoNewLine('Centering trials and analyzing...')
                AlignEventsAndSignals(seg=segment, epoch_name=epoch_name, analog_ch_name=channel, 
                    event_ch_name='Events', event=event, event_type='label', 
                    prewindow=prewindow, postwindow=postwindow, window_type='event', 
                    clip=False, name=dict_name, to_csv=False, dpath=dpath)
                print('Done!')



    ######################## PROCESS SIGNALS (IF NECESSARY); PLOT; STATS ######
            # Load data

            signal = segment.analyzed[lookup['Filtered_signal']]['all_traces']
            reference = segment.analyzed[lookup['Filtered_reference']]['all_traces']

            # Down sample data
            if downsample > 0:
                signal.reset_index(inplace=True)
                reference.reset_index(inplace=True)
                sample = (signal.index.to_series() / downsample).astype(int)
                signal = signal.groupby(sample).mean()
                reference = reference.groupby(sample).mean()
                signal = signal.set_index('index')
                reference = reference.set_index('index') 

            # # Scale signal if it is too weak (want std to be at least 1)
            # if (np.abs(signal.mean().std()) < 1.) or (np.abs(reference.mean().std()) < 1.):
            #     scale_factor = 10**(np.ceil(np.log10(1/(signal.mean().std()))))

            #     signal = signal * scale_factor
            #     reference = reference * scale_factor

            # Get plotting read
            figure = plt.figure(figsize=(12, 12))
            figure.subplots_adjust(hspace=1.3)
            ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
            ax3 = plt.subplot2grid((6, 2), (4, 0), rowspan=2)
            ax4 = plt.subplot2grid((6, 2), (0, 1), rowspan=3)
            ax5 = plt.subplot2grid((6, 2), (3, 1), rowspan=3)
            # fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
            # fig.set_size_inches(12, 12)

        ############################### PLOT AVERAGE EVOKED RESPONSE ######################
            PrintNoNewLine('Calculating average filtered responses for {} trials...'.format(epoch_name))
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

            # # Scale signal if it is too weak (want std to be at least 1)
            # if (np.abs(signal_avg_response.std()) < 1.) or (np.abs(reference_avg_response.std()) < 1.):
            #     scale_factor = 10**(np.ceil(np.log10(1/(signal_avg_response).std())))

            #     signal_avg_response = signal_avg_response * scale_factor
            #     signal_se = signal_se * scale_factor
            #     reference_avg_response = reference_avg_response * scale_factor
            #     reference_se = reference_se * scale_factor

            # Plotting signal
            # current axis
            #curr_ax = axs[0, 0]
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
        ############################# Calculate detrended signal #################################
            if z_score_before_alignment:
                detrended_signal = segment.analyzed[lookup['Detrended']]['all_traces']
            else:
                # Detrending
                PrintNoNewLine('Detrending signal...')
                fits = np.array([np.polyfit(reference.values[:, i],signal.values[:, i],1) for i in xrange(signal.shape[1])])
                Y_fit_all = np.array([np.polyval(fits[i], reference.values[:,i]) for i in np.arange(reference.values.shape[1])]).T
                Y_df_all = signal.values - Y_fit_all
                detrended_signal = pd.DataFrame(Y_df_all, index=signal.index)

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
            if z_score_before_alignment:
                pass
            else:
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
            if z_score_before_alignment:
                pass
            else:
                curr_ax.legend(['z-score window'])
            curr_ax.axvline(0, color='black', linestyle='--')
            curr_ax.set_ylabel('Voltage (V)')
            curr_ax.set_xlabel('Time (s)')
            curr_ax.set_title('465 nm Average Detrended Signal $\pm$ SEM')

            print('Done!')
        
        # ########### Calculate z-scores ###############################################
            if z_score_before_alignment:
                zscores = segment.analyzed[lookup['DeltaF_F_or_Z_score']]['all_traces']
            else:
                PrintNoNewLine('Calculating Z-Scores for %s trials...' % event)
                # calculate z_scores
                zscores = ZScoreCalculator(detrended_signal, baseline_start=z_score_window[0], 
                    baseline_end=z_score_window[1])
                print('Done!')


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
            if z_score_before_alignment:
                sampling_per = segment.analogsignals[0].sampling_period
                curr_ax.set_title('Z-Score Heat Map \n Baseline Window: {} to {} Seconds'.format(round(deltaf_options['period'][0]*sampling_per), round(deltaf_options['period'][1]*sampling_per)));
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
            baseline_start = zscores[baseline_window[0]:baseline_window[1]].index[0]
            baseline_end = zscores[baseline_window[0]:baseline_window[1]].index[-1]
            response_start = zscores[response_window[0]:response_window[1]].index[0]
            response_end = zscores[response_window[0]:response_window[1]].index[-1]
            baseline_height = zscores[baseline_window[0]:baseline_window[1]].mean(axis=1).min() - 0.5
            response_height = zscores[response_window[0]:response_window[1]].mean(axis=1).max() + .5

            curr_ax.plot([baseline_start, baseline_end], [baseline_height, baseline_height], color='.6', linewidth=3)
            curr_ax.plot([response_start, response_end], [response_height, response_height], color='r', linewidth=3)

            curr_ax.plot(zscores_mean.index, zscores_mean.values, color='b', linewidth=2)
            curr_ax.fill_between(zscores_mean.index, (zscores_mean - zscores_sem).values, 
                (zscores_mean + zscores_sem).values, color='b', alpha=0.05)

            # Plot event onset
            curr_ax.axvline(0, color='black', linestyle='--')

            curr_ax.set_ylabel('Z-Score')
            curr_ax.set_xlabel('Time (s)')
            curr_ax.legend(['baseline window', 'response window'])
            curr_ax.set_title('465 nm Average Z-Score Signal $\pm$ SEM')
            print('Done!')
        ##################### Quantification #################################
            PrintNoNewLine('Performing statistical testing on baseline vs response periods...')
            # Generating summary statistics
            if quantification == 'AUC':
                base = np.trapz(zscores[baseline_window[0]:baseline_window[1]], axis=0)
                resp = np.trapz(zscores[response_window[0]:response_window[1]], axis=0)
                ylabel = 'AUC'
            elif quantification == 'mean':
                base = np.mean(zscores[baseline_window[0]:baseline_window[1]], axis=0)
                resp = np.mean(zscores[response_window[0]:response_window[1]], axis=0)
                ylabel = 'Z-Score'
            elif quantification == 'median':
                base = np.median(zscores[baseline_window[0]:baseline_window[1]], axis=0)
                resp = np.median(zscores[response_window[0]:response_window[1]], axis=0)
                ylabel = 'Z-Score'

            if isinstance(base, pd.core.series.Series):
                base = base.values
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
            if (base_normal[1] >= normal_alpha) or (resp_normal[1] >= normal_alpha):
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
            save_path = dpath + save_file_as
            figure.savefig(save_path + '.png', format='png')
            figure.savefig(save_path + '.pdf', format='pdf')
            plt.close()
            print('Done!')

            # Trial z-scores
            # Fix columns
            zscores.columns = np.arange(1, zscores.shape[1] + 1)
            zscores.columns.name = 'trial'
            # Fix rows 
            zscores.index.name = 'time'
            zscores.to_csv(save_path + '_zscores_aligned.csv')
            # Trial point estimates
            point_estimates = pd.DataFrame({'baseline': base, 'response': resp}, 
                index=np.arange(1, base.shape[0]+1))
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
            smoothed_zscore.to_csv(save_path + '_smoothed_zscores.csv')


    print('Finished processing datapath: %s' % dpath)
