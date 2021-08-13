#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
combine_session.py: Python script that combines data from a single animal across sessions.
"""

__author__ = "DM Brady"
__datewritten__ = "06 Mar 2019"
__lastmodified__ = "06 Mar 2019"

import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from imaging_analysis.utils import PrintNoNewLine
from imaging_analysis.signal_processing import SmoothSignalWithPeriod
import json

sns.set_style('darkgrid')

# This is the list of sessions/epoch types that we group together.

# First we have a list of datapaths (where the data lives)

# Then we have the name, this is how we look for specific epochs. For example, 
# name: 'correct' will look for 'correct_point_estimates.csv' and 'correct_zscores_aligned.csv'

# Finally we have where we want to save the combined data. Will create a folder if it does not exist.
plot_data = True

groupings = [
    {
        'dpaths':
            [
                '/Users/DB/Development/Monkey_frog/data/912_m1/FirstFibPho-180817-160254/correct_processed',
                '/Users/DB/Development/Monkey_frog/data/921_m1/FirstFibPho-180817-160254/correct_processed'
            ],
        'save_folder': '/Users/DB/Development/Monkey_frog/data/m1/',
        'save_filename': 'correct_sessions_combined',
        'plot_paramaters': {
            'heatmap_range': [None, None],
            'smoothing_window': 500,
            'yrange': [-5, 10]
        }
    },
    {
        'dpaths':
            [
                '/Users/DB/Development/Monkey_frog/data/912_m1/FirstFibPho-180817-160254/iti_start_processed',
                '/Users/DB/Development/Monkey_frog/data/921_m1/FirstFibPho-180817-160254/iti_start_processed'
            ],
        'save_folder': '/Users/DB/Development/Monkey_frog/data/m1/',
        'save_filename': 'iti_start_sessions_combined',
        'plot_paramaters': {
            'heatmap_range': [-2, 2],
            'smoothing_window': None,
            'yrange': []
        }
    }
]

for group in groupings:
    dpaths = group['dpaths']
    filename = group['save_filename']
    save_path = group['save_folder'] + os.sep
    heatmap_range = group['plot_paramaters']['heatmap_range']
    smoothing_window = group['plot_paramaters']['smoothing_window']
    yrange = group['plot_paramaters']['yrange']

    print(('Combining sessions of to %s' % (save_path)))
    # See if save_path exists, if not creates a folder
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Group data
    traces = []
    point_estimates = []
    metadata = []
    for dpath in dpaths:
        data = [x for x in glob.glob(dpath +  '*.csv')]
        metadatas = [x for x in glob.glob(dpath +  '*.json')]
        if (len(data) == 0) or (len(metadatas) == 0):
            raise ValueError('Could not find the appropriate files (zscores, point estimates, and metadata). Please check your "dpaths" and "name".')
        
        traces.append([x for x in data if 'zscores_aligned' in x][0])
        point_estimates.append([x for x in data if 'point_estimates' in x][0])

        current_metadata = [x for x in metadatas if 'metadata' in x][0]
        with open(current_metadata, 'r') as fp:
            json_file = json.load(fp)
        metadata.append(json_file)

    # Check metadata
    metadict = {}
    for key in list(metadata[0].keys()):
        value_list = {tuple(x[key]) if isinstance(x[key], list) else x[key] for x in metadata}
        if (len(value_list) != 1) and ('sampling_rate' not in key):
            raise ValueError('You have different analysis parameters (quantification type, baseline window, response window, etc.). Please run process_data.py again with similar values.')
        else:
            metadict[key] = list(value_list)[0]

    # Combine traces into dataframe
    zscores = pd.concat([pd.read_csv(x, index_col=0) for x in traces], axis=1)
    zscores.columns = np.arange(1, zscores.shape[1] + 1)
    zscores.columns.name = 'trial'
    zscores = zscores.ffill().bfill()

    # Recalculate sampling rate because of backward/forward fills
    metadict['downsampled_sampling_rate'] = zscores.shape[0]/(np.abs(zscores.index[0]) + np.abs(zscores.index[-1]))

    # Combine point estimates into dataframe
    pe_df = pd.concat([pd.read_csv(x, index_col=0) for x in point_estimates], axis=0)
    pe_df.index = np.arange(1, pe_df.shape[0]+1)
    pe_df.index.name = 'trial'

    # Save combined data
    zscores.to_csv(save_path + filename + '_zscores_aligned.csv')
    pe_df.to_csv(save_path + filename + '_point_estimates.csv')

    # Save metadata
    with open(save_path + filename + '_metadata.json', 'w') as fp:
        json.dump(metadict, fp)

    if plot_data:
        # Get plotting read
        figure = plt.figure(figsize=(12, 12))
        figure.subplots_adjust(hspace=1.3)
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (2, 0), rowspan=2, colspan=1)
        ax3 = plt.subplot2grid((6, 1), (4, 0), rowspan=2, colspan=1)

        baseline_window = metadict['baseline_window']
        response_window = metadict['response_window']
        quantification = metadict['quantification']
        sampling_rate = float(metadict['downsampled_sampling_rate'])
    ############################ Make rasters #######################################
        PrintNoNewLine('Making heatmap...')
        # indice that is closest to event onset
        # curr_ax = axs[0, 1]
        curr_ax = ax1
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
        curr_ax.set_title('Z-Score Heat Map');
        print('Done!')
    ########################## Plot Z-score waveform ##########################
        PrintNoNewLine('Plotting Z-Score waveforms...')
        zscores_mean = zscores.mean(axis=1)

        zscores_sem = zscores.sem(axis=1)


        if smoothing_window is not None:
            zscores_mean = SmoothSignalWithPeriod(x=zscores_mean, sampling_rate=sampling_rate, 
                ms_bin=smoothing_window, window='flat')
            zscores_sem = SmoothSignalWithPeriod(x=zscores_sem, sampling_rate=sampling_rate, 
                ms_bin=smoothing_window, window='flat')

        # Save smoothed data
        smoothed_zscore = pd.concat([zscores_mean, zscores_sem], axis=1)
        smoothed_zscore.columns = ['mean', 'sem']
        smoothed_zscore.to_csv(save_path + filename + '_smoothed_zscores.csv')
        # Plotting signal
        # current axis
        # curr_ax = axs[1, 1]
        curr_ax = ax2
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

        # set ylim
        if len(yrange) == 2:
            curr_ax.set_ylim(yrange)

        curr_ax.set_ylabel('Z-Score')
        curr_ax.set_xlabel('Time (s)')
        curr_ax.legend(['baseline window', 'response window'])
        curr_ax.set_title('465 nm Average Z-Score Signal $\pm$ SEM')
        print('Done!')
    ##################### Quantification #################################
        PrintNoNewLine('Performing statistical testing on baseline vs response periods...')
        # Generating summary statistics
        if (quantification == 'AUC'):
            ylabel = 'AUC'
        else:
            ylabel = 'Z-Score'

        base = pe_df['baseline'].values
        resp = pe_df['response'].values

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
        curr_ax = ax3
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

        figure.savefig(save_path + filename + '.png', format='png')
        figure.savefig(save_path + filename + '.pdf', format='pdf')
        plt.close()

        print('Done!')
