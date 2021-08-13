#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
markov_permutation_test.py: Python script for performing permutation test on two sets of transition.txts.
"""


__author__ = "DM Brady"
__datewritten__ = "31Jul2018"
__lastmodified__ = "09Aug2018"

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .markov.markov import ReadTransitionsOrCountMatrixFromFile, MaxStates, MarkovToTransitionMatrix, StationaryDistribution, DistanceBewtweenMatrices, ReadStateCsv
import json
import os

# Paths to transition.txt files
group1 = ['/Users/DB/Downloads/group_housed_count_matrix.csv', '/Users/DB/Downloads/group_housed_count_matrix.csv']
group2 = ['/Users/DB/Downloads/isolates_count_matrix.csv', '/Users/DB/Downloads/isolates_count_matrix.csv']
state_csv = '/Users/DB/Downloads/states.csv'
savepath = '/Users/DB/Development/Monkey_frog/'

# can be 'stationary', 'joint', or 'transition'. last will correct for multiple comparisons
analysis_type = 'transition'
right_or_left = 'right'

# Number of permutations to run
num_permutations = 10000
np.random.seed(1234) # sets the random seed generator to get the same distro each time

# Read data into list of lists
group1_data = ReadTransitionsOrCountMatrixFromFile(group1)
group2_data = ReadTransitionsOrCountMatrixFromFile(group2)

# Get max number of states
#num_states = MaxStates(group1_data, group2_data)
states = ReadStateCsv(state_csv)
num_states = states.shape[0]

true_distance = []
# Real differences (finds the stationary distro for each)
if analysis_type == 'stationary':
    group1_sd = StationaryDistribution(MarkovToTransitionMatrix(group1_data, 
        num_states=num_states, replace_nan=True, calc=right_or_left))
    group2_sd = StationaryDistribution(MarkovToTransitionMatrix(group2_data, 
        num_states=num_states, replace_nan=True, calc=right_or_left))
    true_distance.append(DistanceBewtweenMatrices(group1_sd, group2_sd))
    null_distribution = [[] for i in range(1)]
elif analysis_type == 'joint':
    group1_sd = MarkovToTransitionMatrix(group1_data, num_states=num_states, 
        replace_nan=True, calc='probability')
    group2_sd = MarkovToTransitionMatrix(group2_data, num_states=num_states, 
        replace_nan=True, calc='probability')
    true_distance.append(DistanceBewtweenMatrices(group1_sd, group2_sd))
    null_distribution = [[] for i in range(1)]
elif analysis_type == 'transition':
    null_distribution = [[] for i in range(num_states)]
    for i in range(num_states):
        if right_or_left == 'right':
            group1_sd = MarkovToTransitionMatrix(group1_data, num_states=num_states, 
                replace_nan=True, calc=right_or_left)[i,:]
            group2_sd = MarkovToTransitionMatrix(group2_data, num_states=num_states, 
                replace_nan=True, calc=right_or_left)[i,:]
            true_distance.append(DistanceBewtweenMatrices(group1_sd, group2_sd))
        else:
            group1_sd = MarkovToTransitionMatrix(group1_data, num_states=num_states, 
                replace_nan=True, calc=right_or_left)[:, i]
            group2_sd = MarkovToTransitionMatrix(group2_data, num_states=num_states, 
                replace_nan=True, calc=right_or_left)[:, i]
            true_distance.append(DistanceBewtweenMatrices(group1_sd, group2_sd))

# Preparing for permutation test
group1_sample_size = len(group1_data)
group2_sample_size = len(group2_data)
total_n = group1_sample_size + group2_sample_size
combined_data = np.array(group1_data + group2_data)

while len(null_distribution[0]) < num_permutations:
    # random permutation
    shuffle = np.random.permutation(total_n).astype(int)
    group1_idx = shuffle[:group1_sample_size]
    group2_idx = shuffle[group1_sample_size:]

    # Shuffled data
    group1_perm = combined_data[group1_idx]
    group2_perm = combined_data[group2_idx]

    # Calcualte stationary distributions
    if analysis_type == 'stationary':
        group1_perm_sd = StationaryDistribution(MarkovToTransitionMatrix(group1_perm, 
            num_states=num_states, replace_nan=True, calc=right_or_left))
        group2_perm_sd = StationaryDistribution(MarkovToTransitionMatrix(group2_perm, 
            num_states=num_states, replace_nan=True, calc=right_or_left))
        perm_distance = DistanceBewtweenMatrices(group1_perm_sd, group2_perm_sd)
        null_distribution[0].append(perm_distance)
    elif analysis_type == 'joint':
        group1_perm_sd = MarkovToTransitionMatrix(group1_perm, num_states=num_states, 
            replace_nan=True, calc='probability')
        group2_perm_sd = MarkovToTransitionMatrix(group2_perm, num_states=num_states, 
            replace_nan=True, calc='probability') 
        perm_distance = DistanceBewtweenMatrices(group1_perm_sd, group2_perm_sd)
        null_distribution[0].append(perm_distance)
    elif analysis_type == 'transition':
        for i in range(num_states):
            if right_or_left == 'right':
                group1_perm_sd = MarkovToTransitionMatrix(group1_perm, num_states=num_states, 
                    replace_nan=True, calc=right_or_left)[i,:]
                group2_perm_sd = MarkovToTransitionMatrix(group2_perm, num_states=num_states, 
                    replace_nan=True, calc=right_or_left)[i,:]
                perm_distance = DistanceBewtweenMatrices(group1_perm_sd, group2_perm_sd)
                null_distribution[i].append(perm_distance)   
            else:
                group1_perm_sd = MarkovToTransitionMatrix(group1_perm, num_states=num_states, 
                    replace_nan=True, calc=right_or_left)[:, i]
                group2_perm_sd = MarkovToTransitionMatrix(group2_perm, num_states=num_states, 
                    replace_nan=True, calc=right_or_left)[:, i]
                perm_distance = DistanceBewtweenMatrices(group1_perm_sd, group2_perm_sd)
                null_distribution[i].append(perm_distance)       
 

# For plotting
null_distribution = [np.array(x) for x in null_distribution]
num_hypotheses = len(null_distribution)
p_value = []
for j in range(num_hypotheses):
    p_value.append(np.sum(null_distribution[j] >= true_distance[j])/float(null_distribution[j].shape[0]))

# For doing a multiple comparisons correction (bonferroni-holm)
p_value_order = np.argsort(p_value) # start with the lowest p-value
significant = []
for j in range(num_hypotheses):
    row_number = p_value_order[j] # Find which state has the lowest p-value
    dist = true_distance[row_number] # get that distance
    p_val = p_value[row_number] # get that p-value
    null_dist = null_distribution[row_number] # get the null distro for that row
    null_p = (1 - .05/(num_hypotheses-j))*100 # calculate the corrected cutoff percentile
    cut_off = np.percentile(null_dist, null_p) # calculate the actual value associated with cutoff percentile
    is_sig = dist > cut_off # see if true distance is further from cut off  
    
    # Checks if any of the previous tests are failures, if so automatically sets the next
    # test as a failure
    # Also gets the state from row_number if applicable
    if j > 0: # set all future tests to false significance if earlier one failed
        if any(~np.array([bool(significant[x]['significant']) for x in range(len(significant))])):
            is_sig = False
        state = states.iloc[row_number, 0]
    elif analysis_type == 'transition':
        state = states.iloc[row_number, 0]
    else:
        state = analysis_type

    # Build up a dictionary with all relevant info
    significant.append({'state': state, 'significant': str(is_sig), 'distance': float(dist), 
        'p_value': float(p_val), 'alpha_p_val': float((100-null_p)/100.0), 'alpha_distance': float(cut_off)})

    # Plotting
    histplot = sns.distplot(null_dist, norm_hist=True)
    histplot.set_title('Significant: ' + str(is_sig))
    histplot.axvline(cut_off, color='r', linestyle='--')
    histplot.axvline(dist, color='k', linestyle='-')
    fig = histplot.get_figure()

    # Saving plot
    if analysis_type == 'transition':
        fig.savefig(savepath + os.sep + 'permutation_test_' + analysis_type + '_starting_state_' + state + '.pdf')
    else:
        fig.savefig(savepath + os.sep + 'permutation_test_' + analysis_type + '.pdf')
    plt.close(fig)

# Writing dictionary to json
with open(savepath + os.sep + 'permutation_test_significance_' + analysis_type + '.json', 'w') as fout:
    json.dump(significant, fout, indent=4, separators=(',', ': '))
