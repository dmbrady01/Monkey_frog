#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
transition_matrix.py: Python script that contains functions for making markov models.
"""


__author__ = "DM Brady"
__datewritten__ = "19Aug2018"
__lastmodified__ = "09Aug2018"


import numpy as np
import pandas as pd
import os
from .markov.markov import ProcessExcelToCountMatrix, AddingCountMatrices, StochasticMatrix, ExcelToStateMapping, ReadStateCsv

cohorts = [
    {
        'name': 'group_housed',
        'savepath': '/Users/DB/Development/Monkey_frog/data/social/csvs',
        'statespath': 'markov/group_housed_states.csv',
        'column': 'Bout type',
        'fixed_states_csv': False,
        'resample': True,
        'resample_window': 2,
        'csvs': [
            '/Users/DB/Development/Monkey_frog/data/social/csvs/PVFP_1024_362_social.xlsx',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9949_3_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9924_2_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_1024_372_social.xlsx',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_1024_371_social.xlsx',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9919_2_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9949_1_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_3949_2_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_1024_373_social.xlsx'
            ]
    },
    {
        'name': 'isolates',
        'savepath': '/Users/DB/Development/Monkey_frog/data/social/csvs',
        'statespath': 'markov/isolates_states.csv',
        'column': 'Bout type',
        'fixed_states_csv': False,
        'resample': True,
        'resample_window': 2,
        'csvs': [
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9919_4_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_1024_363_social.xlsx',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9949_2_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_3949_1_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9919_1_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9924_1_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_9919_3_social.csv',
            '/Users/DB/Development/Monkey_frog/data/social/csvs/FP_41718_PVGHjSI_3949_3_social.csv'
            ]
    }
]
all_cohort_states = True
all_cohort_states_path = 'markov/states.csv'
all_column = 'Bout type'
# paths = [
#     '/Users/DB/Development/Monkey_frog/data/social/FP_example_object.csv',
#     '/Users/DB/Development/Monkey_frog/data/social/FP_example_social.csv'
# ]
# path_to_save = '/Users/DB/Development/Monkey_frog/data/social/'
# Build up your count matrices
if all_cohort_states:
    # all_data = [cohort['csvs'] for cohort in cohorts]
    # all_data = [item for sublist in all_data for item in sublist]
    state_set_list = []
    
    for cohort in cohorts:
        for csv in cohort['csvs']:
            # builds up all the transitions per group
            _, _ = ExcelToStateMapping(csv, column=all_column, state_csv='temp.csv')
        # reads the transitions per group
        df = ReadStateCsv('temp.csv')
        cohort_set = set(df['states'].unique())
        state_set_list.append(cohort_set)
        os.remove('temp.csv')
        cohort['column'] = all_column
        cohort['statespath'] = all_cohort_states_path
        cohort['fixed_states_csv'] = True

    intersection_of_states = list(set.intersection(*state_set_list))
    state_code = pd.DataFrame(intersection_of_states, columns=['states'])
    state_code.to_csv(all_cohort_states_path, index=False)
    
    # for csv in all_data:
    #     _, _ = ExcelToStateMapping(csv, column=all_column, state_csv=all_cohort_states_path)


for cohort in cohorts:
    count_matrices = []
    transitions_list = []
    for csv in cohort['csvs']:
        count_matrix, transitions = ProcessExcelToCountMatrix(csv, column=cohort['column'], 
            state_csv=cohort['statespath'], fixed_states_csv=cohort['fixed_states_csv'],
            resample=cohort['resample'], resample_window=cohort['resample_window'])
        count_matrices.append(count_matrix)
        transitions_list.append(list(transitions))


    with open(cohort['savepath'] + os.sep + cohort['name'] + "_transitions.txt", 'w') as fp:
        for transition in transitions_list:
            fp.write(str(transition) + "\n")

    # Add all count matrices and get transistion matric
    total_count_matrix = AddingCountMatrices(count_matrices)
    right_transistion_matrix = StochasticMatrix(total_count_matrix, calc='right')
    left_transistion_matrix = StochasticMatrix(total_count_matrix, calc='left')
    joint_distribution_matrix = StochasticMatrix(total_count_matrix, calc='probability')
    # TO SAVE csv
    np.savetxt(cohort['savepath'] + os.sep + cohort['name'] + "_count_matrix.csv", 
        total_count_matrix, delimiter=",")
    np.savetxt(cohort['savepath'] + os.sep + cohort['name'] + "_right_transition_matrix.csv", 
        right_transistion_matrix, delimiter=",")
    np.savetxt(cohort['savepath'] + os.sep + cohort['name'] + "_left_transition_matrix.csv", 
        left_transistion_matrix, delimiter=",")
    np.savetxt(cohort['savepath'] + os.sep + cohort['name'] + "_joint_distribution_matrix.csv", 
        joint_distribution_matrix, delimiter=",")




