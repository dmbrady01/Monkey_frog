#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
markov.py: Python script that contains functions for making markov models.
"""


__author__ = "DM Brady"
__datewritten__ = "19 Jul 2018"
__lastmodified__ = "31 Jul 2018"


# Required modules
import pandas as pd
import numpy as np
import os
import json
import scipy.sparse.linalg as sla
from scipy.stats import wasserstein_distance
from imaging_analysis.event_processing import FormatManualExcelFile, ReadManualExcelFile

def ReadStateCsv(state_csv='markov/states.csv'):
    "Reads the possible state csv"
    if os.path.isfile(state_csv): 
        return ReadManualExcelFile(state_csv)
    else:
        df = pd.DataFrame(columns=['states'])
        df.to_csv(state_csv, index=False)
        return df

def GetTransitionsFromExcel(excel_file=None, column='Bout type', resample=False, resample_window=1):
    "Reads the excel file to be processed and returns the Series we care about"
    return FormatManualExcelFile(excel_file=excel_file, event_col=column, 
        resample=resample, resample_window=resample_window)[column]

# def AddStartAndEnds(df):
#     "Adds starts and ends to Series with transitions"
#     return pd.concat([pd.Series(['start']), df, pd.Series(['end'])]).reset_index(drop=True)

def StateMapping(df, state_csv='markov/states.csv', fixed_states_csv=False):
    "Reads the current transition sequence to be analyzed and updates states_csv. Returns numerical codes."
    set_df = set(df.unique())
    # Read state csv
    state_code = ReadStateCsv(state_csv=state_csv)
    set_state = set(state_code['states'].unique())
    # See if states in set_df are not in set_state
    if not fixed_states_csv:
        not_mapped = set_df.difference(set_state)
        if len(not_mapped) > 0:
            state_code = state_code.append(pd.DataFrame(list(not_mapped), 
                columns=['states']), ignore_index=True)
            state_code.to_csv(state_csv, index=False)

    return {y:x for x, y in state_code['states'].to_dict().items()}

def ExcelToStateMapping(excel_file, column='Bout type', state_csv='markov/states.csv', 
        fixed_states_csv=False, resample=False, resample_window=1):
    "Reads an excel file and states csv and updates states csv. Outputs dataframe and code."
    df = GetTransitionsFromExcel(excel_file=excel_file, column=column, resample=resample, 
        resample_window=resample_window)
    code = StateMapping(df, state_csv=state_csv, fixed_states_csv=fixed_states_csv)
    return df, code

def EncodeStates(df, code):
    "Given a dataframe and a state_code, creates a list of transitions"
    transitions = df.map(code).values
    # remove nan transitions (not in code)
    transitions = transitions[[~np.isnan(x) for x in transitions]]
    return transitions.astype(int)

def CountMatrix(transitions, num_states=None):
    """Given a list of transitions and the code, creates a count matrix. If numpy array
    is already passed, it just passes through."""
    if isinstance(transitions, np.ndarray):
        return transitions
    else:
        if not num_states:
            num_states = int(1 + max(transitions))
        else:
            num_states = int(1 + num_states)

        matrix = [[0]*num_states for _ in range(num_states)]

        for (i,j) in zip(transitions,transitions[1:]):
            matrix[i][j] += 1.0

        return np.array(matrix)

def ProcessExcelToCountMatrix(excel_file, column='Bout type', state_csv='markov/states.csv', 
        fixed_states_csv=False, resample=False, resample_window=1):
    "Reads an excel file and states csv, updates states csv, outputs count matrix"
    df, code = ExcelToStateMapping(excel_file=excel_file, column=column, state_csv=state_csv, 
        fixed_states_csv=fixed_states_csv, resample=resample, resample_window=resample_window)
    transitions = EncodeStates(df, code)
    count_matrix = CountMatrix(transitions)
    return count_matrix, transitions

def StochasticMatrix(count_matrix, replace_nan=True, calc='right'):
    "Takes a count matrix and generates a stochastic matrix"
    if calc == 'right':
        tm = np.array([np.divide(count_matrix[:,x], count_matrix.sum(axis=1)) 
            for x in range(count_matrix.shape[0])]).T
    elif calc == 'left':
         tm = np.array([np.divide(count_matrix[x, :], count_matrix.sum(axis=0)) 
            for x in range(count_matrix.shape[1])])       
    elif calc == 'probability':
        tm = np.array(np.divide(count_matrix, count_matrix.sum()))
    if replace_nan:
        tm[np.isnan(tm)] = 0
    return tm

def StationaryDistribution(tm):
    "Assumes a right stochastic matrix (if not pass it as mymatrix.T). Returns stationary distro"
    first_eval, first_evec = sla.eigs(tm.T, k=1, which='LM')
    return (first_evec/first_evec.sum()).real

def DistanceBewtweenMatrices(matrix1, matrix2, method=wasserstein_distance):
    "Finds distance between two matrices using method (must pass a function)"
    return method(matrix1.flat, matrix2.flat)

def AddEmptyRowAndColumn(matrix, goal_shape=10):
    shape = matrix.shape[0]
    num = goal_shape - shape
    if num > 0:
        new_matrix = np.zeros((shape + num, shape + num))
        new_matrix[:-num, :-num] = matrix
        return new_matrix
    else:
        return matrix

def AddingCountMatrices(list_of_count_matrices):
    shapes_list = [x.shape[0] for x in list_of_count_matrices]
    max_shape = max(shapes_list)
    reshaped = [AddEmptyRowAndColumn(x, max_shape) for x in list_of_count_matrices]
    sum_of_matrices = reshaped[0]
    for matrix in reshaped[1:]:
        sum_of_matrices = np.add(sum_of_matrices, matrix)
    return sum_of_matrices

def MarkovToTransitionMatrix(list_of_data, mode='transitions', num_states=None, replace_nan=True, calc='right'):
    if mode == 'transitions':
        return StochasticMatrix(AddingCountMatrices([
            CountMatrix(data, num_states=num_states) for data in list_of_data]), 
            replace_nan=replace_nan, calc=calc)
    elif mode == 'counts':
        return StochasticMatrix(AddingCountMatrices(list_of_data), 
            replace_nan=replace_nan, calc=calc)

def ReadTransitionsOrCountMatrixFromFile(text_file):
    """Reads transitions.txt file and returns list of transitions (list of lists)
    Or reads count matrix.csv file and returns numpy array"""
    if isinstance(text_file, str):
        with open(text_file) as fp:
            contents = fp.read()
        contents = contents.split('\n')
        contents = [json.loads(x) for x in contents if len(x) > 0]
    else:
        contents = [pd.read_csv(x, header=None).values for x in text_file]
    return contents

def MaxStates(*args):
    "Given a set of transitions, finds out how many states there are."
    return max(max(max(x) for x in args)) + 1
