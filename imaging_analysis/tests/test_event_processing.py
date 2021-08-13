#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_event_processing.py: Python script that contains tests for event_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "11 Jun 2018"

# Import unittest modules and event_processing
import unittest
from mock import patch
from neo.core import Event, Segment
import neo
import quantities as pq
import numpy as np
import pandas as pd
from collections import OrderedDict
from imaging_analysis.event_processing import LoadEventParams
from imaging_analysis.event_processing import TruncateEvent
from imaging_analysis.event_processing import TruncateEvents
from imaging_analysis.event_processing import ExtractEventsToList
from imaging_analysis.event_processing import ProcessEventList
from imaging_analysis.event_processing import ProcessEvents
from imaging_analysis.event_processing import ResultOfTrial
from imaging_analysis.event_processing import ProcessTrials
from imaging_analysis.event_processing import CalculateStartsAndDurations
from imaging_analysis.event_processing import GroupTrialsByEpoch
from imaging_analysis.event_processing import GetImagingDataTTL

class TestLoadEventParams(unittest.TestCase):
    "Code tests for LoadEventParams function"
    
    def setUp(self):
        self.dpath = 'imaging_analysis/ttl_event_params.json'
        self.evtdict = OrderedDict({
            'channels': ['ch1'], 
            'events': {
                'one': {
                    'code': [1],
                    'plot': '-k',
                    'type': 'results'
                },
                'two': {
                    'code': [0],
                    'plot': 'b',
                    'type': 'start'
                }
            },
            'startoftrial': ['start'],
            'endoftrial': ['results'],
            'epochs': ['results']
        })
        self.codedf = pd.DataFrame(data=[[1], [0]], index=['one', 'two'], 
            columns=['ch1'])
        self.codedf.index.name = 'event'
        self.plotdf = pd.DataFrame(data=[['-k'], ['b']], index=['one', 'two'],
            columns=['plot'])
        self.plotdf.index.name = 'event'
        self.typedf = pd.DataFrame(data=[['results'], ['start']], index=['one', 'two'],
            columns=['type'])
        self.typedf.index.name = 'event'
        self.mode1 = 'TTL'
        self.mode2 = 'manual'
        self.badmode = 'not a mode'

    def tearDown(self):
        del self.dpath
        del self.evtdict
        del self.codedf
        del self.plotdf
        del self.typedf
        del self.mode1 
        del self.mode2 
        del self.badmode

    def test_returns_a_dataframe(self):
        "Makes sure function returns a dataframe and checks dpath works"
        start, end, epochs, code, typedf = LoadEventParams(dpath=self.dpath)
        self.assertIsInstance(code, pd.core.frame.DataFrame)

    def test_makes_sure_dpath_is_correct(self):
        "Makes sure dpath is correct is supplied"
        self.assertRaises(IOError, LoadEventParams, dpath='/not/path')

    def test_makes_sure_evtdict_is_correct(self):
        "Makes sure evtdict is correct is supplied"
        self.assertRaises(TypeError, LoadEventParams, evtdict='not a dict')

    def test_makes_sure_mode_is_correct(self):
        "Makes sure mode is correct is supplied"
        self.assertRaises(ValueError, LoadEventParams, evtdict=self.evtdict,
            mode=self.badmode)

    def test_makes_sure_startoftrial_is_returned(self):
        "Makes sure start of trials is returned"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(start, self.evtdict['startoftrial'])

    def test_makes_sure_endoftrial_is_returned(self):
        "Makes sure end of trials is returned"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(end, self.evtdict['endoftrial'])

    def test_makes_sure_epochs_is_returned(self):
        "Makes sure epochs is returned"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(epochs, self.evtdict['epochs'])

    def test_makes_sure_code_dataframe_has_right_structure_TTL(self):
        "Checks code dataframe has right structure for TTL mode"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict, 
            mode=self.mode1)
        code.sort_index(inplace=True)
        self.codedf.sort_index(inplace=True)
        pd.testing.assert_frame_equal(self.codedf, code)

    def test_makes_sure_code_dataframe_has_right_structure_manual(self):
        "Checks code dataframe has right structure for manual mode (empty df)"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict, 
            mode=self.mode2)
        pd.testing.assert_frame_equal(pd.DataFrame(), code)

    # def test_makes_sure_plot_dataframe_has_right_structure(self):
    #     "Checks plot dataframe has right structure"
    #     start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict)
    #     plot.sort_index(inplace=True)
    #     self.plotdf.sort_index(inplace=True)
    #     pd.testing.assert_frame_equal(self.plotdf, plot)

    def test_makes_sure_results_dataframe_has_right_structure(self):
        "Checks results dataframe has right structure"
        start, end, epochs, code, typedf = LoadEventParams(evtdict=self.evtdict)
        typedf.sort_index(inplace=True)
        self.typedf.sort_index(inplace=True)
        pd.testing.assert_frame_equal(self.typedf, typedf)



class TestTruncateEvent(unittest.TestCase):
    "Code tests for the TruncateEvent function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.not_evt = np.random.randn(1000, 1)
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.not_evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end

    def test_makes_sure_events_are_passed(self):
        "Test to make sure event object is passed"
        self.assertRaises(TypeError, TruncateEvent, self.not_evt)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start"
        trunc_sig = TruncateEvent(self.evt, start=self.evt_start)
        times = trunc_sig.times
        self.assertFalse(self.evt_pre_start in times)

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end"
        trunc_sig = TruncateEvent(self.evt, end=self.evt_end)
        times = trunc_sig.times
        self.assertFalse(self.evt_post_end in times)



class TestTruncateEvents(unittest.TestCase):
    "Code tests for TruncateEvents function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 50))
        self.events = [self.evt, self.evt2]
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end
        del self.events

    def test_makes_sure_list_is_passed(self):
        "Makes sure list is passed to function"
        self.assertRaises(TypeError, TruncateEvents, 100)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start in event_list"
        trunc_sig = TruncateEvents(self.events, start=self.evt_start)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_pre_start in x for x in times))

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end in event_list"
        trunc_sig = TruncateEvents(self.events, end=self.evt_end)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_post_end in x for x in times))



class TestExtractEventsToList(unittest.TestCase):
    "Code tests for ExtractEventsToList"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])
        self.df2 = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch3', 'Ch4'])
        self.df3 = pd.DataFrame(data=[[1, 0, 0], [1, 1, 1]], 
            index=['start', 'stop'], columns=['Ch1', 'Ch2', 'Ch3'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment 
        del self.df
        del self.df2
        del self.df3

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ExtractEventsToList, seg='not a seg object')
    
    def test_evtframe_must_be_a_dataframe_object(self):
        "Makes sure evtframe is a pandas datafrane object"
        self.assertRaises(TypeError, ExtractEventsToList, evtframe='not a dataframe')

    def test_seg_events_must_have_channel_names_that_match_evtframe_columns(self):
        """Makes sure at least one of the event objects has a name that matches
        the columns in dataframe."""
        self.assertRaises(ValueError, ExtractEventsToList, seg=self.segment, 
            evtframe=self.df2)

    def test_eventlist_length_matches_evtframe_columns_length(self):
        """Makes that event list length == evetframe.columns.shape[0]"""
        self.assertRaises(ValueError, ExtractEventsToList, seg=self.segment, 
            evtframe=self.df3)

    def test_list_is_returned(self):
        "Checks that list is returned"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        self.assertIsInstance(output, list)

    def test_list_of_dicts_is_returned(self):
        "Checks that list of dicts is returned"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        check = [isinstance(x, dict) for x in output]
        self.assertTrue(all(check))

    def test_time_name_works(self):
        "Checks that dictionary key is set to time_name"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df, 
            time_name='myname')
        dict_keys = [list(x.keys()) for x in output]
        dict_check = ['myname' in x for x in dict_keys]
        self.assertTrue(all(dict_check))

    def test_ch_name_works(self):
        "Checks that dictionary key is set to ch_name"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df, 
            ch_name='myname')
        dict_keys = [list(x.keys()) for x in output]
        dict_check = ['myname' in x for x in dict_keys]
        self.assertTrue(all(dict_check))

    def test_ch_values_are_integers_that_match_evtframe_column_indices(self):
        "Checks that values from ch_name match evtframe column indices"
        column_indices = [x for x in range(self.df.columns.shape[0])]
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        output_values = [x['ch'] for x in output]
        self.assertEqual(column_indices, output_values)

    def test_time_values_are_extracted_properly(self):
        "Checks that time values are extracted properly"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        output_values = [x['times'] for x in output]
        check = [self.evt.times, self.evt2.times]
        zipped = list(zip(check, output_values))
        check_zip = [np.array_equal(x, y) for x, y in zipped]
        self.assertTrue(all(check_zip))



class TestProcessEventList(unittest.TestCase):
    "Code tests for ProcessEventList"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.eventlist = [{'ch': 0, 'times': self.evt.times}, \
                        {'ch': 1, 'times': self.evt2.times}]
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.df

    def test_tolerance_must_be_number(self):
        "Makes sure tolerance is a number"
        self.assertRaises(TypeError, ProcessEventList, tolerance='not a number')

    def test_eventlist_is_a_list(self):
        "Makes sure eventlist is a list"
        self.assertRaises(TypeError, ProcessEventList, eventlist='not a list')

    def test_ch_name_must_be_a_key(self):
        "Makes sure ch_name is a key in dictionaries in eventlist"
        self.assertRaises(ValueError, ProcessEventList, eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df, ch_name='not_channel_name')

    def test_time_name_must_be_a_key(self):
        "Makes sure time_name is a key in dictionaries in eventlist"
        self.assertRaises(ValueError, ProcessEventList, eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df, time_name='not_time_name')

    def test_returns_two_variables(self):
        "Checks that two variables are returned"
        output = ProcessEventList(eventlist=self.eventlist, tolerance=1, 
            evtframe=self.df)
        self.assertEqual(len(output), 2)

    def test_event_times_is_a_list(self):
        "Checks returned event_times is a list"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        self.assertIsInstance(eventtimes, list)

    def test_event_labels_is_a_list(self):
        "Checks returned event_labels is a list"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        self.assertIsInstance(eventlabels, list)

    def test_event_times_are_correct(self):
        "Checks that events times are correct"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        equal = np.array_equal(np.array(eventtimes), np.arange(0, 100 , 1)*pq.s)
        self.assertTrue(equal)     

    def test_event_labels_are_correct(self):
        "Checks that events labels are correct"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=0.1, evtframe=self.df)
        test_array = np.arange(0, 100, 1)
        mask_array = np.arange(0, 100, 3)
        test_array = test_array.astype('S')
        test_array[mask_array] = 'stop'
        test_array[np.where(test_array != 'stop')[0]] = 'start'
        equal = np.array_equal(eventlabels, test_array)
        self.assertTrue(equal)     



class TestProcessEvents(unittest.TestCase):
    "Code tests for ProcessEvents function"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ProcessEvents, seg='not a segment')

    def test_event_object_is_added(self):
        "Makes sure event object is added to seg"
        init_len = len(self.segment.events)
        ProcessEvents(seg=self.segment, tolerance=1, evtframe=self.df)
        self.assertTrue(init_len + 1, len(self.segment.events))



class TestResultOfTrial(unittest.TestCase):
    "Code tests for ResultOfTrial function"

    def setUp(self):
        self.result = ['omission']
        self.results_mult = ['omission', 'premature']

    def tearDown(self):
        del self.result 
        del self.results_mult

    def test_listtocheck_is_a_list(self):
        "Makes sure listtocheck is a list"
        self.assertRaises(TypeError, ResultOfTrial, listtocheck='not a list')

    def test_noresults_label(self):
        "Checks that noresults is returned of empty list"
        self.assertEqual(ResultOfTrial(listtocheck=[], noresults='NOPE'), 'NOPE')

    def test_result_is_returned_if_length_one(self):
        "Checks that results is returned if length of one"
        self.assertEqual(ResultOfTrial(listtocheck=self.result), self.result[0])

    def test_multipleresults_label(self):
        "Checks that multiple is returned of list containing multiple entries"
        self.assertEqual(ResultOfTrial(listtocheck=self.results_mult, 
            multipleresults='MANY'), 'MANY')

    def test_multipleresults_label_and_append(self):
        """Checks that multiple and events are returned of list containing 
        multiple entries and append is true"""
        self.assertEqual(ResultOfTrial(listtocheck=self.results_mult, 
            multipleresults='MANY', appendmultiple=True), 'MANY_omission_premature')



class TestProcessTrials(unittest.TestCase):
    "Code tests for ProcessTrials function"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])
        self.startoftrial = ['start']
        self.epochs = ['results']
        self.name = 'MyEvents'
        self.typeframe = pd.DataFrame(data=['start', 'results'], columns=['type'], 
            index=['start', 'stop'])
        ProcessEvents(seg=self.segment, tolerance=1, evtframe=self.df, name=self.name)
        self.columns = ['time', 'event', 'trial_idx', 'results', \
            'with_previous_results', 'event_type']

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment
        del self.df 
        del self.startoftrial 
        del self.epochs
        del self.name 
        del self.typeframe 
        del self.columns

    def test_startoflist_is_a_list(self):
        "Makes sure startoflist is a list"
        self.assertRaises(TypeError, ProcessTrials, startoftrial='not a list')

    def test_epochs_is_a_list(self):
        "Makes sure epochs is a list"
        self.assertRaises(TypeError, ProcessTrials, startoftrial=self.startoftrial, 
            epochs='not an epoch')

    def test_typedf_is_a_dataframe(self):
        "Makes sure typedf is a dataframe"
        self.assertRaises(TypeError, ProcessTrials, startoftrial=self.startoftrial, 
            epochs=self.epochs, typedf='not a dataframe')

    def test_seg_is_a_seg_object(self):
        "Makes sure seg is a segment object"
        self.assertRaises(TypeError, ProcessTrials, startoftrial=self.startoftrial, 
            epochs=self.epochs, typedf=self.df, seg='not a seg object')

    def test_process_trials_returns_error_if_name_incorrect(self):
        "Makes sure IndexError is returned if name is incorrect"
        self.assertRaises(IndexError, ProcessTrials, seg=self.segment, 
            name='Wrong Name', startoftrial=self.startoftrial, 
            epochs=self.epochs, typedf=self.typeframe)

    def test_process_trials_returns_dataframe(self):
        "Makes sure dataframe is returned"
        output = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, typedf=self.typeframe)
        self.assertIsInstance(output, pd.core.frame.DataFrame)

    def test_process_trials_starts_with_first_trial_default(self):
        "Makes sure dataframe only has events from first trial if firsttrail=True"
        output = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=True)
        self.assertEqual(output.trial_idx.min(), 1)

    def test_process_trials_starts_with_first_trial(self):
        "Makes sure dataframe can have events before first trial if firsttrail=False"
        output = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=False)
        self.assertEqual(output.trial_idx.min(), 0)

    def test_correct_columns(self):
        "Makes sure dataframe has the correct columns"
        output = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=False)
        self.assertTrue(all(output.columns == self.columns))  

    def test_trials_dataframe_has_name_trials(self):
        "Makes sure dataframe has the correct name"
        output = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=False)
        self.assertEqual(output.name, 'trials')

    def test_dataframes_attribute_added_to_segment(self):
        "Makes sure dataframes attribute is added to segment object"
        ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=False, returndf=False)
        self.assertTrue(hasattr(self.segment, 'dataframes')) 

    def test_dataframe_appended_to_segment(self):
        "Makes sure dataframe is appended to dataframe part of segment object"
        df = ProcessTrials(seg=self.segment, name=self.name, 
            startoftrial=self.startoftrial, epochs=self.epochs, 
            typedf=self.typeframe, firsttrial=False, returndf=True)
        pd.testing.assert_frame_equal(df, self.segment.dataframes['trials'])



class TestCalculateStartsAndDurations(unittest.TestCase):
    "Tests for CalculateStartsAndDurations"

    def setUp(self):
        self.trials = pd.DataFrame()
        self.trials['time'] = np.array([113.64892769, 118.64899683, 125.64938855, \
            125.79111004, 126.99205732, 131.99212646, 138.99219036, 139.09721184, \
            144.09728098, 145.57216859])
        self.trials['event'] = np.array(['iti_start', 'iti_end', 'omission', \
            'tray_activated', 'iti_start', 'stimulus_appears', 'tray_activated', \
            'iti_start', 'stimulus_appears', 'correct'])
        self.trials['trial_idx'] = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.trials['results'] = np.array(['omission', 'omission', 'omission', \
            'omission', 'NONE', 'NONE', 'NONE', 'correct', 'correct', 'correct'])
        self.trials['with_previous_results'] = np.array([np.nan, np.nan, np.nan, \
            np.nan, 'omission_NONE', 'omission_NONE', 'omission_NONE', \
            'NONE_correct', 'NONE_correct', 'NONE_correct'])
        self.trials['event_type'] = np.array(['start', 'end', 'results', \
            'other', 'start', 'stimulus', 'other', 'start', 'stimulus', 'results'])
        self.trials.name = 'trials'
        self.startoftrial = ['start']
        self.endoftrial = ['end']
        self.epoch_col = 'results'

    def tearDown(self):
        del self.trials 
        del self.startoftrial
        del self.endoftrial
        del self.epoch_col

    def test_trials_is_a_dataframe(self):
        "Make sure trials is a dataframe"
        self.assertRaises(TypeError, CalculateStartsAndDurations, \
            trials='not a dataframe')

    def test_startoftrial_is_a_list(self):
        "Make sure startoftrial is a list"
        self.assertRaises(TypeError, CalculateStartsAndDurations, trials=self.trials, 
            startoftrial='not a list')

    def test_endoftrial_is_a_list(self):
        "Make sure endoftrial is a list"
        self.assertRaises(TypeError, CalculateStartsAndDurations, 
            trials=self.trials, startoftrial=self.startoftrial, 
            endoftrial='not a list')

    def test_endeventmissing_is_correct(self):
        "Make sure endeventmissing is correct"
        self.assertRaises(ValueError, lambda: CalculateStartsAndDurations( 
            trials=self.trials, startoftrial=self.startoftrial, 
            endoftrial=self.endoftrial, endeventmissing='not a choice'))

    def test_starts_times_are_correct(self):
        "Makes sure start time for each epoch is correct"
        correct_time = self.trials.loc[0, 'time'] * pq.s
        output = CalculateStartsAndDurations(trials=self.trials, 
            epoch_column=self.epoch_col, startoftrial=self.startoftrial, 
            endoftrial=self.endoftrial)
        output_time = output[0][0]
        self.assertEqual(correct_time, output_time)

    def test_durations_are_correct_if_end_event_exists(self):
        "Makes sure duration is correct if there is an end event"
        correct_duration = self.trials.time.diff().values[1] * pq.s
        output = CalculateStartsAndDurations(trials=self.trials, 
            epoch_column=self.epoch_col, startoftrial=self.startoftrial, 
            endoftrial=self.endoftrial)
        output_duration = output[0][1]
        self.assertEqual(correct_duration, output_duration)

    def test_durations_are_correct_if_endeventmissing_is_last(self):
        "Makes sure duration is correct if endeventmissing = last"
        correct_duration = self.trials.loc[6, 'time'] - self.trials.loc[4, 'time']
        correct_duration = correct_duration * pq.s
        output = CalculateStartsAndDurations(trials=self.trials, 
            epoch_column=self.epoch_col, startoftrial=self.startoftrial, 
            endoftrial=self.endoftrial, endeventmissing='last')
        output_duration = output[1][1]
        self.assertEqual(correct_duration, output_duration)

    def test_durations_are_correct_if_endeventmissing_is_next(self):
        "Makes sure duration is correct if endeventmissing = next"
        correct_duration = self.trials.loc[7, 'time'] - self.trials.loc[4, 'time']
        correct_duration = correct_duration * pq.s
        output = CalculateStartsAndDurations(trials=self.trials, 
            epoch_column=self.epoch_col, startoftrial=self.startoftrial, 
            endoftrial=self.endoftrial, endeventmissing='next')
        output_duration = output[1][1]
        self.assertEqual(correct_duration, output_duration)



class TestGroupTrialsByEpoch(unittest.TestCase):
    "Tests for GroupTrialsByEpoch function"

    def setUp(self):
        self.trials = pd.DataFrame()
        self.trials['time'] = np.array([113.64892769, 118.64899683, 125.64938855, \
            125.79111004, 126.99205732, 131.99212646, 138.99219036, 139.09721184, \
            144.09728098, 145.57216859])
        self.trials['event'] = np.array(['iti_start', 'iti_end', 'omission', \
            'tray_activated', 'iti_start', 'stimulus_appears', 'tray_activated', \
            'iti_start', 'stimulus_appears', 'correct'])
        self.trials['trial_idx'] = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.trials['results'] = np.array(['omission', 'omission', 'omission', \
            'omission', 'NONE', 'NONE', 'NONE', 'correct', 'correct', 'correct'])
        self.trials['with_previous_results'] = np.array(['a', 'a', 'a', 'a', \
            'omission_NONE', 'omission_NONE', 'omission_NONE', \
            'NONE_correct', 'NONE_correct', 'NONE_correct'])
        self.trials['event_type'] = np.array(['start', 'end', 'results', \
            'other', 'start', 'stimulus', 'other', 'start', 'stimulus', 'results'])
        self.trials.loc[self.trials.with_previous_results == 'a', \
            'with_previous_results'] = np.nan
        self.trials.name = 'trials'
        self.startoftrial = ['start']
        self.endoftrial = ['end']
        self.segment = Segment()
        self.segment.dataframes = {}
        self.segment.dataframes.update({'trials': self.trials})

    def tearDown(self):
        del self.trials 
        del self.startoftrial
        del self.endoftrial
        del self.segment

    def test_seg_is_a_segment_object(self):
        "Make sure seg is a segment object"
        self.assertRaises(TypeError, GroupTrialsByEpoch, trials=self.trials, 
            startoftrial=self.startoftrial, endoftrial=self.endoftrial, 
            seg='not a segment object')

    def test_epochs_added_to_segment(self):
        "Makes sure epochs are added to segment object"
        GroupTrialsByEpoch(seg=self.segment, trials=self.trials, 
            startoftrial=self.startoftrial, endoftrial=self.endoftrial)
        to_print = ' '.join(x.name for x in self.segment.epochs)
        print((self.trials))
        self.assertEqual(len(self.segment.epochs), 
            self.trials.results.unique().shape[0] + \
            self.trials.with_previous_results.unique().shape[0] - 1)

    def test_epochs_added_to_segment_if_trials_is_not_passed_but_in_seg(self):
        "Makes sure trials is used in segment.dataframes if no trials variable passed"
        GroupTrialsByEpoch(seg=self.segment, trials=None, 
            startoftrial=self.startoftrial, endoftrial=self.endoftrial)
        self.assertEqual(len(self.segment.epochs), 
            self.trials.results.unique().shape[0] + \
            self.trials.with_previous_results.unique().shape[0] - 1)

class TestGetImagingDataTTL(unittest.TestCase):
    "Tests for GetImagingDataTTL function"

    @patch('imaging_analysis.utils.ReadNeoTdt')
    def test_GetImagingDataTTL(self, mock_tdt):
        event1 = neo.Event(times=[1,2,3]*pq.s, name='A')
        event2 = neo.Event(times=[4,5,6]*pq.s, name='B')
        event3 = neo.Event(times=[7,8,9]*pq.s, name='C')
        segment = neo.Segment()
        segment.events.append(event1)
        segment.events.append(event2)
        segment.events.append(event3)
        block = neo.Block()
        block.segments.append(segment)
        mock_tdt.return_value = block
        self.assertEqual(3, GetImagingDataTTL('/dpath/', time_idx=-1, event_name='A'))
        self.assertEqual(6, GetImagingDataTTL('/dpath/', time_idx=-1, event_idx=1))

if __name__ == '__main__':
    unittest.main()