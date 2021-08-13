#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "06 Mar 2018"
__lastmodified__ = "22 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal, Event, Segment, Epoch
import numpy as np
import quantities as pq
import copy as cp
import pandas as pd
from imaging_analysis.segment_processing import TruncateSegment
from imaging_analysis.segment_processing import TruncateSegments
from imaging_analysis.segment_processing import AppendDataframesToSegment
from imaging_analysis.segment_processing import AlignEventsAndSignals


class TestTruncateSegment(unittest.TestCase):
    "Tests for the TruncateSegment function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal_start = 10
        self.signal_end = 10
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt_start = 15
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 85
        self.evt_post_end = self.evt_end + 5
        self.not_segment = [100]
        self.segment = Segment()
        self.segment.analogsignals.append(self.signal)
        self.segment.events.append(self.evt)

    def tearDown(self):
        del self.signal
        del self.signal_start
        del self.signal_end
        del self.evt
        del self.evt_start
        del self.evt_pre_start
        del self.evt_end
        del self.evt_post_end
        del self.not_segment
        del self.segment

    def test_makes_sure_segment_is_passed(self):
        "Test to make sure segment object is passed"
        self.assertRaises(TypeError, TruncateSegment, self.not_segment)

    def test_signals_are_truncated(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end)
        analog = trunc_seg.analogsignals[0]
        true_start = self.signal_start * pq.s
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_start = true_start == analog.t_start
        test_stop = true_stop == analog.t_stop
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_the_same_if_clip_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end, clip_same=True)
        event_times = trunc_seg.events[0].times
        test_start = self.signal_start - 1 not in event_times
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_stop = true_stop + 1 * pq.s not in event_times
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_differently_if_clip_not_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_seg = TruncateSegment(self.segment, start=self.signal_start, 
            end=self.signal_end, clip_same=False, evt_start=self.evt_start, 
            evt_end=self.evt_end)
        event_times = trunc_seg.events[0].times
        test_start = self.evt_pre_start not in event_times
        test_stop = self.evt_post_end not in event_times
        self.assertTrue(all([test_start, test_stop]))



class TestTruncateSegments(unittest.TestCase):
    "Tests for the TruncateSegments function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal2 = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal_start = 10
        self.signal_end = 10
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(np.arange(0, 100 ,1)*pq.s, 
            labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 50))
        self.evt_start = 15
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 85
        self.evt_post_end = self.evt_end + 5
        self.not_segment = [100]
        self.segment = Segment()
        self.segment.analogsignals.append(self.signal)
        self.segment.events.append(self.evt)
        self.segment2 = Segment()
        self.segment2.analogsignals.append(self.signal2)
        self.segment2.events.append(self.evt2)       
        self.segments = [self.segment, self.segment2]

    def tearDown(self):
        del self.signal
        del self.signal_start
        del self.signal_end
        del self.evt
        del self.evt_start
        del self.evt_pre_start
        del self.evt_end
        del self.evt_post_end
        del self.not_segment
        del self.segment
        del self.segments

    def test_makes_sure_segment_is_passed(self):
        "Test to make sure list object is passed"
        self.assertRaises(TypeError, TruncateSegments, self.segment)

    def test_signals_are_truncated(self):
        "Test to make sure analog starts/ends are truncated for every segment"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end)
        choice_to_test = np.random.choice(list(range(len(trunc_segs))))
        analog = trunc_segs[choice_to_test].analogsignals[0]
        true_start = self.signal_start * pq.s
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_start = true_start == analog.t_start
        test_stop = true_stop == analog.t_stop
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_the_same_if_clip_same(self):
        "Test to make sure analog starts/ends are truncated for every segment"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end, clip_same=True)
        choice_to_test = np.random.choice(list(range(len(trunc_segs))))
        event_times = trunc_segs[choice_to_test].events[0].times
        test_start = self.signal_start - 1 not in event_times
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        test_stop = true_stop + 1 * pq.s not in event_times
        self.assertTrue(all([test_start, test_stop]))

    def test_events_are_truncated_differently_if_clip_not_same(self):
        "Test to make sure analog starts/ends are truncated"
        trunc_segs = TruncateSegments(self.segments, start=self.signal_start, 
            end=self.signal_end, clip_same=False, evt_start=self.evt_start, 
            evt_end=self.evt_end)
        choice_to_test = np.random.choice(list(range(len(trunc_segs))))
        event_times = trunc_segs[choice_to_test].events[0].times
        test_start = self.evt_pre_start not in event_times
        test_stop = self.evt_post_end not in event_times
        self.assertTrue(all([test_start, test_stop]))



class TestAppendDataframesToSegment(unittest.TestCase):
    "Tests for AppendDataframesToSegment"

    def setUp(self):
        self.segment = Segment()
        self.df = pd.DataFrame()
        self.df2 = pd.DataFrame()
        self.segment2 = Segment()
        self.segment2.dataframes = {}
        self.segment2.dataframes.update({'test': pd.DataFrame()})

    def tearDown(self):
        del self.segment 
        del self.df
        del self.segment2
        del self.df2

    def test_segment_check(self):
        "Makes sure segment object is passed"
        self.assertRaises(TypeError, AppendDataframesToSegment, 'not a segment')

    def test_dataframe_check(self):
        "Makes sure dataframe object is passed"
        self.assertRaises(TypeError, AppendDataframesToSegment, self.segment,
            'not a dataframe')

    def test_names_check(self):
        "Makes sure names is string or list"
        self.assertRaises(TypeError, AppendDataframesToSegment, self.segment,
            self.df, {'not': 'a list of string'})

    def test_segment_has_dataframe_attribute_if_none_before(self):
        "Makes sure segment has dataframes attribute if it didnt before"
        AppendDataframesToSegment(self.segment, self.df, 'test')
        self.assertTrue(hasattr(self.segment, 'dataframes'))

    def test_dataframe_object_is_added(self):
        "Makes sure dataframe object is added to dataframes"
        AppendDataframesToSegment(self.segment, self.df, 'test')
        self.assertIsInstance(self.segment.dataframes['test'], pd.core.frame.DataFrame)

    def test_segment_does_not_erase_if_dataframes_already_exists(self):
        "Makes sure it adds to dataframes if dataframes already exists"
        AppendDataframesToSegment(self.segment2, self.df, 'test2')
        self.assertEqual(len(list(self.segment2.dataframes.keys())), 2)

    def test_segment_works_with_list_of_dataframes(self):
        "Makes sure it works with a list of dataframes"
        AppendDataframesToSegment(self.segment, [self.df, self.df2], ['test', 'test2'])
        self.assertEqual(len(list(self.segment.dataframes.keys())), 2)



class TestAlignEventsAndSignals(unittest.TestCase):
    "Tests for AlighEventsAndSignals"

    def setUp(self):
        self.segment = Segment()
        self.epoch = Epoch(name='my epoch')
        self.segment.epochs.append(self.epoch)
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz, name='my signal')
        self.segment.analogsignals.append(self.signal)
        self.trials = pd.DataFrame()
        self.trials.name = 'trials'
        self.segment2 = Segment()
        self.segment2.epochs.append(self.epoch)
        self.segment2.analogsignals.append(self.signal)
        self.segment2.dataframes = [self.trials]

    def tearDown(self):
        del self.segment
        del self.epoch 
        del self.signal
        del self.trials 
        del self.segment2

    def test_segment_object_passed(self):
        "Makes sure segment object is passed"
        self.assertRaises(TypeError, AlignEventsAndSignals, seg='not a segment')

    def test_epoch_name_correct(self):
        "Makes sure epoch_name must be correct"
        self.assertRaises(ValueError, AlignEventsAndSignals, seg=self.segment, 
            epoch_name='not an epoch')

    def test_analog_ch_name_correct(self):
        "Makes sure analog_ch_name must be correct"
        self.assertRaises(ValueError, AlignEventsAndSignals, seg=self.segment, 
            epoch_name='my epoch', analog_ch_name='not a channel')

    def test_trials_dataframe_exists(self):
        "Makes sure trials dataframe exists"
        self.assertRaises(ValueError, AlignEventsAndSignals, seg=self.segment, 
            epoch_name='my epoch', analog_ch_name='my signal')

    def test_event_type_is_correct(self):
        "Makes sure event_type is label or type"
        self.assertRaises(ValueError, AlignEventsAndSignals, seg=self.segment2, 
            epoch_name='my epoch', analog_ch_name='my signal', event_type='wrong')


if __name__ == '__main__':
    unittest.main()