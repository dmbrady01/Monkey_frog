#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_signal_processing.py: Python script that contains tests for signal_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "09 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import AnalogSignal, Segment
import neo
import numpy as np
import quantities as pq
import scipy.signal as ssp
from imaging_analysis.signal_processing import TruncateSignal
from imaging_analysis.signal_processing import TruncateSignals
from imaging_analysis.signal_processing import ButterFilterDesign
from imaging_analysis.signal_processing import FilterSignal
from imaging_analysis.signal_processing import DeltaFOverF
from imaging_analysis.signal_processing import NormalizeSignal
from imaging_analysis.signal_processing import ProcessSignalData
from imaging_analysis.signal_processing import GetImagingDataIndices

class TestTruncateSignal(unittest.TestCase):
    "Tests for the TruncateSignal function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000, 1), units='V',
            sampling_rate=1*pq.Hz)
        self.not_analog = np.random.randn(1000, 1)
        self.signal_start = 10
        self.signal_end = 10

    def tearDown(self):
        del self.signal
        del self.not_analog
        del self.signal_start
        del self.signal_end

    def test_makes_sure_analog_signal_is_passed(self):
        "Test to make sure analog signal object is passed"
        self.assertRaises(TypeError, TruncateSignal, self.not_analog)

    def test_signal_start_works(self):
        "Test to make sure start is truncated"
        trunc_sig = TruncateSignal(self.signal, start=self.signal_start)
        self.assertEqual(trunc_sig.t_start, self.signal_start * pq.s)

    def test_signal_end_works(self):
        "Test to make sure start is truncated"
        trunc_sig = TruncateSignal(self.signal, end=self.signal_end)
        true_stop = self.signal.t_stop - self.signal_end * pq.s
        self.assertEqual(trunc_sig.t_stop, true_stop)



class TestTruncateSignals(unittest.TestCase):
    "Code tests for TruncateSignals function."

    def setUp(self):
        self.signal = AnalogSignal(np.random.randn(1000,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signal2 = AnalogSignal(np.random.randn(900,1), units='V', 
            sampling_rate=1*pq.Hz)
        self.signals = [self.signal, self.signal2]
        self.signal_start = 10
        self.signal_end = 10

    def tearDown(self):
        del self.signal
        del self.signal2
        del self.signals
        del self.signal_start
        del self.signal_end

    def test_makes_sure_list_is_passed(self):
        "Makes sure list is passed to function"
        self.assertRaises(TypeError, TruncateSignals, 100)

    def test_signal_start_works(self):
        "Test to make sure signal is truncated from start in signal_list"
        trunc_sig = TruncateSignals(self.signals, start=self.signal_start)
        t_starts = [x.t_start for x in trunc_sig]
        self.assertTrue(all(self.signal_start * pq.s == x for x in t_starts))

    def test_signal_end_works(self):
        "Test to make sure signal is truncated from end in signal_list"
        trunc_sig = TruncateSignals(self.signals, end=self.signal_end)
        t_stops = [x.t_stop for x in trunc_sig]
        true_stops = [x.t_stop - self.signal_end * pq.s for x in self.signals]
        paired = list(zip(t_stops, true_stops))
        self.assertTrue(all(x == y for x, y in paired))



class TestButterFilterDesign(unittest.TestCase):
    "Code tests for ButterFilterDesign."

    def setUp(self):
        self.order = 10
        self.fs = 30
        self.high = 10
        self.low = .5
        self.wrong_btype = 'hamburger'

    def tearDown(self):
        del self.order
        del self.fs
        del self.low
        del self.high
        del self.wrong_btype

    def test_make_sure_correct_btype_is_passed(self):
        "Makes sure btype must be correct"
        self.assertRaises(ValueError, ButterFilterDesign, btype=self.wrong_btype)

    def test_make_sure_highcut_is_below_nyquist(self):
        "Makes sure highcut is below nyquist frequency"
        self.assertRaises(ValueError, ButterFilterDesign, highcut=self.fs, fs=self.fs)

    def test_lowpass_filter_design_works(self):
        "Makes sure lowpass filter design works"
        output = ButterFilterDesign(highcut=self.high, fs=self.fs, 
            order=self.order, btype='lowpass')
        check = ssp.butter(self.order, self.high/(self.fs*.5), btype='lowpass')
        self.assertTrue(np.array_equal(output, check))

    def test_highpass_filter_design_works(self):
        "Makes sure highpass filter design works"
        output = ButterFilterDesign(lowcut=self.low, fs=self.fs, 
            order=self.order, btype='highpass')
        check = ssp.butter(self.order, self.low/(self.fs*.5), btype='highpass')
        self.assertTrue(np.array_equal(output, check))

    def test_bandpass_filter_design_works(self):
        "Makes sure bandpass filter design works"
        output = ButterFilterDesign(lowcut=self.low, highcut=self.high, fs=self.fs, 
            order=self.order, btype='bandpass')
        check = ssp.butter(self.order, 
            [self.low/(self.fs*.5), self.high/(self.fs*.5)], btype='bandpass')
        self.assertTrue(np.array_equal(output, check))

    def test_bandstop_filter_design_works(self):
        "Makes sure bandstop filter design works"
        output = ButterFilterDesign(lowcut=self.low, highcut=self.high, fs=self.fs, 
            order=self.order, btype='bandstop')
        check = ssp.butter(self.order, 
            [self.low/(self.fs*.5), self.high/(self.fs*.5)], btype='bandstop')
        self.assertTrue(np.array_equal(output, check))



class TestFilterSignal(unittest.TestCase):
    "Code tests for FilterSignal function."

    def setUp(self):
        self.T = 5
        self.fs = 30
        self.nsamples = int(self.T * self.fs)
        self.t = np.linspace(0, self.T, self.nsamples, endpoint=False)
        self.signal = np.sin(1.2*2*np.pi*self.t) + 1.5*np.cos(9*2*np.pi*self.t) + \
                            0.5*np.sin(12.0*2*np.pi*self.t)
        self.highcut = 3.667
        self.lowcut = 0.5
        self.savgol_order = 1
        self.order = 5
        self.window_length = 11

    def tearDown(self):
        del self.T 
        del self.fs
        del self.nsamples
        del self.t
        del self.signal 
        del self.highcut
        del self.lowcut
        del self.savgol_order
        del self.order
        del self.window_length

    def test_lowpass_works(self):
        "Tests lowpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='lowpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='lowpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)

    def test_highpass_works(self):
        "Tests highpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='highpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='highpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)    

    def test_bandpass_works(self):
        "Tests bandpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='bandpass')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='bandpass')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal) 

    def test_bandstop_works(self):
        "Tests bandpass functionality"
        signal = FilterSignal(self.signal, lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='bandstop')
        b, a = ButterFilterDesign(lowcut=self.lowcut, highcut=self.highcut, 
            fs=self.fs, order=self.order, btype='bandstop')
        test_signal = ssp.filtfilt(b, a, self.signal)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal) 

    def test_savgol_works(self):
        "Tests savgol functionality"
        signal = FilterSignal(self.signal, savgol_order=self.savgol_order, 
            window_length=self.window_length, btype='savgol')
        test_signal = ssp.savgol_filter(self.signal, self.window_length, 
            self.savgol_order, axis=0)
        equal = np.array_equal(signal, test_signal)
        self.assertTrue(equal)



class TestDeltaFOverF(unittest.TestCase):
    "Tests for DeltaFOverF function"

    def setUp(self):
        self.signal = np.random.randn(1000, 1)
        self.signal2 = np.random.randn(1000, 1)
        self.period = [0, 100]
        self.wrong_mode = 'mymode'
        self.offset = 10

    def tearDown(self):
        del self.signal
        del self.signal2
        del self.period
        del self.wrong_mode

    def test_mode_must_be_correct(self):
        "Makes sure error is thrown if wrong mode selected."
        self.assertRaises(ValueError, DeltaFOverF, self.signal, mode=self.wrong_mode)

    def test_median_mode_works(self):
        "Makes sure median mode works"
        test_signal = (self.signal - np.median(self.signal))/np.median(self.signal)
        test_signal = test_signal * 100.0
        equal = np.array_equal(test_signal, DeltaFOverF(self.signal, mode='median'))
        self.assertTrue(equal)

    def test_mean_mode_works(self):
        "Makes sure mean mode works"
        test_signal = (self.signal - np.mean(self.signal))/np.mean(self.signal)
        test_signal = test_signal * 100.0
        equal = np.array_equal(test_signal, DeltaFOverF(self.signal, mode='mean'))
        self.assertTrue(equal)

    def test_reference_mode_works(self):
        "Makes sure reference mode works"
        test_signal = (self.signal - self.signal2)/self.signal2
        test_signal = test_signal * 100.0
        signal = DeltaFOverF(self.signal, reference=self.signal2, mode='reference')
        equal = np.array_equal(test_signal, signal)
        self.assertTrue(equal)

    def test_period_mean_works(self):
        "Makes sure period_mean mode works"
        reference = np.mean(self.signal[self.period[0]:self.period[1]])
        test_signal = (self.signal - reference)/reference
        test_signal = test_signal * 100.0
        equal = np.array_equal(test_signal, DeltaFOverF(self.signal, 
            period=self.period, mode='period_mean'))
        self.assertTrue(equal)

    def test_period_median_works(self):
        "Makes sure period_median mode works"
        reference = np.median(self.signal[self.period[0]:self.period[1]])
        test_signal = (self.signal - reference)/reference
        test_signal = test_signal * 100.0
        equal = np.array_equal(test_signal, DeltaFOverF(self.signal, 
            period=self.period, mode='period_median'))
        self.assertTrue(equal)

    def test_offset_works(self):
        "Makes sure offset works"
        reference = DeltaFOverF(self.signal, period=self.period, mode='mean')
        test = DeltaFOverF(self.signal, period=self.period, mode='mean', offset=10000)
        equal = np.array_equal(test, reference)
        self.assertFalse(equal)


class TestNormalizeSignal(unittest.TestCase):
    "Tests for NormalizeSignal function"

    def setUp(self):
        self.fs = 1000.0
        self.N = self.fs*10
        self.t = np.linspace(0,self.N/self.fs, self.N)
        self.f1 = 25
        self.f2 = 1
        self.f3 = 65
        self.signal = 2*np.cos(2*np.pi*self.f1*self.t) + \
            .25*np.cos(2*np.pi*self.f2*self.t) + \
            .1*np.cos(2*np.pi*self.f3*self.t) + \
            .05*np.random.randn(self.t.shape[0])
        self.reference = 1*np.cos(2*np.pi*self.f1*self.t) + \
            .1*np.cos(2*np.pi*self.f2*self.t) + \
            .05*np.cos(2*np.pi*self.f3*self.t) + \
            .05*np.random.randn(self.t.shape[0])

    def tearDown(self):
        del self.fs
        del self.N
        del self.t 
        del self.f1
        del self.f2 
        del self.f3 
        del self.signal
        del self.reference

    def test_no_reference_nor_detrend(self):
        "Tests with no reference and no detrend."
        output = NormalizeSignal(self.signal, fs=self.fs, detrend=False, mode='median')
        test_output = FilterSignal(self.signal, highcut=40.0, fs=self.fs)
        test_output = DeltaFOverF(test_output)
        equal = np.array_equal(test_output, output)
        self.assertTrue(equal)

    def test_no_reference_with_detrend(self):
        "Tests with no reference but detrend"
        output = NormalizeSignal(self.signal, fs=self.fs, detrend=True)
        test_output = FilterSignal(self.signal, highcut=40.0, fs=self.fs)
        test_output = DeltaFOverF(test_output)
        test_output = test_output - FilterSignal(test_output, btype='savgol')
        equal = np.array_equal(test_output, output)
        self.assertTrue(equal)

    def test_with_reference_no_detrend(self):
        "Tests with reference and no detrend."
        output = NormalizeSignal(self.signal, self.reference, 
            fs=self.fs, detrend=False)
        test_sig = FilterSignal(self.signal, highcut=40.0, fs=self.fs)
        test_ref = FilterSignal(self.reference, highcut=40.0, fs=self.fs)
        test_sig = DeltaFOverF(test_sig)
        test_ref = DeltaFOverF(test_ref)
        test_output = test_sig - test_ref
        equal = np.array_equal(test_output, output)
        self.assertTrue(equal)

    def test_with_reference_with_detrend(self):
        "Tests with reference and detrend."
        output = NormalizeSignal(self.signal, self.reference, 
            fs=self.fs, detrend=True)
        test_sig = FilterSignal(self.signal, highcut=40.0, fs=self.fs)
        test_ref = FilterSignal(self.reference, highcut=40.0, fs=self.fs)
        test_sig = DeltaFOverF(test_sig)
        test_ref = DeltaFOverF(test_ref)
        test_sig = test_sig - FilterSignal(test_sig, btype='savgol')
        test_ref = test_ref - FilterSignal(test_ref, btype='savgol')
        test_output = test_sig - test_ref
        equal = np.array_equal(test_output, output)
        self.assertTrue(equal)

    def test_return_all_signals_no_reference(self):
        "Tests with return all signals and no reference."
        output = NormalizeSignal(self.signal, fs=self.fs, return_all_signals=True)
        self.assertEqual(len(output), 3)

    def test_return_all_signals_with_reference(self):
        "Tests with return all signals with reference."
        output = NormalizeSignal(self.signal, self.reference, fs=self.fs,
            return_all_signals=True)
        self.assertEqual(len(output), 5)



class TestProcessSignalData(unittest.TestCase):
    "Tests for ProcessSignalData"
    
    def setUp(self):
        self.fs = 1000.0
        self.N = self.fs*10
        self.t = np.linspace(0,self.N/self.fs, self.N)
        self.f1 = 25
        self.f2 = 1
        self.f3 = 65
        self.signal = 2*np.cos(2*np.pi*self.f1*self.t) + \
            .25*np.cos(2*np.pi*self.f2*self.t) + \
            .1*np.cos(2*np.pi*self.f3*self.t) + \
            .05*np.random.randn(self.t.shape[0])
        self.analog1 = AnalogSignal(self.signal, units='V', 
            sampling_rate=self.fs * pq.s, name='Signal')
        self.reference = 1*np.cos(2*np.pi*self.f1*self.t) + \
            .1*np.cos(2*np.pi*self.f2*self.t) + \
            .05*np.cos(2*np.pi*self.f3*self.t) + \
            .05*np.random.randn(self.t.shape[0])
        self.analog2 = AnalogSignal(self.reference, units='V', 
            sampling_rate=self.fs * pq.s, name='Reference')
        self.segment = Segment()
        self.segment.analogsignals.append(self.analog1)
        self.segment.analogsignals.append(self.analog2)

    def tearDown(self):
        del self.fs
        del self.N
        del self.t 
        del self.f1
        del self.f2 
        del self.f3 
        del self.signal
        del self.reference
        del self.analog1 
        del self.analog2
        del self.segment

    def test_must_pass_segment(self):
        "Makes sure segment object is passed"
        self.assertRaises(TypeError, ProcessSignalData, [100])

    def test_check_sig_ch_exists(self):
        "Makes sure error is raised if sig_ch is not the name of an AnalogSignal"
        self.assertRaises(ValueError, ProcessSignalData, seg=self.segment,
            sig_ch='Not here', ref_ch='Reference')

    def test_check_ref_ch_exists(self):
        "Makes sure error is raised if ref_ch is not the name of an AnalogSignal"
        self.assertRaises(ValueError, ProcessSignalData, seg=self.segment,
            sig_ch='Signal', ref_ch='Not here')

    def test_analogsignal_is_added_to_segment(self):
        "Makes sure AnalogSignal is added to segment"
        original_length = len(self.segment.analogsignals)
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = len(self.segment.analogsignals)
        self.assertEqual(original_length + 1, output)

    def test_analogsignal_is_added_to_segment_is_correct_object(self):
        "Makes sure AnalogSignal is added to segment and is correct object"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1]
        self.assertIsInstance(output, neo.core.AnalogSignal)

    def test_added_analogsignal_has_correct_name(self):
        "Makes sure added AnalogSignal has correct name"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1]
        self.assertEqual('Test', output.name)

    def test_added_analogsignal_has_correct_units(self):
        "Makes sure added AnalogSignal has correct units (%)"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1]
        self.assertEqual(output.units, pq.percent)

    def test_added_analogsignal_has_correct_t_start(self):
        "Makes sure added AnalogSignal has correct t_start"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1]
        output_tstart = output.t_start 
        correct_tstart = self.segment.analogsignals[0].t_start
        self.assertEqual(output_tstart, correct_tstart)

    def test_added_analogsignal_has_correct_sampling_rate(self):
        "Makes sure added AnalogSignal has correct sampling_rate"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1]
        output_rate = output.sampling_rate
        correct_rate = self.segment.analogsignals[0].sampling_rate
        self.assertEqual(output_rate, correct_rate)

    def test_correct_functionality(self):
        "Tests that NormalizeSignal is run correctly"
        ProcessSignalData(seg=self.segment, sig_ch='Signal', ref_ch='Reference', 
            name='Test', fs=self.fs)
        output = self.segment.analogsignals[-1].magnitude
        test_signal = NormalizeSignal(self.analog1, self.analog2, fs=self.fs)
        equal = np.array_equal(output, test_signal)
        self.assertTrue(equal)

class TestGetImagingDataIndices(unittest.TestCase):

    def test_get_imaging_data_indices(self):
        self.assertEqual(GetImagingDataIndices(10.5, 50.2, 12), (11*12, 50*12))



if __name__ == '__main__':
    unittest.main()