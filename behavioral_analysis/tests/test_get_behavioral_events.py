import unittest
import pandas as pd
import os
from mock import patch
import neo
import quantities as pq
import numpy as np

from behavioral_analysis.get_behavioral_events import GetBehavioralEvents, BOUT_TYPE_DICT, STIMULUS_NAME_SET

class TestBoutTypeDict(unittest.TestCase):

    def test_class_of_bout_type_dict(self):
        # Make sure BOUT_TYPE_DICT is a list of dictionaries
        self.assertIsInstance(BOUT_TYPE_DICT, list)
        self.assertTrue(all([isinstance(x, dict) for x in BOUT_TYPE_DICT]))

    def test_keys_of_each_dict(self):
        self.assertTrue(all(['location' in list(x.keys()) for x in BOUT_TYPE_DICT]))
        self.assertTrue(all(['zone' in list(x.keys()) for x in BOUT_TYPE_DICT]))
        self.assertTrue(all(['name' in list(x.keys()) for x in BOUT_TYPE_DICT]))

class TestStimulusNameSet(unittest.TestCase):

    def test_stimulus_name_set_is_a_set(self):
        self.assertIsInstance(STIMULUS_NAME_SET, set)

class TestGetBehavioralEvents(unittest.TestCase):
    "Tests for the GetBehavioralEvents object."

    def test_initialization_attributes(self):
        e = GetBehavioralEvents(
                                datapath='a', 
                                savefolder='b', 
                                time_offset='c', 
                                time_column='d', 
                                minimum_bout_time='e', 
                                datatype='f', 
                                name_match='g', 
                                max_session_time='h', 
                                label_dict='i', 
                                offset_datapath='j', 
                                fp_datapath='k', 
                                stimulus_name_set='l', 
                                latency_threshold='m',
                                cast='n'
                                )
        self.assertEqual(e.datapath, 'a')
        self.assertEqual(e.savefolder, 'b/')
        self.assertEqual(e.time_offset, 'c')
        self.assertEqual(e.time_column, 'd')
        self.assertEqual(e.minimum_bout_time, 'e')
        self.assertEqual(e.datatype, 'f')
        self.assertEqual(e.name_match, 'g')
        self.assertEqual(e.max_session_time, 'h')
        self.assertEqual(e.label_dict, 'i')
        self.assertEqual(e.offset_datapath, 'j')
        self.assertEqual(e.fp_datapath, 'k')
        self.assertEqual(e.stimulus_name_set, 'l')
        self.assertEqual(e.latency_threshold, 'm')
        self.assertFalse(e.cast)

    def test_set_datapath(self):
        default_datapath = './data/ethovision.csv'
        self.assertEqual(GetBehavioralEvents().datapath, default_datapath)

    def test_set_save_folder_default(self):
        e = GetBehavioralEvents(datapath=None)
        e.set_savefolder()
        default_datapath = './data/ethovision.csv'
        default_save_folder = './data/'
        self.assertEqual(e.datapath, default_datapath)
        self.assertEqual(e.savefolder, default_save_folder)

    def test_set_save_folder_add_os_sep(self):
        e = GetBehavioralEvents(savefolder='/abc')
        e.set_savefolder()
        default_save_folder = '/abc/'
        self.assertEqual(e.savefolder, default_save_folder)

    @patch('pandas.DataFrame.to_csv')
    def test_save_files(self, mock_to_csv):
        e = GetBehavioralEvents(
                                savefolder='/path',
                                datatype='abc'
                                )
        data = [
                ('123', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})), 
                ('456', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
                ]
        e.save_files(data)
        check_file = '/path' + os.sep + 'abc_123.csv'
        check_file2 = '/path' + os.sep + 'abc_456.csv'
        mock_to_csv.assert_any_call(check_file, index=False)
        mock_to_csv.assert_any_call(check_file2, index=False)

    def test_cast_to_bool(self):
        e = GetBehavioralEvents(cast='y')
        self.assertTrue(e.cast)
        e.cast = 'n'
        e.cast_to_bool()
        self.assertFalse(e.cast)

    def test_prune_minimum_bouts(self):
        data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        reduced_data = {
                    'Bout start': [2, 3],
                    'Bout end': [4, 10]
        }
        reduced_df = pd.DataFrame(reduced_data, index=[1,2])
        pd.testing.assert_frame_equal(GetBehavioralEvents(minimum_bout_time=1).prune_minimum_bouts(df), reduced_df)

    def test_add_time_offset(self):
        data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        pd.testing.assert_frame_equal(df + 1, GetBehavioralEvents(time_offset=1).add_time_offset(df))

    def test_sort_by_bout_start(self):
        data = {
                    'Bout start': [2, 1, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        ordered_data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [4, 1, 10]
        }
        ordered_df = pd.DataFrame(ordered_data, index=[0, 1,2])
        pd.testing.assert_frame_equal(GetBehavioralEvents()
.sort_by_bout_start(df), ordered_df)

    def test_relabel_bout_type(self):
        label_dict = [
            {
                'location': 'right',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'object'
            },
            {
                'location': 'left',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'social'
            }
        ]
        e = GetBehavioralEvents(label_dict=label_dict)
        self.assertEqual(e.relabel_bout_type('left sniffing', 'left'), 'social')
        self.assertEqual(e.relabel_bout_type('left sniffing', 'right'), 'object')
        self.assertEqual(e.relabel_bout_type('not in label dict', 'right'), 'not in label dict')
        self.assertEqual(e.relabel_bout_type('left sniffing', 'not in label dict'), 'left sniffing')

    def test_relabel_bout_type_for_df(self):
        label_dict = [
            {
                'location': 'right',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'object'
            },
            {
                'location': 'left',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'social'
            }
        ]
        data = {
            'Bout type': ['left interaction', 'right interaction', 'nothing'],
            'Stimulus Location': ['right', 'right', 'right']
        }
        df = pd.DataFrame(data)
        relabeled_data = {
            'Bout type': ['object', 'social', 'nothing'],
            'Stimulus Location': ['right', 'right', 'right']
        }
        relabeled_df = pd.DataFrame(df)
        pd.testing.assert_frame_equal(relabeled_df, GetBehavioralEvents(label_dict=label_dict).relabel_bout_type_for_df(df))

    def test_clean_and_strip_string(self):
        self.assertEqual(GetBehavioralEvents().clean_and_strip_string(' Hello   world again!', sep=','), 
            'Hello,world,again!')

    @patch('pandas.read_csv')
    def test_get_ethovision_header_info(self, mock_read_csv):
        data = [
            ['Header', 5],
            ['Animal Animal', 'abc'],
            ['Stimulus Location', 'right']
        ]
        mock_read_csv.return_value = pd.DataFrame(data)
        print((pd.DataFrame(data)))
        l, a, s = GetBehavioralEvents().get_ethovision_header_info('/path/', 
            stimulus_name_set={'stimulus location'}, animal_name_set={'animal animal'})
        self.assertEqual(l, 5)
        self.assertEqual(a, 'abc')
        self.assertEqual(s, 'right')
        # test only_skip_lines
        l, a, s = GetBehavioralEvents().get_ethovision_header_info('/path/', only_skip_lines=True)
        self.assertEqual(l, 5)  
        self.assertIsNone(a)
        self.assertIsNone(s)      

    @patch('behavioral_analysis.get_behavioral_events.GetBehavioralEvents.get_ethovision_header_info')
    @patch('pandas.read_csv')
    def test_load_ethovision_data(self, mock_read_csv, mock_header_data):
        mock_header_data.return_value = (3, 'abc', 'right')
        test_df = pd.DataFrame({'time': [1,2], 'value': [3,4]})
        mock_read_csv.return_value = test_df
        data, animal, location = GetBehavioralEvents().load_ethovision_data('/a/', {'stimulus location'})
        mock_read_csv.assert_called_with('/a/', skiprows=[0, 2]) # skips lines check
        pd.testing.assert_frame_equal(data, test_df) # returns df check
        self.assertEqual(animal, 'abc') # returns animal name
        self.assertEqual(location, 'right') # returns location

    @patch('behavioral_analysis.get_behavioral_events.GetBehavioralEvents.load_ethovision_data')
    def test_get_ethovision_start_ttl(self, mock_data):
        mock_data.return_value = (pd.DataFrame({'Time': [1, 2, 3], 'Value': [4, 5, 6]}), 'animal name', 'animal location')
        self.assertEqual(GetBehavioralEvents().get_ethovision_start_ttl(time_column='Time'), 2)

    @patch('imaging_analysis.utils.ReadNeoTdt')
    def test_get_fp_start_ttl(self, mock_tdt):
        event = neo.Event(times=[1,2,3]*pq.s, name='A')
        segment = neo.Segment()
        segment.events.append(event)
        segment.events.append(event)
        block = neo.Block()
        block.segments.append(segment)
        mock_tdt.return_value = block
        self.assertEqual(1, GetBehavioralEvents().get_fp_start_ttl('/path/to/data/', event_name='A'))
        self.assertEqual(1, GetBehavioralEvents().get_fp_start_ttl('/path/to/data/', event_idx=0))

    @patch('behavioral_analysis.get_behavioral_events.GetBehavioralEvents.get_ethovision_start_ttl')
    @patch('behavioral_analysis.get_behavioral_events.GetBehavioralEvents.get_fp_start_ttl')
    def test_get_ethovision_offset(self, mock_fp, mock_etho):
        mock_etho.return_value = 4
        mock_fp.return_value = 6
        self.assertEqual(GetBehavioralEvents().get_ethovision_offset(), 2)

    def test_calculate_bout_duration(self):
        data = {
            'start': [1,2,3],
            'end': [4,5,6]
        }
        df = pd.DataFrame(data)
        new_df = df.copy()
        new_df['Bout duration'] = pd.Series([3, 3, 3])
        pd.testing.assert_frame_equal(GetBehavioralEvents().calculate_bout_duration(df, 'start', 'end'), new_df)

    def test_calculate_bout_duration(self):
        data = {
            'start': [1.,3.,5.],
            'end': [2.,4.,6.]
        }
        df = pd.DataFrame(data)
        new_df = df.copy()
        new_df['NEW'] = pd.Series([np.nan, 1., 1.])
        pd.testing.assert_frame_equal(GetBehavioralEvents().calculate_interbout_latency(df, 'start', 'end', 'NEW'), new_df)

    def test_calculate_bout_durations_and_latencies(self):
        data = {
            'Bout start': [1.,3.,5.],
            'Bout end': [2.,4.,6.]
        }
        df = pd.DataFrame(data)
        new_df = df.copy()
        new_df['Bout duration'] = pd.Series([1., 1., 1])
        new_df['Latency from previous bout end'] = pd.Series([np.nan, 1., 1.])
        new_df['Latency from previous bout start'] = pd.Series([np.nan, 2., 2.])
        new_df['Latency to next bout start'] = pd.Series([2, 2, np.nan])
        pd.testing.assert_frame_equal(GetBehavioralEvents().calculate_bout_durations_and_latencies(df), new_df)

    def test_anneal_bouts(self):
        data = {
            'Bout type': ['a', 'a', 'b', 'a', 'a'],
            'Latency': [100, 1, 10, 3, 6],
            'Bout end': [5, 7, 3, 2, 1]
        }
        df = pd.DataFrame(data)
        annealed_data = {
            'Bout type': ['a', 'b', 'a', 'a'],
            'Latency': [100, 10, 3, 6],
            'Bout end': [7, 3, 2, 1]        
        }
        annealed_df = pd.DataFrame(annealed_data, index=[1,2,3,4])
        pd.testing.assert_frame_equal(GetBehavioralEvents().anneal_bouts(df, latency_threshold=5, latency_col='Latency'), annealed_df)

    @patch.object(GetBehavioralEvents, 'add_time_offset', return_value='b')
    @patch.object(GetBehavioralEvents, 'sort_by_bout_start', return_value='c')
    @patch.object(GetBehavioralEvents, 'relabel_bout_type_for_df', return_value='d')
    @patch.object(GetBehavioralEvents, 'calculate_bout_durations_and_latencies', return_value='e')
    @patch.object(GetBehavioralEvents, 'anneal_bouts', return_value='f')
    @patch.object(GetBehavioralEvents, 'prune_minimum_bouts', return_value='abc')
    def test_process_dataset(self, mock_prune, mock_anneal, mock_dur, mock_relabel, 
            mock_sort, mock_offset):
        dataset = [('123', 'a')]
        GetBehavioralEvents(latency_threshold=10).process_dataset(dataset)
        mock_offset.assert_called_with('a')
        mock_sort.assert_called_with('b')
        mock_relabel.assert_called_with('c')
        mock_dur.assert_any_call('d')
        self.assertEqual(mock_dur.call_count, 3)
        mock_anneal.assert_any_call('e', latency_threshold=10)
        self.assertEqual(mock_anneal.call_count, 2)
        mock_prune.assert_any_call('f')

    def test_convert_ethovision_nulls(self):
        df = pd.DataFrame({'a': [1., '-']})
        results = GetBehavioralEvents().convert_ethovision_nulls(df, 'a')
        self.assertTrue(np.isnan(results.loc[1, 'a']))

    def test_forward_fill_ethovision_data(self):
        df = pd.DataFrame({'a': [1., np.nan]})
        results = GetBehavioralEvents().forward_fill_ethovision_data(df, 'a')
        self.assertEqual(1, results.loc[1, 'a'])

    def test_convert_ethovision_to_int(self):
        df = pd.DataFrame({'a': [1.0, 2.0]})
        results = GetBehavioralEvents().convert_ethovision_to_int(df, 'a')
        self.assertIsInstance(results.loc[1, 'a'], int)

    @patch.object(GetBehavioralEvents, 'convert_ethovision_nulls', return_value='a')
    @patch.object(GetBehavioralEvents, 'forward_fill_ethovision_data', return_value='b')
    @patch.object(GetBehavioralEvents, 'convert_ethovision_to_int', return_value='c')
    def test_convert_ethovision_data_to_proper_format(self, mock_int, mock_ff, mock_null):
        df = pd.DataFrame({'a': [1.0, 2.0]})
        results = GetBehavioralEvents().convert_ethovision_data_to_proper_format(df, 'A')
        mock_null.assert_called_with(df, 'A')
        mock_ff.assert_called_with('a', 'A')
        mock_int.assert_called_with('b', 'A')
        self.assertEqual(results, 'c')

    def test_boolean_cast_data(self):
        df = pd.DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0]})
        check_df = pd.DataFrame({'a': [1, 1, 0], 'b': [0, 1, 0]})
        results = GetBehavioralEvents().boolean_cast_data(df, 'a', 'b')
        pd.testing.assert_frame_equal(results, check_df)

