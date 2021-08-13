import unittest
from mock import patch
import os
import numpy as np
import pandas as pd
from behavioral_analysis.get_behavioral_sequences import GetBehavioralSequences, SEQUENCE_DICT

class TestSimpleSequenceDict(unittest.TestCase):

    def test_class_of_bout_type_dict(self):
        # Make sure BOUT_TYPE_DICT is a list of dictionaries
        self.assertIsInstance(SEQUENCE_DICT, list)
        self.assertTrue(all([isinstance(x, dict) for x in SEQUENCE_DICT]))

    def test_keys_of_each_dict(self):
        self.assertTrue(all(['sequence' in list(x.keys()) for x in SEQUENCE_DICT]))
        self.assertTrue(all(['name' in list(x.keys()) for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout duration' in list(x.keys()) for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Latency to next bout start' in list(x.keys()) for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout start' in list(x.keys()) for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout end' in list(x.keys()) for x in SEQUENCE_DICT]))

class TestGetBehavioralSequences(unittest.TestCase):

    def test_initialize_properly(self):
        e = GetBehavioralSequences(
                                datapath='a', 
                                savepath='b',
                                sequences={'a'} 
                                )
        self.assertEqual(e.datapath, 'a')
        self.assertEqual(e.savepath, 'b')
        self.assertEqual(e.sequences, {'a'})

    def test_set_datapath(self):
        default_datapath = './data/ethovision.csv'
        self.assertEqual(GetBehavioralSequences().datapath, default_datapath)

    def test_get_animal_name(self):
        datapath = './etho_123.csv'
        self.assertEqual(GetBehavioralSequences().get_animal_name(datapath), '123')

    def test_set_savepath_default(self):
        e = GetBehavioralSequences(datapath=None)
        e.set_savepath()
        default_datapath = './data/ethovision.csv'
        default_savepath = './data/ethovision_behavioral_sequences.csv'
        self.assertEqual(e.datapath, default_datapath)
        self.assertEqual(e.savepath, default_savepath)

    @patch('pandas.DataFrame.to_csv')
    def test_save_files(self, mock_to_csv):
        e = GetBehavioralSequences(datapath='/path/etho.csv')
        data = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
        e.save_files(data)
        check_file = '/path/etho_behavioral_sequences.csv'
        mock_to_csv.assert_any_call(check_file, index=False)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = 123
        df = GetBehavioralSequences().load_data('/path/to.csv')
        self.assertEqual(df, 123)
        mock_read_csv.assert_called_with('/path/to.csv')

    def test_sort_by_bout_start(self):
        data = {
                    'Bout start': [2, 1, 1, 2],
                    'Bout type': ['a', 'b', 'b', 'c'],
                    'Bout end': [1, 4, 4, 10],
                    'Seq': [0, 1, 2, 3]
        }
        df = pd.DataFrame(data, index=[0, 1, 2, 3])
        ordered_data = {
                    'Bout start': [1, 1, 2, 2],
                    'Bout type': ['b', 'b', 'a','c'],
                    'Bout end': [4, 4, 1, 10],
                    'Seq': [1, 2, 0, 3]
        }
        ordered_df = pd.DataFrame(ordered_data, index=[0, 1, 2, 3])
        pd.testing.assert_frame_equal(GetBehavioralSequences().sort_by_bout_start(df), ordered_df)

    def test_chain_columns_to_list(self):
        data = {
            'Bout type': ['1', '2', '3', '4']
        }
        df = pd.DataFrame(data)
        check = [
            ['1', '2', '3'],
            ['2', '3', '4'],
            ['3', '4', np.nan],
            ['4', np.nan, np.nan]
        ]
        results = GetBehavioralSequences().chain_columns_to_list(df, 'Bout type', 3)
        # Need to test frames because nans don't match for lists
        pd.testing.assert_frame_equal(pd.DataFrame(check), pd.DataFrame(results))

    def test_map_and_eval(self):
        eval_list = ['>=3', '<5', '>100', '==10']
        value_list = ['4', '1', '10', '10']
        value2_list = ['4', '4', '101', '10']
        value3_list = ['4', '4', np.nan, '4']
        self.assertEqual(GetBehavioralSequences().map_and_eval(value_list, eval_list),
            False)
        self.assertEqual(GetBehavioralSequences().map_and_eval(value2_list, eval_list),
            True)
        self.assertEqual(GetBehavioralSequences().map_and_eval(value3_list, eval_list),
            False)

    def test_equivalent(self):
        self.assertTrue(GetBehavioralSequences().equivalent(['a', 'b'], ['a', 'b']))
        self.assertFalse(GetBehavioralSequences().equivalent(['a', 'b'], ['a', 'c']))

    def test_match_sequence_for_bout_types(self):
        data = {
            'Bout type': ['a', 'b', 'c', 'd', 'b', 'c']
        }
        df = pd.DataFrame(data)
        check = pd.Series([False, True, False, False, False, False])
        results = GetBehavioralSequences().match_sequence(df, 'Bout type', 
            ['b', 'c', 'd'], fnc='equivalent')
        pd.testing.assert_series_equal(results, check)

    def test_match_sequence_for_durations(self):
        data = {
            'Bout duration': [4, 5, 6, 2, 1, 3]
        }
        df = pd.DataFrame(data)
        check = pd.Series([False, True, False, False, False, False])
        results = GetBehavioralSequences().match_sequence(df, 'Bout duration', 
            ['==5', '==6', '==2'], 'map_and_eval')
        pd.testing.assert_series_equal(results, check)

    def get_last_from_bout_type_run(self):
        data = {
            'Bout type': ['a', 'b', 'b', 'c', 'b', 'b', 'b', 'c', 'c', 'c'],
            'Bout start': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Bout end': [21, 24, 25, 26, 28, 29, 30, 32, 33, 40]
        }
        df = pd.DataFrame(data)
        check_data = {
            'Bout type': ['a', 'b', 'b', 'c', 'b', 'b', 'b', 'c', 'c', 'c'],
            'Bout start': [1, 3, 3, 4, 7, 7, 7, 10, 10, 10],
            'Bout end': [21, 25, 25, 26, 30, 30, 30, 40, 40, 40]        
        }
        check_df = pd.DataFrame(check_data)
        results = GetBehavioralSequences().get_last_from_bout_type_run(df)
        pd.testing.assert_frame_equal(results, check_df)

    def test_get_bout_time(self):
        data = {
            'Bout type': ['a', 'b', 'b', 'c', 'b', 'b', 'b', 'c', 'c', 'c'],
            'Bout start': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Bout end': [21, 24, 25, 26, 28, 29, 30, 32, 33, 40]
        }
        sequence = ['c', 'b']
        df = pd.DataFrame(data)
        e = GetBehavioralSequences()
        # simple bout version
        results = e.get_bout_time(df, 'Bout start', 2, sequence)
        check = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan], name='Bout start')
        pd.testing.assert_series_equal(results, check)

        # last bout version
        results = e.get_bout_time(df, 'Bout start', 'last', sequence)
        check = pd.Series([3, 3, 4, 7, 7, 7, 10, 10, 10, np.nan], name='Bout start')
        pd.testing.assert_series_equal(results, check)

    def test_calculate_bout_duration(self):
        data = {
            'start': [1,2,3],
            'end': [4,5,6]
        }
        df = pd.DataFrame(data)
        new_df = df.copy()
        new_df['Bout duration'] = pd.Series([3, 3, 3])
        pd.testing.assert_frame_equal(GetBehavioralSequences().calculate_bout_duration(df, 'start', 'end'), new_df)

    def test_calculate_bout_duration(self):
        data = {
            'start': [1.,3.,5.],
            'end': [2.,4.,6.]
        }
        df = pd.DataFrame(data)
        new_df = df.copy()
        new_df['NEW'] = pd.Series([np.nan, 1., 1.])
        pd.testing.assert_frame_equal(GetBehavioralSequences().calculate_interbout_latency(df, 'start', 'end', 'NEW'), new_df)

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
        pd.testing.assert_frame_equal(GetBehavioralSequences().calculate_bout_durations_and_latencies(df), new_df)

    def test__find_sequences(self):
        seq_dict =     {
                'name': 'prop_object',
                'sequence': ['a', 'b'],
                'Bout duration': ['>=0', '<=2'],
                'Latency to next bout start': ['<=4', '>=0'],
                'Bout start': ('1', 'Bout start'),
                'Bout end': ('last', 'Bout end')
        }
        data = {
            'Bout type': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'c'],
            'Bout duration': [2., 1., 2., 3., 2., 4., 1., 17.],
            'Latency to next bout start': [4., 4., 2., 4., 2., 4., 2., np.nan],
            'Bout start': [1., 5., 9., 11., 15., 17., 21., 23.],
            'Bout end': [3., 6., 11., 14., 17., 21., 22., 40.],
        }
        df = pd.DataFrame(data)
        check_data = {
            'Bout type': ['prop_object'],
            'Bout duration': [9.],
            'Latency to next bout start': [4.],
            'Bout start': [5.],
            'Bout end': [14.],
        }
        check = pd.DataFrame(check_data, index=[1])
        results = GetBehavioralSequences()._find_sequences(df, seq_dict)
        pd.testing.assert_frame_equal(results, check)

    def test_add_new_sequences(self):
        data1 = { 
            'Bout start': [3, 5, 7, 9, 11],
            'Bout type': ['a', 'c', 'd', 'e', 'g'],
            'Seq': [0, 0, 0, 0, 0]
        }
        data2 = {
            'Bout start': [4, 4],
            'Bout type': ['aa', 'ab'],
            'Seq': [1, 1]
        }
        check_data = {
            'Bout start': [3, 4, 4, 5, 7, 9, 11],
            'Bout type': ['a', 'aa', 'ab', 'c', 'd', 'e', 'g'],
            'Seq': [0, 1, 1, 0, 0, 0, 0]
        }
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        check_df = pd.DataFrame(check_data)
        results = GetBehavioralSequences().add_new_sequences(df1, df2)
        pd.testing.assert_frame_equal(check_df, results)

    def test_add_seq_column(self):
        df = pd.DataFrame({'a': [1,2,3]})
        check = df.copy()
        results = GetBehavioralSequences.add_seq_column(df, 2)
        check['Seq'] = 2
        print(check)
        print(results)
        pd.testing.assert_frame_equal(results, check)

    @patch.object(GetBehavioralSequences, 'add_seq_column', return_value = 'a')
    @patch.object(GetBehavioralSequences, '_find_sequences', return_value='b')
    @patch.object(GetBehavioralSequences, 'add_new_sequences', return_value='c')
    def test_find_simple_sequences(self, mock_add, mock_find, mock_seq_col):
        sequence_dict = ['seq1', 'seq2']
        df = 'df'
        GetBehavioralSequences(sequences=sequence_dict).find_sequences(df)
        mock_seq_col.assert_any_call('df', 0)
        mock_find.assert_any_call('a', 'seq1')
        mock_add.assert_any_call('a', 'a')
        self.assertEqual(mock_seq_col.call_count, 3)
        self.assertEqual(mock_find.call_count, 2)
        self.assertEqual(mock_add.call_count, 2)

    @patch.object(GetBehavioralSequences, 'load_data', return_value='a')
    @patch.object(GetBehavioralSequences, 'find_sequences', return_value='b')
    @patch.object(GetBehavioralSequences, 'save_files')
    def test_run(self, mock_save, mock_seq, mock_load):
        e = GetBehavioralSequences()
        e.run()
        mock_load.assert_called_with(e.datapath)
        mock_seq.assert_called_with('a')
        mock_save.assert_called_with('b')