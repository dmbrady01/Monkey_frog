import pandas as pd
import argparse
import os
import numpy as np

SEQUENCE_DICT = [
    {
        'name': 'chamber_to_social',
        'sequence': ['social_chamber', 'social'],
        'Bout duration': ['>=0', '>=0'],
        'Latency to next bout start': ['>=0', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('1', 'Bout end')
    },
    {
        'name': 'chamber_to_object',
        'sequence': ['object_chamber', 'object'],
        'Bout duration': ['>=0', '>=0'],
        'Latency to next bout start': ['>=0', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('1', 'Bout end')
    },
    {
        'name': 'proper_first_social',
        'sequence': ['chamber_to_social', 'social'],
        'Bout duration': ['>=0', '>=1.5'],
        'Latency to next bout start': ['<4', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('2', 'Bout end')
    },
    {
        'name': 'proper_entire_object',
        'sequence': ['chamber_to_object', 'object'],
        'Bout duration': ['>=0', '>=3'],
        'Latency to next bout start': ['<=4', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('last', 'Bout start')
    }
]

class GetBehavioralSequences(object):

    def __init__(self, datapath=None, savepath=None, sequences=SEQUENCE_DICT):
        # Setting datapath and savefolder
        self.datapath = datapath
        self.savepath = savepath
        self.set_savepath()
        # simple sequence analysis
        self.sequences = sequences

    def set_datapath(self):
        if self.datapath is None:
            self.datapath = './data/ethovision.csv'

    @staticmethod
    def get_animal_name(datapath):
        return datapath.split('_')[-1].split('.')[0]

    def set_savepath(self):
        # Will save to the base folder of the datapath (example above saves to ~/Downloads/)
        self.set_datapath()
        if self.savepath is None:
            self.savepath = os.sep.join(self.datapath.split('.csv')[:-1]) + '_behavioral_sequences.csv'

    def save_files(self, df):
        df.to_csv(self.savepath, index=False)

    @staticmethod
    def load_data(datapath):
        return pd.read_csv(datapath)

    @staticmethod
    def sort_by_bout_start(df, cols=['Bout start', 'Seq', 'Bout type'], 
            ascending=[True, True, True]):
        # Combine all results and sort them
        df = df.sort_values(cols, ascending=ascending)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def chain_columns_to_list(df, column, length):
        return pd.concat([df[column].shift(-1*x) for x in range(length)], axis=1).values.tolist()

    @staticmethod
    def map_and_eval(value_list, eval_list):
        zipped_list = list(zip(value_list, eval_list))
        return all([eval(str(x)+y) if pd.notna(x) else False for x, y in zipped_list])

    @staticmethod
    def equivalent(value_list, eval_list):
        return value_list == eval_list

    def match_sequence(self, df, column, sequence, fnc='map_and_eval'):
        iterated_sequences = self.chain_columns_to_list(df, column, len(sequence))
        if fnc == 'map_and_eval':
            mask = [self.map_and_eval(x, sequence) for x in iterated_sequences]
        else:
            mask = [self.equivalent(x, sequence) for x in iterated_sequences]
        return pd.Series(mask)

    @staticmethod
    def get_last_from_bout_type_run(df):
        change_bout_type_mask = df['Bout type'].ne(df['Bout type'].shift().bfill())
        change_bout_type_mask[0] = True
        grouper = change_bout_type_mask.astype(int).cumsum()
        agg_fns = {col: 'last' for col in df.columns}
        new_df = df.groupby(grouper).agg(agg_fns)
        recast = pd.merge(grouper, df.groupby(grouper).agg(agg_fns), left_on='Bout type', right_index=True).drop('Bout type', axis=1)
        recast['Bout type'] = df['Bout type']
        return recast

    def get_bout_time(self, df, column, position, sequence):
        if position == 'last':
            position = len(sequence)
            df = self.get_last_from_bout_type_run(df)

        position = int(position) - 1
        return df[column].shift(-1*position)

    @staticmethod
    def calculate_bout_duration(df, start_col='Bout start', end_col='Bout end'):
        df['Bout duration'] = df[end_col] - df[start_col]
        return df

    @staticmethod
    def calculate_interbout_latency(df, start_col='Bout start', end_col='Bout end', 
        name='Latency from previous bout end', shift=1):
        if shift > 0:
            df[name] = df[start_col] - df[end_col].shift(shift)
        else:
            df[name] = df[end_col].shift(shift) - df[start_col]
        # df['Latency to next bout'] = df['Latency from previous bout'].shift(-1)
        return df

    def calculate_bout_durations_and_latencies(self, df):
        df = self.calculate_bout_duration(df)
        df = self.calculate_interbout_latency(df)
        df = self.calculate_interbout_latency(df, end_col='Bout start', 
            name='Latency from previous bout start')
        df = self.calculate_interbout_latency(df, end_col='Bout start', 
            name='Latency to next bout start', shift=-1)
        return df

    def _find_sequences(self, df, sequence_dict):
        sequence = sequence_dict['sequence']
        duration = sequence_dict['Bout duration']
        latency = sequence_dict['Latency to next bout start']
        bout_start_pos, bout_start_col = sequence_dict['Bout start']
        bout_end_pos, bout_end_col = sequence_dict['Bout end']
        name = sequence_dict['name']
        # Matching sequence, duration, and latencies
        sequence_mask = self.match_sequence(df, 'Bout type', sequence, fnc='equivalent')
        duration_mask = self.match_sequence(df, 'Bout duration', duration, fnc='map_and_eval')
        latency_mask = self.match_sequence(df, 'Latency to next bout start', latency, fnc='map_and_eval')
        full_mask = sequence_mask & duration_mask & latency_mask
        # Extracting start and end times
        start = self.get_bout_time(df, bout_start_col, bout_start_pos, sequence)
        end = self.get_bout_time(df, bout_end_col, bout_end_pos, sequence)
        # copy data
        new_df = df.loc[full_mask, :].copy()
        new_df['Bout type'] = name
        # add updated start/end times and durations
        new_df['Bout start'] = start.loc[full_mask]
        new_df['Bout end'] = end.loc[full_mask]
        new_df = self.calculate_bout_duration(new_df)
        return new_df

    def add_new_sequences(self, df, new_df):
        df = pd.concat([df, new_df], axis=0)
        df = self.sort_by_bout_start(df)
        return df

    @staticmethod
    def add_seq_column(df, sequence_idx):
        df['Seq'] = sequence_idx
        return df

    def find_sequences(self, df):
        df = self.add_seq_column(df, 0)
        for idx, sequence in enumerate(self.sequences):
            idx += 1
            new_events = self._find_sequences(df, sequence)
            new_events = self.add_seq_column(new_events, idx)
            df = self.add_new_sequences(df, new_events)
        return df

    def run(self):
        df = self.load_data(self.datapath)
        df = self.find_sequences(df)
        self.save_files(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', type=str, default='./data/ethovision.csv',
        help='Path to ethovision csv.'
    )
    parser.add_argument(
        '--savepath', type=str, default=None,
        help='Filename to save output to'
    )
    args = parser.parse_args()
    datapath = args.datapath 
    savepath = args.savepath 
    
    event_parser = GetBehavioralSequences(
                                        datapath=datapath, 
                                        savepath=savepath
                                    )
    event_parser.run()