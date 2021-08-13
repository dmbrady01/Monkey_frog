import pandas as pd
import numpy as np
import os
import re
import argparse
from imaging_analysis import utils

BOUT_TYPE_DICT = [
    {
        'location': 'right',
        'zone': ['left interaction', 'left sniffing'],
        'name': 'object'
    },
    {
        'location': 'left',
        'zone': ['left interaction', 'left sniffing'],
        'name': 'social'
    },
    {
        'location': 'right',
        'zone': ['right interaction', 'right sniffing'],
        'name': 'social'
    },
    {
        'location': 'left',
        'zone': ['right interaction', 'right sniffing'],
        'name': 'object'
    },
    {
        'location': 'right',
        'zone': ['left chamber', 'left basin'],
        'name': 'object_chamber'
    },
    {
        'location': 'left',
        'zone': ['left chamber', 'left basin'],
        'name': 'social_chamber'
    },
    {
        'location': 'right',
        'zone': ['right chamber', 'right basin'],
        'name': 'social_chamber'
    },
    {
        'location': 'left',
        'zone': ['right chamber', 'right basin'],
        'name': 'object_chamber'
    },
    {
        'location': 'left',
        'zone': ['center'],
        'name': 'center_chamber'
    },
    {
        'location': 'right',
        'zone': ['center'],
        'name': 'center_chamber'
    }
]

STIMULUS_NAME_SET = {
    'milkshake location',
    'stranger location',
    'stimulus location'
}

ANIMAL_NAME_SET = {
    'animal id',
    'subject id'
}

class GetBehavioralEvents(object):

    def __init__(self, datapath=None, savefolder=None, time_offset=0, 
        time_column='Trial time', minimum_bout_time=0, datatype='ethovision',
        name_match=r'\d{5,}-\d*', max_session_time=600, label_dict=BOUT_TYPE_DICT, 
        offset_datapath=None, fp_datapath=None, stimulus_name_set=STIMULUS_NAME_SET,
        animal_name_set=ANIMAL_NAME_SET, latency_threshold=10, cast='Y'):
        # Timing info
        self.time_offset = time_offset
        self.time_column = time_column
        self.minimum_bout_time = minimum_bout_time
        self.max_session_time = max_session_time
        # Datapaths
        self.savefolder = savefolder
        self.datapath = datapath
        self.datatype = datatype
        self.offset_datapath = offset_datapath
        self.fp_datapath = fp_datapath
        # Animal info
        self.name_match = name_match
        self.label_dict = label_dict
        # Added to stimulus set
        self.stimulus_name_set = stimulus_name_set
        self.animal_name_set = animal_name_set
        self.latency_threshold = latency_threshold
        # cast interaction zone to chamber zone
        self.cast = cast
        self.cast_to_bool()

        #start processes
        self.set_savefolder()

    def set_datapath(self):
        if self.datapath is None:
            self.datapath = './data/ethovision.csv'

    def set_savefolder(self):
        # Will save to the base folder of the datapath (example above saves to ~/Downloads/)
        self.set_datapath()
        if self.savefolder is None:
            self.savefolder = os.sep.join(self.datapath.split(os.sep)[:-1]) + os.sep
        elif self.savefolder[-1] != os.sep:
            self.savefolder = self.savefolder + os.sep

    def save_files(self, dataset):
        for data in dataset:
            animal_name = data[0]
            df = data[1]

            df.to_csv(self.savefolder + self.datatype + '_' + animal_name + '.csv', 
                index=False)

    def cast_to_bool(self):
        if self.cast.lower() == 'y':
            self.cast = True
        else:
            self.cast = False

    def prune_minimum_bouts(self, df):
        return df.loc[df['Bout end'] - df['Bout start'] >= self.minimum_bout_time]

    def add_time_offset(self, df):
        df['Bout start'] = df['Bout start'] + self.time_offset
        df['Bout end'] = df['Bout end'] + self.time_offset
        return df

    @staticmethod
    def sort_by_bout_start(df):
        # Combine all results and sort them
        df = df.sort_values('Bout start')
        df.reset_index(drop=True, inplace=True)
        return df

    def relabel_bout_type(self, zone, stimulus):
        new_name = None
        for case in self.label_dict:
            if any([x in zone.lower() for x in case['zone']]) and (case['location'] in stimulus.lower()):
                new_name = case['name']
        if new_name is None:
            new_name = zone
        return new_name

    def relabel_bout_type_for_df(self, df):
        # Relabel bout types
        df['Bout type'] = df.apply(lambda x: self.relabel_bout_type(x['Bout type'], x['Stimulus Location']), axis=1)
        return df

    @staticmethod
    def clean_and_strip_string(string, sep=' '):
        return sep.join(string.split())

    @staticmethod
    def get_ethovision_header_info(datapath, stimulus_name_set=STIMULUS_NAME_SET, 
        animal_name_set=ANIMAL_NAME_SET, only_skip_lines=False):
        # Reads the dataframe to figure out how many rows to skip
        header_df = pd.read_csv(datapath, header=None)
        lines_to_skip = int(header_df.iloc[0, 1])
        if only_skip_lines:
            return lines_to_skip, None, None
        else:
            # Gets the animal name
            animal_name = header_df.loc[header_df[0].str.lower().isin(animal_name_set), 1].values[0]
            stimulus_location = header_df.loc[header_df[0].str.lower().isin(stimulus_name_set), 1].values[0]

            return lines_to_skip, animal_name, stimulus_location

    def load_ethovision_data(self, datapath=None, 
            stimulus_name_set=STIMULUS_NAME_SET, animal_name_set=ANIMAL_NAME_SET,
            only_skip_lines=False):
        
        lines_to_skip, animal_name, stimulus_location = self.get_ethovision_header_info(
                                        datapath=datapath, 
                                        stimulus_name_set=stimulus_name_set, 
                                        animal_name_set=ANIMAL_NAME_SET, 
                                        only_skip_lines=only_skip_lines
                                    )

        # read the data again
        data = pd.read_csv(datapath, skiprows=[x for x in range(lines_to_skip) if x != lines_to_skip-2])

        return data, animal_name, stimulus_location

    def get_ethovision_start_ttl(self, datapath=None, stimulus_name_set=STIMULUS_NAME_SET, time_column='Trial time'):
        data, _, _ = self.load_ethovision_data(datapath, stimulus_name_set=stimulus_name_set, only_skip_lines=True)

        # find first time value after initialization
        start_value = data.loc[data[time_column] > 1.034, time_column].values[0]

        return start_value

    @staticmethod
    def get_fp_start_ttl(fp_datapath, time_idx=0, event_name='PC0/', event_idx=1):
        block = utils.ReadNeoTdt(path=fp_datapath)
        seglist = block.segments
        seg = seglist[0]
        try:
            event = [x for x in seg.events if event_name in x.name][0]
        except:
            event = seg.events[event_idx]
        return event.times[time_idx].magnitude

    def get_ethovision_offset(self):
        etho_start = self.get_ethovision_start_ttl(datapath=self.offset_datapath, 
            stimulus_name_set=self.stimulus_name_set, time_column=self.time_column)
        fp_start = self.get_fp_start_ttl(self.fp_datapath)
        offset = fp_start - etho_start
        return offset

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

    @staticmethod
    def anneal_bouts(df, latency_threshold=10, latency_col='Latency from previous bout end'):
        # transition mask
        change_bout_type_mask = df['Bout type'].ne(df['Bout type'].shift().bfill())
        change_bout_type_mask.name = None
        change_bout_type_mask[0] = True
        # latency mask
        latency_mask = df[latency_col] >= latency_threshold
        latency_mask.name = None
        # full mask
        full_mask = change_bout_type_mask | latency_mask
        grouper = full_mask.astype(int).cumsum()
        agg_fns = {col: 'first' for col in df.columns}
        agg_fns['Bout end'] = 'last'
        new_df = df.groupby(grouper).agg(agg_fns)
        return new_df[df.columns]

    def process_dataset(self, dataset):
        "Runs the following jobs: add time offset, prune minimum bouts, sort by bout start, relabel bout types, anneal bouts"
        cleaned_dataset = []
        for name, dataframe in dataset:
            sep_df = []
            for bout_type in dataframe['Bout type'].unique():
                df = dataframe.loc[dataframe['Bout type'] == bout_type, :].copy()
                df = self.add_time_offset(df)
                df = self.sort_by_bout_start(df)
                df = self.relabel_bout_type_for_df(df)
                df = self.calculate_bout_durations_and_latencies(df)
                df = self.anneal_bouts(df, latency_threshold=self.latency_threshold)
                df = self.prune_minimum_bouts(df)
                df = self.calculate_bout_durations_and_latencies(df)
                df = self.anneal_bouts(df, latency_threshold=self.latency_threshold)
                df = self.calculate_bout_durations_and_latencies(df)
                sep_df.append(df)
            final_df = pd.concat(sep_df, axis=0)
            final_df = self.sort_by_bout_start(final_df)
            cleaned_dataset.append((name, final_df))
        # dataset = [(x[0], self.anneal_bouts(self.relabel_bout_type_for_df(self.sort_by_bout_start(self.prune_minimum_bouts(self.add_time_offset(x[1])))))) for x in dataset]
        return cleaned_dataset

    @staticmethod
    def convert_ethovision_nulls(data, column):
        mask = ~data.loc[:, column].astype(str).isin({'0', '1'})
        data.loc[mask, column] = np.nan
        return data

    @staticmethod
    def forward_fill_ethovision_data(data, column):
        data.loc[:, column] = data.loc[:, column].ffill()
        return data

    @staticmethod
    def convert_ethovision_to_int(data, column):
        data.loc[:, column] = data.loc[:, column].astype(int)
        return data

    def convert_ethovision_data_to_proper_format(self, data, column):
        data = self.convert_ethovision_nulls(data, column)
        data = self.forward_fill_ethovision_data(data, column)
        data = self.convert_ethovision_to_int(data, column)
        return data

    @staticmethod
    def boolean_cast_data(data, to_column, from_column):
        data.loc[:, to_column] = data.loc[:, to_column] | data.loc[:, from_column]
        return data

    def process_ethovision(self):
        data, animal_name, stimulus_location = self.load_ethovision_data(self.datapath, 
            stimulus_name_set=self.stimulus_name_set)

        # Get the zone columns
        zone_columns = [x for x in data.columns if ( ('In zone' in x) and ('nose-point' in x) )]
        # # relabel zone columns
        # zone_columns = [self.relabel_bout_type(x, stimulus_location) for x in zone_columns]

        results_df = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end', 'Stimulus Location'])

        # Fix nulls and data types
        for column in zone_columns:
            data = self.convert_ethovision_data_to_proper_format(data, column)

        if self.cast:
            # cast data left
            left_int = [x for x in zone_columns if ('left' in x.lower()) and ( ('interaction' in x.lower()) or ('sni' in x.lower()) )][0]
            left_cham = [x for x in zone_columns if ('left' in x.lower()) and ( ('chamber' in x.lower()) or ('basin' in x.lower()) )][0]

            right_int = [x for x in zone_columns if ('right' in x.lower()) and ( ('interaction' in x.lower()) or ('sni' in x.lower()) )][0]
            right_cham = [x for x in zone_columns if ('right' in x.lower()) and ( ('chamber' in x.lower()) or ('basin' in x.lower()) )][0]
            
            data = self.boolean_cast_data(data, right_cham, right_int)
            data = self.boolean_cast_data(data, left_cham, left_int)

        for column in zone_columns:
            zone_df = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end', 'Stimulus Location'])
            # Separate by zone type
            # zone_series = data.loc[:, column]
            # Get rid of missing data
            zone_series = data.loc[data.loc[:, column].astype(str).isin({'0', '1'}), column].astype(int)
            # Mask for entering/exiting
            change_mask = zone_series.diff().abs() > 0
            # Entries and exits indices
            entries_index = zone_series.loc[change_mask & (zone_series == 1)].index
            exits_index = zone_series.loc[change_mask & (zone_series == 0)].index
            # Entries and exits time
            entry_times = data.loc[entries_index, self.time_column].values
            exit_times = data.loc[exits_index, self.time_column].values
            # Add final time if mouse never does a last exit
            if len(exit_times) < len(entry_times):
                exit_times = np.append(exit_times, data.loc[data.index[-1], self.time_column])
            elif len(exit_times) > len(entry_times):
                entry_times = np.append(data.loc[0, self.time_column], entry_times)
            # Add to zone_df
            zone_df['Bout start'] = entry_times
            zone_df['Bout end'] = exit_times
            zone_df['Bout type'] = column
            zone_df['Stimulus Location'] = stimulus_location

            results_df = pd.concat([results_df, zone_df], axis=0)

        return [(animal_name, results_df)]

    def process_anymaze(self):
        with open(self.datapath, 'r') as fp:
            contents = fp.readlines()

        animal_regex = re.compile(self.name_match)
        zone_regex = re.compile(r'zone')
        milkshake_loc_regex = re.compile(r'^(Left|Right)')
        time_regex = re.compile(r'secs\. 1')

        animals = set([x.split()[0] for x in contents if bool(animal_regex.search(x))])
        datadict = {}
        for animal in animals:
            datadict[animal] = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end', 'Stimulus Location'])

        zone = None
        milkshake_loc = None
        animal = None
        prev_time = 0
        curr_time = 0
        bout_start = None
        bout_end = None

        for idx, line in enumerate(contents):
            line = self.clean_and_strip_string(line)
            # Check zone
            if bool(zone_regex.search(line)):
                zone = line

            # Check milkshake location
            elif bool(milkshake_loc_regex.search(line)):
                milkshake_loc = line

            # Check animal location
            elif bool(animal_regex.search(line)):
                animal = line

            # Check time information - skip case (N = 0)
            elif time_regex.search(line):
                # Splits the time row '100 - 101 secs.          1   0.00    0.0' ->
                # [100, 101, 1, 0, 0]
                time_row = [float(x) for x in line.replace('- ', '').replace('secs. ', '').split(' ')]
                curr_time = time_row[-1]

                # bout start
                if (prev_time == 0) and (curr_time > 0):
                    bout_start = time_row[0] + curr_time
                    start_ind = idx
                # bout start at beginning of session
                elif (curr_time > 0) and (time_row[0] == 0):
                    bout_start = time_row[0] + curr_time
                    start_ind = idx
                # bout end
                elif (prev_time > 0) and (curr_time == 0):
                    bout_end = time_row[0] + prev_time - 1
                    end_ind = idx
                # # bout end because session ends
                elif (curr_time > 0) and (time_row[1] == self.max_session_time):
                    bout_end = time_row[0] + curr_time
                    end_ind = idx

                # Log bout and reset bout start/end
                if (bout_start is not None) and (bout_end is not None):
                    # prepare dictionary for row append
                    row = {
                            'Bout type': zone,
                            'Bout start': bout_start,
                            'Bout end': bout_end,
                            'Stimulus Location': milkshake_loc
                    }
                    datadict[animal] = datadict[animal].append(row, ignore_index=True)
                    bout_start = None
                    bout_end = None
                    prev_time = 0
                    curr_time = 0

                prev_time = curr_time

        return list(datadict.items())

    def run(self):
        self.set_savefolder()
        if self.datatype == 'ethovision':
            # Calculate offset
            if (self.fp_datapath is not None) and (self.offset_datapath is not None):
                self.time_offset = self.get_ethovision_offset()
            dataset = self.process_ethovision()
        elif self.datatype == 'anymaze':
            dataset = self.process_anymaze()
        dataset = self.process_dataset(dataset)
        self.save_files(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', type=str, default='./data/ethovision.csv',
        help='Path to ethovision csv.'
    )
    parser.add_argument(
        '--savefolder', type=str, default=None,
        help='Folder to save data to'
    )
    parser.add_argument(
        '--time-column', type=str, default='Trial time',
        help='Column in dataframe with times',
    )
    parser.add_argument(
        '--time-offset', type=float, default=0,
        help='How much time to offset all bouts (to align with imaging data)',
    )
    parser.add_argument(
        '--minimum-bout-time', type=float, default=1,
        help='minimum length of time for a bout',
    )
    parser.add_argument(
        '--datatype', type=str, default='ethovision',
        help='datatype (ethovision, anymaze, etc.)'
        )
    parser.add_argument(
        '--max-session-time', type=float, default=600,
        help='Maximum time for a recording session'
        )
    parser.add_argument(
        '--offset-datapath', type=str, default=None,
        help='Path to ethovision hardware arena csv'
        )
    parser.add_argument(
        '--fp-datapath', type=str, default=None,
        help='Path to FP data')
    parser.add_argument(
        '--latency-threshold', type=float, default=None,
        help='Latency threshold between similar bouts to prevent annealing')
    parser.add_argument(
        '--cast', type=str, default='Y',
        help='True/False that interaction zones are a subset of chamber zones')
    args = parser.parse_args()
    datapath = args.datapath 
    savefolder = args.savefolder 
    time_offset = args.time_offset 
    time_column = args.time_column 
    minimum_bout_time = args.minimum_bout_time
    datatype = args.datatype
    max_session_time = args.max_session_time
    offset_datapath = args.offset_datapath
    fp_datapath = args.fp_datapath
    latency_threshold = args.latency_threshold
    cast = args.cast
    
    event_parser = GetBehavioralEvents(
                                        datapath=datapath, 
                                        savefolder=savefolder, 
                                        time_offset=time_offset, 
                                        time_column=time_column, 
                                        minimum_bout_time=minimum_bout_time,
                                        datatype=datatype,
                                        max_session_time=max_session_time,
                                        offset_datapath=offset_datapath,
                                        fp_datapath=fp_datapath,
                                        latency_threshold=latency_threshold,
                                        cast=cast
                                    )
    event_parser.run()