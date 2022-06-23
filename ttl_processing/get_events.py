import pandas as pd
from imaging_analysis.utils import ReadNeoTdt
import pathlib
import click

class GetEvents(object):

    def __init__(self, path: str, channel: str='PC0/') -> None:
        self.path = path
        self.channel = channel

    def _load_segment(self) -> None:
        self.segment = ReadNeoTdt(path=self.path, return_block=False)[0]
    
    def _get_events(self) -> pd.DataFrame:
        events = [e for e in self.segment.events if e.name == self.channel]
        self.events = pd.DataFrame({'time': events[0].times})
        return self.events

    def _write_to_file(self, df: pd.DataFrame, filename: str) -> None:
        write_path = pathlib.Path(self.path).joinpath(filename)
        df.to_csv(write_path, index=False)

    def _convert_ttl_to_dataframe(self) -> pd.DataFrame:
        self._load_segment()
        return self._get_events()

class GetMovementBouts(GetEvents):

    def __init__(self, min_bout: float=2.0, anneal_duration: float=0.3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_bout = min_bout
        self.anneal_duration = anneal_duration
    
    def _calculate_bout_number(self) -> None:
        self.events['bout_number'] = (self.events['time'].diff() > self.anneal_duration).cumsum() + 1
    
    def _calculate_bout_timing(self) -> None:
        self.bouts = pd.DataFrame({
            'Bout start': self.events.groupby('bout_number').first()['time'],
            'Bout end': self.events.groupby('bout_number').last()['time'],
            'Bout type': 'Movement'
        })
        self.bouts['Bout duration'] = self.bouts['Bout end'] - self.bouts['Bout start']
        self.bouts = self.bouts.loc[self.bouts['Bout duration'] >= self.min_bout, :]
        self.bouts.reset_index(drop=True, inplace=True)
    
    def run(self) -> None:
        self._convert_ttl_to_dataframe()
        self._calculate_bout_number()
        self._calculate_bout_timing()
        self._write_to_file(df=self.bouts, filename='movement_bouts.csv')

def _get_ttl_events(path: str, channel: str, filename: str) -> None:
    gb = GetEvents(path=path, channel=channel)
    df = gb._convert_ttl_to_dataframe()
    gb._write_to_file(df=df, filename=filename)

def _get_movement_bouts(path: str, channel: str, min_bout: float, anneal_duration: float) -> None:
    gmb = GetMovementBouts(path=path, channel=channel, min_bout=min_bout, anneal_duration=anneal_duration)
    gmb.run()

@click.group()
def cli() -> None:
    "Group of cli methods"

@cli.command()
@click.option('-p', '--path', help='Path to the data folder')
@click.option('-c', '--channel', default='PC2/', help='TTL channel with events of interest', show_default=True)
@click.option('-f', '--filename', default='event_timestamps.csv', help='Name for the CSV output file', show_default=True)
def get_ttl_events(path: str, channel: str, filename: str) -> None:
    """Given a folder, TTL channel, and file name, will extract those events, and write the 
    timestamps to a CSV file."""
    _get_ttl_events(path=path, channel=channel, filename=filename)

@cli.command()
@click.option('-p', '--path', help='Path to the data folder')
@click.option('-c', '--channel', default='PC0/', help='TTL channel with events of interest', show_default=True)
@click.option('-m', '--min-bout', default=2.0, help='Minimum duration (seconds) for a bout', show_default=True)
@click.option('-a', '--anneal-duration', default=0.2, help='Maximum time (seconds) between TTL pulses to be considered part of the same bout', show_default=True)
def get_movement_bouts(path: str, channel: str, min_bout: float, anneal_duration: float) -> None:
    """Given a folder and TTL channel, will extract those events, and find start/end times
    for all movement bouts of min-bout duration and all TTL pulses in a bout having a latency
    less than anneal-duration."""
    _get_movement_bouts(path=path, channel=channel, min_bout=min_bout, anneal_duration=anneal_duration)

if __name__ == '__main__':
    cli()