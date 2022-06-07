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
    
    def _get_events(self) -> None:
        events = [e for e in self.segment.events if e.name == self.channel]
        self.events = pd.DataFrame({'time': events[0].times})
        self.events

    def _convert_ttl_to_dataframe(self) -> None:
        self._load_segment()
        self._get_events()

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
    
    def _write_to_file(self) -> None:
        write_path = pathlib.Path(self.path).joinpath('movement_bouts.csv')
        self.bouts.to_csv(write_path, index=False)
    
    def run(self) -> None:
        self._convert_ttl_to_dataframe()
        self._calculate_bout_number()
        self._calculate_bout_timing()
        self._write_to_file()

def get_movement_bouts(path: str, channel: str, min_bout: float, anneal_duration: float) -> None:
    gmb = GetMovementBouts(path=path, channel=channel, min_bout=min_bout, anneal_duration=anneal_duration)
    gmb.run()

@click.command()
@click.option('-p', '--path', help='Path to the data folder')
@click.option('-c', '--channel', default='PC0/', help='TTL channel with events of interest', show_default=True)
@click.option('-m', '--min-bout', default=2.0, help='Minimum duration (seconds) for a bout', show_default=True)
@click.option('-a', '--anneal-duration', default=0.2, help='Maximum time (seconds) between TTL pulses to be considered part of the same bout', show_default=True)
def run(path: str, channel: str, min_bout: float, anneal_duration: float) -> None:
    """Given a folder and TTL channel, will extract those events, and find start/end times
    for all movement bouts of min-bout duration and all TTL pulses in a bout having a latency
    less than anneal-duration."""
    get_movement_bouts(path=path, channel=channel, min_bout=min_bout, anneal_duration=anneal_duration)

if __name__ == '__main__':
    run()