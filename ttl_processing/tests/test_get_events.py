import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pathlib
from click.testing import CliRunner
from typing import List

from ttl_processing.get_events import GetEvents, GetMovementBouts, get_movement_bouts, run

class MockEvent(object):
    def __init__(self, name: str, times: List[float]):
        self.name = name
        self.times = times

class MockSegment(object):
    events = [MockEvent('channel', [1., 2.]), MockEvent('other_channel', [3., 4.])]

EVENTS = pd.DataFrame({
    'time': [0, 1.2, 1.4, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 10, 10.8, 11.5, 12.2, 13, 20]
})

class TestGetEvents(unittest.TestCase):
    
    def test_init(self):
        ge = GetEvents(path='path', channel='channel')
        assert ge.path == 'path'
        assert ge.channel == 'channel'

    @patch("ttl_processing.get_events.ReadNeoTdt", return_value=['segment'])
    def test_load_segment(self, mock_read):
        ge = GetEvents(path='path', channel='channel')
        ge._load_segment()
        mock_read.assert_called_with(path=ge.path, return_block=False)
        ge.segment == 'segment'

    def test_get_events(self):
        ge = GetEvents(path='path', channel='channel')
        ge._load_segment = MagicMock()
        ge.segment = MockSegment()
        ge._get_events()
        check_df = pd.DataFrame({'time': [1., 2.]})
        pd.testing.assert_frame_equal(check_df, ge.events)

    def test_convert_ttl_to_dataframe(self):
        ge = GetEvents(path='path', channel='channel')
        ge._load_segment = MagicMock()
        ge._get_events = MagicMock()
        ge._convert_ttl_to_dataframe()
        ge._load_segment.assert_called_once()
        ge._get_events.assert_called_once()

class TestGetMovementBouts(unittest.TestCase):

    def test_init(self):
        gmb = GetMovementBouts(path='path', channel='channel', min_bout=5, anneal_duration=1)
        assert gmb.path == 'path'
        assert gmb.channel == 'channel'
        assert gmb.min_bout == 5
        assert gmb.anneal_duration == 1
    
    def test_calculate_bout_number(self):
        gmb = GetMovementBouts(path='path', channel='channel', min_bout=2, anneal_duration=1)
        gmb.events = EVENTS
        gmb._calculate_bout_number()
        bout_check = pd.Series([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4], name='bout_number')
        pd.testing.assert_series_equal(gmb.events['bout_number'], bout_check)

    def test_calculate_bout_timing(self):
        gmb = GetMovementBouts(path='path', channel='channel', min_bout=2, anneal_duration=1)
        gmb.events = EVENTS
        gmb._calculate_bout_timing()
        check_df = pd.DataFrame({
            'Bout start': [1.2, 10.],
            'Bout end': [6., 13.],
            'Bout type': ['Movement', 'Movement'],
            'Bout duration': [4.8, 3.]
        })
        pd.testing.assert_frame_equal(check_df, gmb.bouts)
    
    def test_write_to_file(self):
        gmb = GetMovementBouts(path='path', channel='channel', min_bout=2, anneal_duration=1)
        gmb.bouts = MagicMock()
        gmb._write_to_file()
        path_check = pathlib.Path('path/movement_bouts.csv')
        gmb.bouts.to_csv.assert_called_with(path_check, index=False)

    def test_run(self):
        gmb = GetMovementBouts(path='path', channel='channel', min_bout=2, anneal_duration=1)
        gmb._convert_ttl_to_dataframe = MagicMock()
        gmb._calculate_bout_number = MagicMock()
        gmb._calculate_bout_timing = MagicMock()
        gmb._write_to_file = MagicMock()
        gmb.run()
        gmb._convert_ttl_to_dataframe.assert_called_once()
        gmb._calculate_bout_number.assert_called_once()
        gmb._calculate_bout_timing.assert_called_once()
        gmb._write_to_file.assert_called_once()

class TestCli(unittest.TestCase):

    @patch.object(GetMovementBouts, 'run')
    def test_get_movement_bouts(self, mock_run):
        get_movement_bouts('path', 'ch', 10, 1)
        mock_run.assert_called_once()

    @patch("ttl_processing.get_events.get_movement_bouts")
    def test_run(self, mock_get_bouts):
        runner = CliRunner()
        result = runner.invoke(run, ['--path', 'mypath', '--channel', 'mych', '--min-bout', 10, '--anneal-duration', 1])
        assert result.exit_code == 0
        mock_get_bouts.assert_called_with(path='mypath', channel='mych', min_bout=10, anneal_duration=1)