#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
get_channel_and_event_names.py: Python script that takes a path to a TDT data folder and spits
out the segment channel and event names
"""


__author__ = "DM Brady"
__datewritten__ = "07 Mar 2018"
__lastmodified__ = "13 Jun 2018"

import sys
from imaging_analysis.utils import ReadNeoTdt, PrintNoNewLine

if len(sys.argv) < 2:
    dpath="data/KN_newRigData/RS/12/FirstFibPho-180817-160254/"
else:
    dpath = sys.argv[1]

PrintNoNewLine('\nReading TDT folder...')
block = ReadNeoTdt(path=dpath, return_block=True)
seglist = block.segments
print('Done!')

for segment in seglist:
    # get channel names
    channel_names = [x.name for x in segment.analogsignals]
    event_names = [x.name for x in segment.events]

    print(("The channel names are %s and event names are %s\n"
        % (str(channel_names), str(event_names))))