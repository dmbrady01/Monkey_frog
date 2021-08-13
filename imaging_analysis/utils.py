#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py: Python script that contains utlilty functions (saving, loading, etc.).

"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "05 Mar 2018"


from neo import io
import neo
import os
import six
import sys

def CleanUpNames(segment, attribute_list=None):
    attributes = attribute_list or ['analogsignals', 'events']
    for attribute in attributes:
        for channel in getattr(segment, attribute):
            channel.name = channel.name.replace("'","").replace("b","")
    return segment

def ReadNeoPickledObj(path='', name="processed.pkl", return_block=False):
    """Reads a pickled neo object. If return_block is True a block is returned.
    Otherwise the segments are returned."""
    # Makes sure that path and name are strings
    if not all(isinstance(arg, six.string_types) for arg in [path, name]):
        raise TypeError('%s or %s is not a strings' % (path, name))
    # Constructs path to pickled object
    if '.pkl' in path:
        dpath = path
    else:
        dpath = path + os.sep + name
    # Checks to see if path exists
    if not os.path.exists(dpath):
        raise IOError('%s cannot be found. Please check that it exists' % dpath)
    # Reads pickled object
    reader = io.PickleIO(dpath)
    # Reads block
    block = reader.read_block()
    # Returns segments or blocks
    if return_block:
        return block
    else:
        return block.segments

def ReadNeoTdt(path, return_block=True):
    """Reads a TdT folder for processing. If return_block is True, a block is
    returned. Otherwise, segments are returned"""
    if not isinstance(path, six.string_types):
        raise TypeError('%s is not a string' % path)
    # Checks to see if path exists
    if not os.path.exists(path):
        raise IOError('%s cannot be found. Please check that it exists' % path)
    # Reads folder of files
    reader = io.TdtIO(dirname=path)
    # Reads block
    block = reader.read_block()
    # Throw error if there are no segments
    if len(block.segments) == 0:
        raise OSError('The data did not load. Please check the path again. Alternatively, make sure you have all the file types necessary.')
    for segment in block.segments:
        segment = CleanUpNames(segment)
    # Returns segments or blocks
    if return_block:
        return block
    else:
        return block.segments


def WriteNeoPickledObj(block=None, path='', name="processed.pkl"):
    """Writes a neo object into a pickle object."""
    # Makes sure block argument is a neo block object
    if not isinstance(block, neo.core.block.Block):
        raise TypeError('%s is not a neo block object' % block)
    # Makes sure that path and name are strings
    if not all(isinstance(arg, six.string_types) for arg in [path, name]):
        raise TypeError('%s or %s is not a strings' % (path, name))
    # Constructs path for our pickled object
    dpath = path + os.sep + name
    # Construct a writing object
    writer = io.PickleIO(dpath)
    # Write object
    writer.write_block(block)


def PrintNoNewLine(string):
    """Prints with out going to a new line
    >>> PrintNoNewLine('hello')
    hello
    """
    sys.stdout.write(string)
    sys.stdout.flush()


