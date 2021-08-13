# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:23:37 2017

@author: Morishita Lab
"""

from neo import io
from neo.core import Event, SpikeTrain, Epoch, AnalogSignal
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import matplotlib.patches as pch
import numpy as np
import quantities as pq
import copy as cp
import elephant.signal_processing as esp
import scipy.signal as ssp
import warnings
import os
import sys


def process_events(seg, tolerence):
    if 'Events' not in [cur_evts.name for cur_evts in seg.events]:
        evtlist = list()
        event_times = list()
        event_labels = list()
        for evtarr in seg.events:
            if 'DIn' in evtarr.name:
                evtlist.append(dict(times=evtarr.times,
                                    ch=int(evtarr.name[-1]) - 1))
        while any(event_array['times'].size for event_array in evtlist):
            evtlist_non_empty = [x for x in evtlist if x['times'].size]
            first_elements = [x['times'][0] for x in evtlist_non_empty]
            cur_first = np.amin(first_elements) * pq.s
            cur_event = 0
            cur_event_list = [0] * len(evtlist)
            for evtarr in evtlist_non_empty:
                if evtarr['times'][0] - cur_first < tolerence:
                    cur_event_list[evtarr['ch']] = 1
                    evtarr['times'] = np.delete(evtarr['times'], 0) * pq.s
            for bit in cur_event_list:
                cur_event = (cur_event << 1) | bit
            event_times.append(cur_first)
            event_labels.append(cur_event)
            evtlist = evtlist_non_empty
        result = Event(times=np.array(event_times)*pq.s,
                       labels=np.array(event_labels, dtype='S'),
                       name='Events')
        seg.events.append(result)
    else:
        print("Events array already presented!")


def process_trials(seg, evtdict):
    if 'Events' not in [cur_evts.name for cur_evts in seg.events]:
        warnings.warn("Missing Events. Process events first!")
        return []
    else:
        events = next(evt for evt in seg.events if evt.name == 'Events')
        trials = np.empty(
            shape=len(events),
            dtype=[
                ('time', events.times.dtype), ('eventcode', events.labels.dtype),
                ('eventname', 'S50'), ('trialindex', 'i2'), ('result', 'S50')]
        )
        trials['time'] = events.times
        trials['eventcode'] = events.labels
        tidx = 0
        for idevt, evt in enumerate(trials):
            evtname = evtdict[evt['eventcode']]
            if evtname == 'iti_start':
                tidx += 1
            trials['eventname'][idevt] = evtname
            trials['trialindex'][idevt] = tidx
        result_list = ['correct', 'incorrect', 'omission', 'premature']
        if trials:
            for tidx in range(np.max(trials['trialindex']) + 1):
                cur_evtid = np.where(trials['trialindex'] == tidx)
                cur_evts = trials[cur_evtid]
                result_arr = np.array([np.count_nonzero(cur_evts['eventname'] == re) for re in result_list])
                if np.sum(result_arr) < 1:
                    warnings.warn("No result found for trial: " + str(tidx))
                    trials['result'][cur_evtid] = 'NONE'
                elif np.sum(result_arr) > 1:
                    warnings.warn("Multiple results found for trial: " + str(tidx))
                    trials['result'][cur_evtid] = 'MULTIPLE'
                else:
                    trials['result'][cur_evtid] = result_list[np.asscalar(np.flatnonzero(result_arr))]
        else:
            warnings.warn("zero sized trials. check if events are processed.")
        return trials


def process_epoch(seg, trials, event, duration, epochname, mask=None):
    if not isinstance(trials, np.ndarray):
        warnings.warn("not a valid trials!")
        return
    if mask is None:
        mask = [True] * len(trials)
    subtrial = trials[mask]
    subtrial = subtrial[subtrial['eventname'] == event]
    seg.epochs.append(Epoch(
        times=subtrial['time'] * duration[0].units + duration[0],
        durations=[duration[1] - duration[0]] * len(subtrial), name=epochname
    ))


def plot_events(seg, **kwargs):
    ax = plt.gca()
    for ith, evtarr in enumerate(seg.events):
        plt.vlines(evtarr.times, ith, ith + 1, **kwargs)
        if evtarr.name == 'Events':
            for i, t in enumerate(evtarr.times):
                ax.annotate(evtarr.labels[i],
                            xy=(t, ith+1),
                            xytext=(t, ith+1.2),
                            arrowprops=dict(arrowstyle="->"))
    ax.set_yticks(np.arange(.5, len(seg.events) + .5, 1))
    ax.set_yticklabels(list(evtarr.name for evtarr in seg.events))
    return ax


def plot_segment(seg, analist=(), spklist=(), epchlist=(), showevent=None, embedpeaks=True, recenter=None):
    anas = [x for x in seg.analogsignals if x.name in analist]
    spks = [x for x in seg.spiketrains if x.name in spklist]
    evts = next((e for e in seg.events if e.name == 'Events'), None)
    if not evts:
        warnings.warn("No Events presented in data. Turning showevent off.")
        showevent = False
    if (not anas) and (not spks):
        print("Empty data lists. Nothing plotted")
        return
    if not epchlist:
        if spks:
            tunit = spks[0].t_start.units
            tstart = np.min([s.t_start for s in spks])
            tstop = np.max([s.t_stop for s in spks])
        elif anas:
            tunit = anas[0].t_start.units
            tstart = np.min([a.t_start for a in anas])
            tstop = np.max([np.max(a.times) for a in anas])
        else:
            tunit = pq.s
            tstart = 0
            tstop = 0
        epchlist = [Epoch(name='all', times=[tstart] * tunit, durations=[tstop - tstart] * tunit)]
    else:
        epchlist = [ep for ep in seg.epochs if ep.name in epchlist]
    nrow = len(analist) + len(spklist)
    ncol = len(epchlist)
    fig = plt.figure(figsize=(ncol*18, nrow*6))
    fig.suptitle(seg.name)
    gs = grdspc.GridSpec(nrow, ncol)
    for epid, epch in enumerate(epchlist):
        row = 0
        cur_anas = [slice_signals(a, epch, recenter) for a in anas]
        cur_spks = [slice_signals(s, epch, recenter) for s in spks]
        if showevent:
            cur_evts = slice_signals(evts, epch, recenter)
        for ana in cur_anas:
            ax = fig.add_subplot(gs[row, epid])
            for aid, a in enumerate(ana):
                ax.plot(a.times, a, linewidth=0.3, alpha=0.5, color='darkgrey')
                if embedpeaks:
                    arrlen = (np.max(a) - np.min(a)) / 10
                    peaks = next((spk[aid] for spk in cur_spks if spk[aid].name == a.name), [])
                    for p in peaks:
                        pidx = np.argmin(np.abs(a.times - p))
                        pidx = int(pidx)
                        ax.annotate(
                            '', xy=(p, a[pidx]),
                            xytext=(p, a[pidx]+arrlen),
                            arrowprops={'arrowstyle': "->"}
                        )
                if showevent:
                    for et in cur_evts[aid].times:
                        ax.add_patch(pch.Rectangle(
                            (et - showevent, np.min(a)),
                            showevent * 2, np.max(a) - np.min(a),
                            alpha=0.2,
                            edgecolor='none'
                        ))
            ana_mean = np.mean(trim_signals(ana), axis=0)
            ax.plot(ana[0].times, ana_mean)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("signal")
            # ax.set_ylim(0, 1)
            ax.set_title(
                "Signal: " + ana[0].name + "\n" + epch.name
            )
            row += 1
        for spk in cur_spks:
            ax = fig.add_subplot(gs[row, epid])
            for sid, s in enumerate(spk):
                if s.times.size:
                    ax.vlines(s.times, sid, sid + 1)
                    if showevent:
                        for et in cur_evts[sid].times:
                            ax.add_patch(pch.Rectangle(
                                (et - showevent, sid),
                                showevent * 2, 1,
                                alpha=0.2,
                                edgecolor='none'
                            ))
            ax.set_title(
                "Signal: " + spk[0].name + "\n" + epch.name
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel("spike ID")
            row += 1
    return fig


def slice_signals(sig, epch, recenter=None):
    sig_list = []
    for epid, tstart in enumerate(epch.times):
        s = sig.time_slice(tstart, tstart + epch.durations[epid])
        if recenter is not None:
            if isinstance(s, AnalogSignal):
                s.t_start = s.t_start - tstart - recenter
            elif isinstance(s, SpikeTrain):
                s.t_start = s.t_start - tstart - recenter
                # s.t_stop = s.t_stop - tstart - recenter
                s = s - tstart - recenter
            elif isinstance(s, Event):
                s = s - tstart - recenter
            else:
                warnings.warn("Unsupported signal type. No re-centering")
        sig_list.append(s)
    return sig_list


def trim_signals(sigs, size=None):
    if not size:
        size = min([len(s) for s in sigs])
    for isig, sig in enumerate(sigs):
        sigs[isig] = sigs[isig][:size]
    return sigs


def z_score(seg, varname):
    varlist = [v for v in seg.analogsignals if v.name in varname]
    zlist = list()
    for sig in varlist:
        z_sig = cp.copy(sig)
        z_sig[:] = np.absolute(esp.zscore(z_sig))
        z_sig.name = z_sig.name + '_zscore'
        zlist.append(z_sig)
    seg.analogsignals = seg.analogsignals + zlist


def norm_data(seg, varname):
    varlist = [v for v in seg.analogsignals if v.name in varname]
    normlist = list()
    for sig in varlist:
        background = ssp.savgol_filter(sig, 3001, 1, axis=0)
        norm_sig = cp.copy(sig)
        norm_sig[:] = sig - background * sig.units
        norm_sig.name = norm_sig.name + '_norm'
        normlist.append(norm_sig)
    seg.analogsignals = seg.analogsignals + normlist


def find_peak(seg, varname):
    for i, sig in enumerate(seg.analogsignals):
        if sig.name in varname:
            print(("finding peaks for "+sig.name))
            p = ssp.find_peaks_cwt(sig.flatten(), np.arange(100, 550),
                                   min_length=10)
            peaks = p / sig.sampling_rate + sig.t_start
            peaks = SpikeTrain(times=peaks, t_start=sig.t_start,
                               t_stop=sig.t_stop, name=sig.name)
            seg.spiketrains.append(peaks)


def slice_segment(seg, tslice):
    for i, sig in enumerate(seg.analogsignals):
        seg.analogsignals[i] = sig.time_slice(tslice[0], tslice[1])
    for i, spk in enumerate(seg.spiketrains):
        seg.spiketrains[i] = spk.time_slice(tslice[0], tslice[1])
    for i, evt in enumerate(seg.events):
        seg.events[i] = evt.time_slice(tslice[0], tslice[1])


# %% defining path
if __name__ == '__main__':
    event_dict = {
        '1': 'correct', '2': 'incorrect', '3': 'iti_start', '4': 'omission',
        '5': 'premature', '6': 'stimulus', '7': 'tray'
    }
    try:
        dpath = sys.argv[1]
    except:
        dpath = '/Users/DB/Development/Morishita_lab_2photon/data/TDT-LockinRX8-22Oct2014_20-4-15_DT4_1024174'
    # %% load data from tdt files
    try:
        reader = io.PickleIO(dpath + os.sep + "processed.pkl")
        block = reader.read_block()
        seglist = block.segments
    except:
        reader = io.TdtIO(dirname=dpath)
        block = reader.read_block()
        seglist = block.segments
        # %% load data from pkl files
        # %% process normalization/ascore/peaks
        for segid, segment in enumerate(seglist):
            process_events(segment, 5 * pq.s)
            print('processed events')
            norm_data(segment, ['LMag 1'])
            print('normalized data')
            z_score(segment, ['LMag 1_norm'])
            print('z scored data')
            find_peak(segment, ['LMag 1_norm_zscore'])
            print('found all peaks')
        # %% writing the data to pkl files
        print('writing pickled object')
        writer = io.PickleIO(dpath + os.sep + 'processed.pkl')
        writer.write_block(block)
    # %% subsetting part of the data
    seg_toplot = seglist[0]
    slice_segment(seg_toplot, [539*pq.s,554*pq.s])
    # %% process trials
    trialblock = process_trials(seg_toplot, event_dict)
    process_epoch(
        seg_toplot, trialblock, 'iti_start',
        duration=(1 * pq.s, 900 * pq.s),
        epochname='iti_start in correct trials',
        mask=trialblock['result'] == 'correct')
    process_epoch(
        seg_toplot, trialblock, 'iti_start',
        duration=(1 * pq.s, 900 * pq.s),
        epochname='iti_start in incorrect trials',
        mask=trialblock['result'] == 'incorrect')
    process_epoch(
        seg_toplot, trialblock, 'iti_start',
        duration=(1 * pq.s, 900 * pq.s),
        mask=trialblock['result'] == 'omission')
    # %% plot data
    segplot = plot_segment(
        seg_toplot,
        analist=['LMag 1_norm_zscore'],
        spklist=['LMag 1_norm_zscore'],
        showevent=0.1*pq.s
    )
    segplot.savefig("Blk-2.svg", bbox_inches='tight')
