#!/usr/bin/env python

'''
L2 transect processing
'''

import numpy as np
from datetime import datetime
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from scipy.signal import windows as windows
from scipy.ndimage import filters
from scipy.interpolate import interp1d
from pycurrents.system import Bunch
from pycurrents.data.navcalc import (great_circle_distance, diffxy_from_lonlat)
from pysacp.pysadcp import translate_inst_str
# from pycurrents.file import npzfile


dameth = .7  # method of for regularization of grid
dlm = 2.  # segment nominal regular grid resolution in km
Nr = 500.  # segment nominal regular grid length in km
slen = 500.
gp_tol = 10  # max small gap percent tolerated
gs_tol = 20.  # max gap size in km tolerated

# domain specifications (for reductions):
lsl, rsl = -155., -90.
tsl, bsl = 25., 5.

# define target depths:
dois = [15., 29., 45., 62, 125., 240., 350., 500., 1000.]  # depths of interest
tdep_tol = [4., 8., 8., 8., 16., 16., 32., 32., 32.]  # ought to be a function of sonar; used in determining depth bin

tp = 33  # in hrs; running filter window length for uship/vship smoothing
# binavg = True
# nbas = [1, 1, 1, 1, 3, 3, 5, 7, 9]  # ideally also done by sonar freq; give layer
nbas = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3]

out_dir = './data/proc_adcp/proc_seg_stacks/'
out_stack_filename_hdr = "stacks_global_z"


class TransectsDB:
    """
    Workhorse class to represent and intereact with a transects database.

    methods:

        pre_process(transects, binavg)

        method to select subregions and depth slices, and performing
        vertical averaging and assessments of gaps

    """

    def __init__(self, transects,):
        self.sonars = self._heal_sonars(transects['inst_id'])
        self.vessels = self._clip_str(transects['vessel_id'])
        self.cruises = self._clip_str(transects['cruise_id'])
        self.sac_ids = transects['sac_id']
        self.years = transects['year']
        self.months = transects['month']

        self.Idxs = []  # list of bool raw nav indices array
        self.Didx = []  # list of bool depth indices array
        self.Dcover = []  # list of covered distance in km for selection
        self.Nidxs = []  # list of bool nav indices w/ valid depths
        self.Adep = []
        self.Nngaps = []
        self.Ngap_max = []
        self.Wlut = []  # list of valid transect indices for selection
        self.Bad_bin_depth = []
        self.Lost_transects = []  # list w/ indices of lost/bad transects
        self.was_preprocessed = False

        self.transects = transects

    def _heal_sonars(self, sonars):
        # the dict below should be imported from pysadcp
        accepted_sonar_strs = {'wh300', 'wh600', 'wh1200', 'bb150', 'nb150',
                               'os150nb', 'os150bb', 'os150??', 'os150',
                               'os75nb', 'os75bb', 'os75??', 'os75', 'os38nb',
                               'os38bb', 'os38??', 'os38', 'unknown_read_fail',
                               'unknown_no_metafile', 'nb300', 'bb300'}
        for n, sonar in enumerate(sonars):
            if sonar not in accepted_sonar_strs:
                fsonar = None
                for asonar in accepted_sonar_strs:
                    if asonar in sonar:
                        fsonar = asonar
                if fsonar is None:
                    fsonar = translate_inst_str(sonar)
                sonars[n] = fsonar
        return sonars

    def _clip_str(self, str_arr):
        for n, strs in enumerate(str_arr):
            str_arr[n] = strs.rstrip()
        return str_arr

    def pre_process(self, transects, binavg):

        if self.was_preprocessed:
            print("pre_processing has been run. Need to reset lists")
            return

        for n in range(0, len(transects)):

            if type(binavg) == str:
                nbins = get_nbin_by_sonar(self.sonars[n], thickness)  # move out of loop? No! nbins will be a fixed value integer but vary by sonar
            elif type(binavg) == int:
                nbins = np.asarray([binavg])
            elif type(binavg) == list:
                assert len(binavg) == len(dois)
                nbins = np.asarray(binavg)
            else:
                raise ValueError('Error in setting bin averaging')

            tdata = transects['seg_data'][n]
            tlon = tdata.lon
            tlat = tdata.lat
            tdl = tdata.dl
            ila = np.logical_and(tlat >= bsl, tlat <= tsl)
            ilo = np.logical_and(tlon >= lsl, tlon <= rsl)
            idxs = np.logical_and(ila, ilo)  # valid nav indices

            # find gaps in acquisition (small are unrecorded & dealt w/ later):
            time_gap_loc = np.where(tdl > gs_tol)[0]
            if len(time_gap_loc) > 0:
                time_gap_loc = np.concatenate(([0], time_gap_loc, [len(idxs)]))
                new_idxs_list = []
                for m in range(0, len(time_gap_loc) - 1):
                    gidx = idxs[time_gap_loc[m]:time_gap_loc[m+1]]
                    leng_sec = tdl[time_gap_loc[m]:time_gap_loc[m+1]][gidx].sum()
                    if leng_sec >= slen:
                        ridxs = idxs.copy()
                        ridxs[:] = False
                        ridxs[time_gap_loc[m]:time_gap_loc[m+1]] = True
                        new_idxs_list.append(np.logical_and(ridxs, idxs))
                if len(new_idxs_list) > 0:
                    print('Transect # ', n, ' was split into ',
                          len(new_idxs_list), ' segs')
                    if len(new_idxs_list) == 1:
                        idxs = new_idxs_list[0]
                        self.sel_depth_and_gap_screen(tdata, idxs, n, nbins)
                    else:
                        for n_idxs in new_idxs_list:
                            self.sel_depth_and_gap_screen(tdata, n_idxs, n,
                                                          nbins)
                else:
                    print("lost transect #", n, " bc large gap in time/nav")
                    idxs[:] = False
                    self.sel_depth_and_gap_screen(tdata, idxs, n, nbins)
                    self.Lost_transects.append(n)
            else:
                print("no time/nav gap loc, straight into gap assessment")
                self.sel_depth_and_gap_screen(tdata, idxs, n, nbins)
        self.was_preprocessed = True

    def sel_depth_and_gap_screen(self, tdata, idxs, n, nbins):

        ddepth = tdata.depth

        didx = np.empty((len(dois)), dtype="O")
        adep = np.empty((len(dois)))
        nngaps = np.empty((len(dois)))
        ngap_max = np.empty((len(dois)))

        if np.count_nonzero(idxs) > 1:
            self.Dcover.append(tdata.dl[idxs].sum())
            for d in range(0, len(dois)):
                print("doing depth " + str(dois[d]))
                cdeps, cdidx = find_nearest_depths(ddepth, dois[d])
                nidxs = np.abs(cdeps - dois[d]) <= tdep_tol[d]
                nidxs = np.logical_and(idxs, nidxs)  # valid nav inds of dep
                if nidxs.any():
                    # self.Dcover.append(tdata.dl[idxs].sum())
                    if len(set(cdeps[nidxs])) > 1:
                        print('Depth of closest bin not constant along transect')
                        print('Checking if index of closest bin is constant:')
                        if len(set(cdidx[nidxs])) > 1:
                            print('depth changes bin index, # is %s and d is %s' %(n, d))
                            self.Bad_bin_depth.append((n, d))
                            # if index differ by 1, take most common bc target depth falls right in the middle
                            # if by more than 1, then need keep both
                            cdidx_max = max(cdidx[nidxs])
                            cdidx_min = min(cdidx[nidxs])
                            if cdidx_max - cdidx_min == 1:
                                cdidx_min_count = np.count_nonzero(cdidx == cdidx.min())
                                cdidx_max_count = np.count_nonzero(cdidx == cdidx.max())
                                if cdidx_min_count >= cdidx_max_count:
                                    cdidx[cdidx == cdidx.max()] = cdidx.min()
                                    didx[d] = get_adidx_bool(cdidx,
                                                             ddepth.shape,
                                                             nbins[d])
                                else:
                                    cdidx[cdidx == cdidx.min()] = cdidx.max()
                                    didx[d] = get_adidx_bool(cdidx,
                                                             ddepth.shape,
                                                             nbins[d])
                            else:
                                didx[d] = get_adidx_bool(cdidx, ddepth.shape,
                                                         nbins[d])
                        else:
                            print("it's ok, both depths are close to target and index is same")
                            didx[d] = get_adidx_bool(np.asarray(cdidx),
                                                     ddepth.shape, nbins[d])
                    else:
                        didx[d] = get_adidx_bool(np.asarray(cdidx),
                                                 ddepth.shape, nbins[d])
                    tdl = tdata.dl[nidxs]
                    if nbins[d] > 1:
                        usample = np.ma.reshape(tdata.u[didx[d]],
                                                (didx[d].shape[0],
                                                nbins[d]))[nidxs]
                        usample = usample.mean(axis=-1)
                        adep[d] = np.ma.reshape(ddepth[didx[d]],
                                                (didx[d].shape[0],
                                                nbins[d]))[nidxs].mean()
                    else:
                        usample = tdata.u[didx[d]][nidxs]
                        adep[d] = ddepth[didx[d]][nidxs][0]
                    nmdpt = np.ma.count_masked(usample, axis=0)
                    ntp = len(usample)
                    nngaps[d] = 100. * nmdpt / ntp
                    if nmdpt == 0:
                        ngap_max[d] = 0.
                    else:
                        g_slices = np.ma.clump_masked(usample)
                        gs = np.array([np.sum(tdl[p]) for p in g_slices])
                        if np.isnan(np.nanmax(gs)):
                            ngap_max[d] = tdl.mean()  # should be set to zero
                        else:
                            ngap_max[d] = np.nanmax(gs)
                    self.Nidxs.append(nidxs)
                else:
                    print("nidxs was empty: no valid data in target bin")
                    # self.Dcover.append(0e0)
                    didx[d] = find_nearest(ddepth[0], dois[d])[1]  # nan?
                    nngaps[d] = 100.
                    ngap_max[d] = 1e20
                    adep[d] = ddepth[0, didx[d]]  # should be nan
                    self.Nidxs.append(nidxs)
        else:
            print("indxs was empty: no valid nav data")
            self.Dcover.append(0e0)
            for d in range(0, len(dois)):
                didx[d] = find_nearest(ddepth[0], dois[d])[1]  # should be nan
                nngaps[d] = 100.
                ngap_max[d] = 1e20
                adep[d] = ddepth[0, didx[d]]  # should be nan
                self.Nidxs.append(idxs)
        self.Idxs.append(idxs)
        self.Didx.append(didx)
        self.Adep.append(adep)
        self.Ngap_max.append(ngap_max)
        self.Nngaps.append(nngaps)
        self.Wlut.append(n)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def find_nearest_depths(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value), axis=-1)
    near_array = np.asarray([array[n, m] for n, m in enumerate(idx)])
    return near_array, idx


def get_adidx_bool(adidx, ashape, nbins=1):
    adidx_bool = np.zeros(ashape, dtype=np.bool)
    for idx in np.unique(adidx):
        if nbins > 1:
            b = (nbins - 1) // 2
            if adidx_bool[:, idx-b:idx+b+1].shape[1] == nbins:
                adidx_bool[adidx == idx, idx-b:idx+b+1] = True
            elif adidx_bool[:, idx-b:idx+b+1].shape[1] < nbins:
                dinds = np.arange(0, ashape[1])
                if idx + b not in dinds:
                    adidx_bool[adidx == idx, idx-b-b:idx+b+1] = True
                elif idx - b not in dinds:
                    adidx_bool[adidx == idx, idx:idx+b+b+1] = True
        else:
            adidx_bool[adidx == idx, idx] = True
    return adidx_bool


def find_large_gaps_and_clip(flut, idxs, didx, adep, dcover, nngaps,
                             ngap_max):

    large_gaps = ngap_max > gs_tol

    nflut = flut[large_gaps]
    nidxs = idxs[large_gaps]
    ndidx = didx[large_gaps]
    nadep = adep[large_gaps]

    new_flut = []
    new_idxs = []
    new_didx = []
    new_adep = []
    new_ngaps = []
    new_gapmax = []
    new_dcover = []

    for n, alut in enumerate(nflut):
        usample = alut['seg_data'].u

        if ndidx[n].dtype == np.int64:
            usample = usample[:, ndidx[n]]
            g_slices = np.ma.clump_masked(usample[nidxs[n]])
        else:
            nbins = np.count_nonzero(ndidx[n], axis=-1)[0]
            if nbins > 1:
                usample = np.ma.reshape(usample[ndidx[n]],
                                        (ndidx[n].shape[0], nbins)).mean(axis=-1)
                g_slices = np.ma.clump_masked(usample[nidxs[n]])
            else:
                usample = usample[ndidx[n]]
                g_slices = np.ma.clump_masked(usample[nidxs[n]])

        gs = np.array([np.sum(alut['seg_data'].dl[nidxs[n]][p]) for p in g_slices])
        lgs = np.asarray(g_slices)[gs > gs_tol]
        beg = [sl.start for sl in lgs]
        end = [sl.stop for sl in lgs]
        beg.append(np.count_nonzero(nidxs[n]))
        if 0 in beg:
            beg.remove(0)
            new_slices = [slice(e, b) for e, b in zip(end, beg)]
        else:
            end = [0, ] + end
            new_slices = [slice(e, b) for e, b in zip(end, beg)]

        # new_idxs = []
        counter = 0
        for m, aslice in enumerate(new_slices):
            leng_sec = alut['seg_data'].dl[nidxs[n]][aslice].sum()
            if leng_sec >= slen:
                ridxs = nidxs[n].copy()
                ridxs[:] = False
                # ridxs[nidxs[n][aslice]] = True
                num_idxs = np.r_[slice(0, len(nidxs[n]))]
                ridxs[num_idxs[nidxs[n]][aslice]] = True
                ridxs = np.logical_and(ridxs, nidxs[n])
                new_idxs.append(ridxs)
                new_dcover.append(leng_sec)
                nmdpt = np.ma.count_masked(usample[ridxs])
                ntp = len(usample[ridxs])
                gaps = 100. * nmdpt / ntp
                if nmdpt == 0:
                    gap_max = 0.
                else:
                    g_slices = np.ma.clump_masked(usample[ridxs])
                    gs = np.array([np.sum(alut['seg_data'].dl[ridxs][p]) for p in g_slices])
                    if np.isnan(np.nanmax(gs)):
                        gap_max = alut['seg_data'].dl[ridxs].mean()
                    else:
                        gap_max = np.nanmax(gs)
                new_flut.append(alut)
                new_didx.append(ndidx[n])
                new_adep.append(nadep[n])
                new_ngaps.append(gaps)
                new_gapmax.append(gap_max)
                counter = counter + 1

        if counter > 0:
            print("recovered %s appropriate segments" %counter)

    if np.any(ngap_max <= gs_tol):
        combo_flut = np.concatenate((new_flut, flut[ngap_max <= gs_tol]))
        combo_idxs = np.concatenate((new_idxs, idxs[ngap_max <= gs_tol]))
        combo_didx = np.concatenate((new_didx, didx[ngap_max <= gs_tol]))
        combo_adep = np.concatenate((new_adep, adep[ngap_max <= gs_tol]))
        combo_dcover = np.concatenate((new_dcover, dcover[ngap_max <= gs_tol]))
        combo_ngaps = np.concatenate((new_ngaps, nngaps[ngap_max <= gs_tol]))
        combo_gapmax = np.concatenate((new_gapmax,
                                       ngap_max[ngap_max <= gs_tol]))
        return combo_flut, combo_idxs, combo_didx, combo_adep, combo_dcover, combo_ngaps, combo_gapmax
    else:
        return np.asarray(new_flut), np.asarray(new_idxs), np.asarray(new_didx), np.asarray(new_adep), np.asarray(new_dcover), np.asarray(new_ngaps), np.asarray(new_gapmax)


def calc_rms(b1, dday):
    dts = np.median(np.diff(dday)) * 24. * 3600.
    gps_err_vel = np.sqrt(2) * 5. / dts
    err_vel_rms = np.sqrt(np.mean(b1**2))
    return np.sqrt(err_vel_rms**2 + gps_err_vel**2)


def masked_interp(t, y):
    """
    gap filling with linear interolation for masked arrays
    loops over 2nd dim and interps masked indices of 1st dim
    """
    yn = y.data.copy().astype(t.dtype)

    for n in range(0, y.shape[1]):
        yn[y[:, n].mask, n] = np.interp(t[y[:, n].mask], t[~y[:, n].mask],
                                        y[:, n].compressed())

    return yn


def block_avg(y, x, xi, dl, fac=.7):
    yi = np.empty(xi.shape, dtype=np.float64)
    for p in range(0, len(xi)):
        idxs = np.logical_and(x <= xi[p] + dl * fac,
                              x >= xi[p] - dl * fac)
        yi[p] = y[idxs].mean(axis=0)
    if np.isnan(yi).any():
        yi = np.ma.masked_invalid(yi)
        yi = masked_interp(xi, yi[:, None])[:, 0]
    return yi


def get_tran_data(atran, nav_inds, dep_inds=None, nbins=1):
    lon = atran['seg_data'].lon[nav_inds]
    lat = atran['seg_data'].lat[nav_inds]
    dday = atran['seg_data'].dday[nav_inds]
    uship = atran['seg_data'].uship[nav_inds]
    vship = atran['seg_data'].vship[nav_inds]
    dl = atran['seg_data'].dl[nav_inds]
    svel = atran['seg_data'].svel[nav_inds]
    cogs = atran['seg_data'].cogs[nav_inds]
    headings = atran['seg_data'].headings[nav_inds]
    ymdhms = atran['seg_data'].ymdhms[nav_inds]

    if dep_inds is not None:
        if dep_inds.dtype == np.int64:
            u = atran['seg_data'].u[nav_inds, dep_inds]
            v = atran['seg_data'].v[nav_inds, dep_inds]
            dep = atran['seg_data'].depth[nav_inds, dep_inds]
            errs = atran['seg_data'].errs[nav_inds, dep_inds]
            pg = atran['seg_data'].pg[nav_inds, dep_inds]
            amp = atran['seg_data'].amp[nav_inds, dep_inds]
            amp1 = atran['seg_data'].amp1[nav_inds, dep_inds]
            amp2 = atran['seg_data'].amp2[nav_inds, dep_inds]
            amp3 = atran['seg_data'].amp3[nav_inds, dep_inds]
            amp4 = atran['seg_data'].amp4[nav_inds, dep_inds]
        elif dep_inds.dtype == np.bool:
            if nbins > 1:
                u = np.mean(np.ma.reshape(atran['seg_data'].u[dep_inds],
                                          (dep_inds.shape[0], nbins)),
                            axis=-1)[nav_inds]
                v = np.mean(np.ma.reshape(atran['seg_data'].v[dep_inds],
                                          (dep_inds.shape[0], nbins)),
                            axis=-1)[nav_inds]
                pg = np.mean(np.ma.reshape(atran['seg_data'].pg[dep_inds],
                                          (dep_inds.shape[0], nbins)),
                            axis=-1)[nav_inds]
                dep = np.mean(np.ma.reshape(atran['seg_data'].depth[dep_inds],
                                            (dep_inds.shape[0], nbins)),
                              axis=-1)[nav_inds]
                amp = np.mean(np.ma.reshape(atran['seg_data'].amp[dep_inds],
                                            (dep_inds.shape[0], nbins)),
                              axis=-1)[nav_inds]
                amp1 = np.mean(np.ma.reshape(atran['seg_data'].amp1[dep_inds],
                                             (dep_inds.shape[0], nbins)),
                               axis=-1)[nav_inds]
                amp2 = np.mean(np.ma.reshape(atran['seg_data'].amp2[dep_inds],
                                             (dep_inds.shape[0], nbins)),
                               axis=-1)[nav_inds]
                amp3 = np.mean(np.ma.reshape(atran['seg_data'].amp3[dep_inds],
                                             (dep_inds.shape[0], nbins)),
                               axis=-1)[nav_inds]
                amp4 = np.mean(np.ma.reshape(atran['seg_data'].amp4[dep_inds],
                                             (dep_inds.shape[0], nbins)),
                               axis=-1)[nav_inds]
                errs = np.mean(np.ma.reshape(atran['seg_data'].errs[dep_inds],
                                             (dep_inds.shape[0], nbins)),
                               axis=-1)[nav_inds]
            else:
                u = atran['seg_data'].u[dep_inds][nav_inds]
                v = atran['seg_data'].v[dep_inds][nav_inds]
                dep = atran['seg_data'].depth[dep_inds][nav_inds]
                errs = atran['seg_data'].errs[dep_inds][nav_inds]
                pg = atran['seg_data'].pg[dep_inds][nav_inds]
                amp = atran['seg_data'].amp[dep_inds][nav_inds]
                amp1 = atran['seg_data'].amp1[dep_inds][nav_inds]
                amp2 = atran['seg_data'].amp2[dep_inds][nav_inds]
                amp3 = atran['seg_data'].amp3[dep_inds][nav_inds]
                amp4 = atran['seg_data'].amp4[dep_inds][nav_inds]
        else:
            raise ValueError("Something went wrong with depth indices")
    else:
        u = atran['seg_data'].u[nav_inds]
        v = atran['seg_data'].v[nav_inds]
        dep = atran['seg_data'].depth[nav_inds]
        errs = atran['seg_data'].errs[nav_inds]
        pg = atran['seg_data'].pg[nav_inds]
        amp = atran['seg_data'].amp[nav_inds]
        amp1 = atran['seg_data'].amp1[nav_inds]
        amp2 = atran['seg_data'].amp2[nav_inds]
        amp3 = atran['seg_data'].amp3[nav_inds]
        amp4 = atran['seg_data'].amp4[nav_inds]

    return lon, lat, u, v, errs, uship, vship, dday, dep, pg, amp, amp1, amp2, amp3, amp4


def to_regular_transect(atransect, nav_inds, dep_inds, nbins, dameth, dlm):
    # read just the segment (per depth) data:
    data_vars = get_tran_data(atransect, nav_inds, dep_inds, nbins)

    # assign arrays:
    lon, lat = data_vars[0], data_vars[1]
    u, v, errs = data_vars[2], data_vars[3], data_vars[4]
    uship, vship = data_vars[5], data_vars[6]
    dday, dep = data_vars[7], data_vars[8]

    # deal with gaps via mask interpolation:
    ui = masked_interp(dday, u[:, None])[:, 0]
    vi = masked_interp(dday, v[:, None])[:, 0]
    erri = masked_interp(dday, errs[:, None])[:, 0]

    lati = masked_interp(dday, lat[:, None])[:, 0]
    loni = masked_interp(dday, lon[:, None])[:, 0]

    if np.isscalar(uship.mask) is False:
        ushipi = masked_interp(dday, uship[:, None])[:, 0]
        vshipi = masked_interp(dday, vship[:, None])[:, 0]
    else:
        ushipi = uship
        vshipi = vship

    # calculate x, distance along transect, in km:
    x = np.zeros(lati.shape)
    dx, dy = diffxy_from_lonlat(loni, lati, pad=False)
    dl = 1e-3 * np.sqrt(dx**2 + dy**2)
    x[1:] = np.cumsum(dl)

    # put on a regular grid or block avg to coarser but filled low res:
    edist = x[-1]  # either this or the estimated g_dist?
    xi = np.arange(0, np.floor(edist), dlm)
    if dameth == 'linear' or dameth == 'cubic':
        # nav data uses linear interp:
        fla = interp1d(x, lati, kind='linear', axis=0)
        flo = interp1d(x, loni, kind='linear', axis=0)
        ftm = interp1d(x, dday, kind='linear', axis=0)
        if dameth == 'cubic' and np.any(x[1:] == x[:-1]):
            __, cidx, counts = np.unique(x, return_counts=True,
                                         return_index=True)
            x = x[cidx]
            ui = ui[cidx]
            vi = vi[cidx]
            erri = erri[cidx]
            ushipi = ushipi[cidx]
            vshipi = vshipi[cidx]
        fu = interp1d(x, ui, kind=dameth, axis=0)
        fv = interp1d(x, vi, kind=dameth, axis=0)
        fe = interp1d(x, erri, kind=dameth, axis=0)
        U, V, E = fu(xi), fv(xi), fe(xi)

        fus = interp1d(x, ushipi, kind=dameth, axis=0)
        fvs = interp1d(x, vshipi, kind=dameth, axis=0)
        lats, lons, times = fla(xi), flo(xi), ftm(xi)
        uships, vships = fus(xi), fvs(xi)
    elif dameth >= .5 and dameth <= 2.:
        U = block_avg(ui, x, xi, dlm, dameth)
        V = block_avg(vi, x, xi, dlm, dameth)
        E = block_avg(erri, x, xi, dlm, dameth)
        lats = block_avg(lati, x, xi, dlm, dameth)
        lons = block_avg(loni, x, xi, dlm, dameth)
        times = block_avg(dday, x, xi, dlm, dameth)
        uships = block_avg(ushipi, x, xi, dlm, dameth)
        vships = block_avg(vshipi, x, xi, dlm, dameth)
        # need to assert there are no nan's and if present, are dealt with
    else:
        raise ValueError('Wrong regrid method selection')

    reg_transect = Bunch()

    reg_transect.u = U
    reg_transect.v = V
    reg_transect.err_vel = E
    reg_transect.x = xi
    reg_transect.lats = lats
    reg_transect.lons = lons
    reg_transect.ddays = times
    reg_transect.uship = uships
    reg_transect.vship = vships
    reg_transect.depth = dep.mean()
    reg_transect.cruise_id = atransect['cruise_id']
    reg_transect.inst_id = atransect['inst_id']
    reg_transect.vessel_id = atransect['vessel_id']
    reg_transect.sac_id = atransect['sac_id']
    reg_transect.year = atransect['seg_data'].ymdhms[nav_inds][0, 0]
    reg_transect.month = atransect['seg_data'].ymdhms[nav_inds][0, 1]
    reg_transect.g_dist = atransect['g_dist']
    reg_transect.dcover = atransect['dcover']
    reg_transect.lat_min = atransect['lat_min']
    reg_transect.lon_min = atransect['lon_min']
    reg_transect.lat_max = atransect['lat_max']
    reg_transect.lat_max = atransect['lon_max']
    reg_transect.duration = atransect['seg_days']
    reg_transect.orientation = atransect['trans_orient']
    reg_transect.avg_spd = atransect['avg_spd']
    reg_transect.dlm = dlm
    reg_transect.start_date = np.array(datetime(*atransect['seg_data'].ymdhms[nav_inds][0]), dtype='datetime64[s]')

    return reg_transect


def smooth1d(y, winsize, ax=0, wintype='blackman'):
    """
    smooths NON-masked arrays with blackman or boxcar window.
    """
    if wintype not in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'boxcar', 'hanning', 'hamming', 'bartlett', 'blackman'")

    y = np.asanyarray(y)
    win = eval('windows.' + wintype + '(winsize)')
    yn = filters.convolve1d(y, win/win.sum(), axis=ax)
    return yn


def classify_orientation(number):
    if number < 20.:
        return 'zonal'
    elif number > 70.:
        return 'meridional'
    else:
        return 'slanted'


def pad_w_mask(y, Npad):
    """
    pads the beguining and end of y along the 1st axis with 9999
    then mask them
    """
    if Npad % 2 == 0:
        pads = np.ones((Npad // 2,)) * 9999e0
        yn = np.concatenate((pads, y, pads), axis=0)
    else:
        padl = np.ones((Npad // 2 + 1,)) * 9999e0
        padr = np.ones((Npad // 2,)) * 9999e0
        yn = np.concatenate((padl, y, padr), axis=0)
    return np.ma.masked_values(yn, 9999e0)


def to_overlaping_segment(rt, overlap):

    xi = rt.x.copy()
    u = rt.u.copy()
    v = rt.v.copy()
    err_vel = rt.err_vel.copy()
    uships = rt.uship.copy()
    vships = rt.vship.copy()
    dlm = rt.dlm
    lats = rt.lats.copy()
    lons = rt.lons.copy()
    times = rt.ddays.copy()

    # find Nr long uniform, with x% overlap segments:
    n500 = int(Nr / dlm)
    noverlap = int(overlap * n500)
    step = n500 - noverlap
    exs = len(xi) % n500
    topad = int(n500 - len(xi) % n500)
    if exs != 0 and exs <= topad:
        npts = len(xi[exs//2:-exs//2])
    elif exs != 0 and exs > topad:
        npts = topad + len(xi)
        u = pad_w_mask(u, topad)
        v = pad_w_mask(v, topad)
        err_vel = pad_w_mask(err_vel, topad)
        xpad = np.arange(xi[-1] + dlm, dlm*topad + xi[-1] + dlm, dlm)
        xi = np.concatenate((xi, xpad), axis=0)
        # the following vars can be used later to identify segs padded
        lats = pad_w_mask(lats, topad)
        lons = pad_w_mask(lons, topad)
        times = pad_w_mask(times, topad)
        uships, vships = pad_w_mask(uships, topad), pad_w_mask(vships, topad)
        if npts % 2 != 0:
            u, v, err_vel = u[:-1], v[:-1], err_vel[:-1]
            lats, lons, xi = lats[:-1], lons[:-1], xi[:-1]
            uships, vships, times = uships[:-1], vships[:-1], times[:-1]
    else:
        npts = len(xi)

    # segmentation:
    ind = np.arange(0, npts - n500 + 1, step)
    xis = np.ma.empty((n500, len(ind)))  # ends up w/ unit mask if ind unmasked
    lat_seg = np.ma.empty((n500, len(ind)))
    lon_seg = np.ma.empty((n500, len(ind)))
    time_seg = np.ma.empty((n500, len(ind)))
    u_seg = np.ma.empty((n500, len(ind)))
    v_seg = np.ma.empty((n500, len(ind)))
    e_seg = np.ma.empty((n500, len(ind)))
    us_seg = np.ma.empty((n500, len(ind)))
    vs_seg = np.ma.empty((n500, len(ind)))
    for m in range(0, len(ind)):
        if exs != 0 and exs <= topad:
            xis[:, m] = xi[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            lat_seg[:, m] = lats[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            lon_seg[:, m] = lons[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            time_seg[:, m] = times[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            u_seg[:, m] = u[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            v_seg[:, m] = v[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            e_seg[:, m] = err_vel[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            us_seg[:, m] = uships[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            vs_seg[:, m] = vships[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
        else:
            xis[:, m] = xi[ind[m]:ind[m]+n500]
            lat_seg[:, m] = lats[ind[m]:ind[m]+n500]
            lon_seg[:, m] = lons[ind[m]:ind[m]+n500]
            time_seg[:, m] = times[ind[m]:ind[m]+n500]
            xis[:, m] = xi[ind[m]:ind[m]+n500]
            u_seg[:, m] = u[ind[m]:ind[m]+n500]
            v_seg[:, m] = v[ind[m]:ind[m]+n500]
            e_seg[:, m] = err_vel[ind[m]:ind[m]+n500]
            us_seg[:, m] = uships[ind[m]:ind[m]+n500]
            vs_seg[:, m] = vships[ind[m]:ind[m]+n500]

    laidx, loidx = lat_seg < 9999., lon_seg < 9999.
    dx, dy = np.ma.empty((n500, len(ind))), np.ma.empty((n500, len(ind)))
    g_dists = np.empty((len(ind),))
    for m in range(0, len(ind)):
        dx[:, m], dy[:, m] = diffxy_from_lonlat(lon_seg[:, m],
                                                lat_seg[:, m])
        g_dists[m] = great_circle_distance(lon_seg[loidx[:, m], m][0],
                                           lat_seg[laidx[:, m], m][0],
                                           lon_seg[loidx[:, m], m][-1],
                                           lat_seg[laidx[:, m], m][-1]) / 1e3
    dx_int, dy_int = dx.sum(axis=0), dy.sum(axis=0)  # their ratio ~ mean(thetas)
    orient = np.rad2deg(np.arctan(np.abs(dy_int / dx_int)))
    d_skew = np.abs(g_dists - Nr) / Nr  # carefully interpret when padding is active
    dcover_seg = 1e-3 * np.sqrt(dx_int**2 + dy_int**2)
    d_skew_r = np.abs(g_dists - dcover_seg) / g_dists

    # rotate vel, either use median uship/vship or low pass (for long segs)
    thetas = np.ma.angle(us_seg + 1j*vs_seg)
    thetam = np.ma.angle(np.ma.median(us_seg, axis=0) +
                         1j*np.ma.median(vs_seg, axis=0))

    # Using smoothed time varying angle from ship speed:
    # tp = round_up_to_even(24 * 3600 / sg['dt']) + 1  # 215 ~18hr
    if exs != 0 and exs > topad:
        uship_smo = np.empty((n500, len(ind)))
        vship_smo = np.empty((n500, len(ind)))
        for m in range(0, len(ind)):
            uship_smo[~laidx.mask[:, m], m] = smooth1d(us_seg[:, m].compressed(), int(tp))
            vship_smo[~laidx.mask[:, m], m] = smooth1d(vs_seg[:, m].compressed(), int(tp))
        uship_smo = np.ma.masked_where(lat_seg.mask, uship_smo)
        vship_smo = np.ma.masked_where(lat_seg.mask, vship_smo)
    else:
        uship_smo = smooth1d(us_seg, int(tp))
        vship_smo = smooth1d(vs_seg, int(tp))

    # 2) get the smoothed angle:
    thetas_smo = np.ma.angle(uship_smo + 1j*vship_smo)

    # 3) rotate:
    uv = u_seg + 1j*v_seg
    at_seg_m = (uv * np.ma.exp(-1j * thetam[np.newaxis])).real
    xt_seg_m = (uv * np.ma.exp(-1j * thetam[np.newaxis])).imag
    at_seg_s = (uv * np.ma.exp(-1j * thetas_smo)).real
    xt_seg_s = (uv * np.ma.exp(-1j * thetas_smo)).imag

    # Now store in (rewrite) the Bunch:
    sg = Bunch()
    sg.u = u_seg
    sg.v = v_seg
    sg.err_vel = e_seg
    sg.atm = at_seg_m
    sg.xtm = xt_seg_m
    sg.ats = at_seg_s
    sg.xts = xt_seg_s
    sg.xis = xis
    sg.lats = lat_seg
    sg.lons = lon_seg
    sg.times = time_seg
    sg.lat_center = lat_seg.mean(axis=0)
    sg.lon_center = lon_seg.mean(axis=0)
    sg.lat_min = lat_seg.min(axis=0)
    sg.lon_min = lon_seg.min(axis=0)
    sg.lat_max = lat_seg.max(axis=0)
    sg.lon_max = lon_seg.max(axis=0)
    sg.cruise_id = rt.cruise_id.repeat(len(ind))
    sg.inst_id = rt.inst_id.repeat(len(ind))
    sg.vessel_id = rt.vessel_id.repeat(len(ind))
    sg.sac_id = rt.sac_id.repeat(len(ind))
    sg.thetam = thetam
    sg.thetas_smo = thetas_smo
    sg.oangle = orient
    sg.orientation = np.array([classify_orientation(k) for k in orient])
    sg.dskew = d_skew
    sg.dskew_r = d_skew_r
    sg.year = rt.year.repeat(len(ind))
    sg.month = rt.month.repeat(len(ind))
    sg.depth = rt.depth.repeat(len(ind))
    sg.start_date = rt.start_date.repeat(len(ind))
    return sg


def to_stack(proc_segs_list):
    # this starts to stack the segments:
    n500 = int(Nr / dlm)
    ns500 = np.empty((len(proc_segs_list),), dtype=np.int)
    for n, psg in enumerate(proc_segs_list):
        ns500[n] = psg.xis.shape[-1]
    tns500 = ns500.sum()

    # Need unique identifiers to easily select transects (and match them to map)!
    # create the array stacks
    ats_stack = np.ma.empty((n500, tns500))
    xts_stack = np.ma.empty((n500, tns500))
    atm_stack = np.ma.empty((n500, tns500))
    xtm_stack = np.ma.empty((n500, tns500))
    xi_stack = np.ma.empty((n500, tns500))
    u_stack = np.ma.empty((n500, tns500))
    v_stack = np.ma.empty((n500, tns500))
    e_stack = np.ma.empty((n500, tns500))
    lat_stack = np.ma.empty((n500, tns500))
    lon_stack = np.ma.empty((n500, tns500))
    time_stack = np.ma.empty((n500, tns500))

    # Noise floor estimates:
    # dts = 300.  # 5 min ensemble
    # gps_err_vel = np.sqrt(2) * 5. / dts
    e_rms = np.empty((tns500,))

    # keep track of where, when, how segment is:
    lat_center = np.empty((tns500,))
    lon_center = np.empty((tns500,))
    lat_min = np.empty((tns500,))
    lon_min = np.empty((tns500,))
    lat_max = np.empty((tns500,))
    lon_max = np.empty((tns500,))
    orientations = np.empty((tns500,), dtype='<U10')
    track_angle = np.empty((tns500,))
    thetams = np.empty((tns500,))
    theta_smos = np.empty((n500, tns500,))
    dskews = np.empty((tns500,))
    dskews_r = np.empty((tns500,))
    years = np.empty((tns500,), dtype=np.int32)
    months = np.empty((tns500,), dtype=np.int32)
    start_dates = np.empty((tns500,), dtype='datetime64[s]')
    depths = np.empty((tns500))
    sonars = np.empty((tns500), dtype='<U19')
    cruises = np.empty((tns500), dtype='<U19')
    vessels = np.empty((tns500), dtype='<U19')
    sacids = np.empty((tns500), dtype='<U19')

    st = 0
    acheck = []
    identifier = np.empty((tns500,), dtype=np.int)
    for n, psg in enumerate(proc_segs_list):
        nseg = psg.xis.shape[-1] + st
        acheck.append(nseg)
        atm_stack[:, st:nseg] = psg.atm
        xtm_stack[:, st:nseg] = psg.xtm
        ats_stack[:, st:nseg] = psg.ats
        xts_stack[:, st:nseg] = psg.xts
        xi_stack[:, st:nseg] = psg.xis
        u_stack[:, st:nseg] = psg.u
        v_stack[:, st:nseg] = psg.v
        e_stack[:, st:nseg] = psg.err_vel
        lon_stack[:, st:nseg] = psg.lons
        lat_stack[:, st:nseg] = psg.lats
        time_stack[:, st:nseg] = psg.times

        e_rms[st:nseg] = calc_rms(np.ma.masked_greater(psg.err_vel, 9999e0),
                                  psg.times)
        lat_center[st:nseg] = psg.lat_center
        lon_center[st:nseg] = psg.lon_center
        lat_min[st:nseg] = psg.lat_min
        lon_min[st:nseg] = psg.lon_min
        lat_max[st:nseg] = psg.lat_max
        lon_max[st:nseg] = psg.lon_max
        orientations[st:nseg] = psg.orientation
        track_angle[st:nseg] = psg.oangle
        thetams[st:nseg] = psg.thetam
        theta_smos[:, st:nseg] = psg.thetas_smo
        dskews[st:nseg] = psg.dskew
        dskews_r[st:nseg] = psg.dskew_r
        years[st:nseg] = psg.year
        start_dates[st:nseg] = psg.start_date
        months[st:nseg] = psg.month
        depths[st:nseg] = psg.depth
        sonars[st:nseg] = psg.inst_id
        cruises[st:nseg] = psg.cruise_id
        vessels[st:nseg] = psg.vessel_id
        sacids[st:nseg] = psg.sac_id
        identifier[st:nseg] = n
        st = nseg

    # save in npz:
    dtyp = np.dtype([('sonar', sonars.dtype), ('cruise', cruises.dtype),
                     ('vessel', vessels.dtype), ('sac_id', sacids.dtype),
                     ('year', years.dtype),
                     ('month', months.dtype), ('depth', depths.dtype),
                     ('lon_center', lon_center.dtype),
                     ('lat_center', lat_center.dtype),
                     ('lon_min', lon_min.dtype), ('lat_min', lat_min.dtype),
                     ('lon_max', lon_max.dtype), ('lat_max', lat_max.dtype),
                     ('dskew', dskews.dtype), ('dskew_r', dskews_r.dtype),
                     ('thetam', thetams.dtype),
                     ('theta_smos', "O"),
                     ('start_date', start_dates.dtype),
                     ('orientation', orientations.dtype),
                     ('tangle', track_angle.dtype), ('err_rms', e_rms.dtype),
                     ('lons', "O"), ('lats', "O"), ('xi', "O"), ('err', "O"),
                     ('u', "O"), ('v', "O"), ('atm', "O"), ('xtm', "O"),
                     ('ats', "O"), ('xts', "O"), ('times', "O")])
    dastack = np.array(list(zip(*(sonars, cruises, vessels, sacids,
                                years, months, depths, lon_center, lat_center,
                                lon_min, lat_min, lon_max, lat_max,
                                dskews, dskews_r, thetams, theta_smos.T,
                                start_dates, orientations, track_angle, e_rms,
                                lon_stack.T, lat_stack.T, xi_stack.T, e_stack.T,
                                u_stack.T, v_stack.T, atm_stack.T, xtm_stack.T,
                                ats_stack.T, xts_stack.T, time_stack.T))),
                       dtype=dtyp)

    return dastack, acheck


def main():
    """
    Runs the loop for desired target depths and writes final stack data file

    TODO: improve this main and the initialization, including the loading stage
    to use the netcdf functions and files
    """
    transects_dbs = np.load('./data/proc/transects_dbs_in_dir_codas_dbs.npz',
                            allow_pickle=True,)['seg_dbase']
    alltran = TransectsDB(transecs_dbs)
    Wlut = np.asarray(alltran.Wlut)
    Dcover = np.asarray(alltran.Dcover)
    Didx = np.asarray(alltran.Didx)
    Adep = np.asarray(alltran.Adep)
    Nngaps = np.asarray(alltran.Nngaps)
    Ngap_max = np.asarray(alltran.Ngap_max)
    Nidxs = np.reshape(np.asarray(alltran.Nidxs), Didx.shape)
    for d in range(0, len(dois)):
        idxs = Nidxs[:, d]
        didx = Didx[:, d]
        adep = Adep[:, d]
        nngaps = Nngaps[:, d]
        ngap_max = Ngap_max[:, d]
        valid_ones = np.logical_and(Dcover > 0e0, ngap_max < 1e20)
        wlut = Wlut[valid_ones]
        outputs = find_large_gaps_and_clip(alltran.transects[wlut], idxs[valid_ones],
                                           didx[valid_ones], adep[valid_ones],
                                           Dcover[valid_ones], nngaps[valid_ones],
                                           ngap_max[valid_ones])

        flut, idxs = outputs[0], outputs[1]
        didx, adep = outputs[2], outputs[3]
        dcover, ngaps, gapmax = outputs[4], outputs[5], outputs[6]

        # pass again on find_large_gaps_and_clip:
        gs_tol = 15.  # [15., 10., 5.]
        chosen_ones = np.logical_and(ngaps > gp_tol, dcover >= slen)
        # chosen_ones = np.logical_and(gapmax > gs_tol, chosen_ones)
        if np.any(chosen_ones):
            outputs = find_large_gaps_and_clip(flut[chosen_ones],
                                               idxs[chosen_ones],
                                               didx[chosen_ones],
                                               adep[chosen_ones],
                                               dcover[chosen_ones],
                                               ngaps[chosen_ones],
                                               gapmax[chosen_ones])

            if len(outputs[0]) > 0:
                new_goods = np.logical_and(outputs[-3] >= slen,
                                           np.round(outputs[-2]) <= gp_tol)
                flut = np.concatenate((flut[~chosen_ones], outputs[0][new_goods]))
                if outputs[1].ndim > 1:
                    dummy1 = np.empty((outputs[1][new_goods].shape[0],), dtype='O')
                    for n, vals in enumerate(outputs[1][new_goods]):
                        dummy1[n] = vals
                    idxs = np.concatenate((idxs[~chosen_ones], dummy1))
                    dummy2 = np.empty((outputs[2][new_goods].shape[0],), dtype='O')
                    for n, vals in enumerate(outputs[2][new_goods]):
                        dummy2[n] = vals
                    # dummy2[0] = outputs[2][0, ...]
                    didx = np.concatenate((didx[~chosen_ones], dummy2))
                else:
                    idxs = np.concatenate((idxs[~chosen_ones], outputs[1][new_goods]))
                    didx = np.concatenate((didx[~chosen_ones], outputs[2][new_goods]))
                adep = np.concatenate((adep[~chosen_ones], outputs[3][new_goods]))
                dcover = np.concatenate((dcover[~chosen_ones], outputs[4][new_goods]))
                ngaps = np.concatenate((ngaps[~chosen_ones], outputs[5][new_goods]))
                gapmax = np.concatenate((gapmax[~chosen_ones], outputs[6][new_goods]))


        # and again on the leftovers:
        gs_tol = 10.  # [15., 10., 5.]
        chosen_ones = np.logical_and(outputs[5][~new_goods] > gp_tol,
                                     outputs[4][~new_goods] >= slen)
        if np.any(chosen_ones):
            outputs = find_large_gaps_and_clip(outputs[0][~new_goods][chosen_ones],
                                               outputs[1][~new_goods][chosen_ones],
                                               outputs[2][~new_goods][chosen_ones],
                                               outputs[3][~new_goods][chosen_ones],
                                               outputs[4][~new_goods][chosen_ones],
                                               outputs[5][~new_goods][chosen_ones],
                                               outputs[6][~new_goods][chosen_ones])

            if len(outputs[0]) > 0:
                new_goods = np.logical_and(outputs[-3] >= slen,
                                           np.floor(outputs[-2]) <= gp_tol)
                flut = np.concatenate((flut, outputs[0][new_goods]))
                if outputs[1].ndim > 1:
                    dummy1 = np.empty((outputs[1][new_goods].shape[0],), dtype='O')
                    for n, vals in enumerate(outputs[1][new_goods]):
                        dummy1[n] = vals
                    idxs = np.concatenate((idxs[~chosen_ones], dummy1))
                    dummy2 = np.empty((outputs[2][new_goods].shape[0],), dtype='O')
                    for n, vals in enumerate(outputs[2][new_goods]):
                        dummy2[n] = vals
                    # dummy2[0] = outputs[2][0, ...]
                    didx = np.concatenate((didx, dummy2))
                else:
                    idxs = np.concatenate((idxs, outputs[1][new_goods]))
                    didx = np.concatenate((didx, outputs[2][new_goods]))
                adep = np.concatenate((adep, outputs[3][new_goods]))
                dcover = np.concatenate((dcover, outputs[4][new_goods]))
                ngaps = np.concatenate((ngaps, outputs[5][new_goods]))
                gapmax = np.concatenate((gapmax, outputs[6][new_goods]))


        # subset again the list based on the update info:
        gs_tol = 20.
        sidx = np.logical_and(dcover >= slen, ngaps <= gp_tol)
        sidx = np.logical_and(sidx, gapmax <= gs_tol)
        proc_segs_list = []
        for atransect, nav_inds, dep_inds in zip(flut[sidx], idxs[sidx], didx[sidx]):
            RT = to_regular_transect(atransect, nav_inds, dep_inds, nbas[d], dameth, dlm)
            SG = to_overlaping_segment(RT, .5)
            proc_segs_list.append(SG)
        dastack, acheck = to_stack(proc_segs_list)
        out_stack_filename = out_stack_filename_hdr + str(int(dois[d])) + '.npz'
        np.savez_compressed(out_dir + out_stack_filename, data_stack=dastack)

if __name__ == '__main__':
    main()
