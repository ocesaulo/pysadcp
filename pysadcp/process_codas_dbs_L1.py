#!/usr/bin/env python

'''
Workhorse script to turn several CODAS dbs (cruises) into a data set of
transects and time series.

Input: (path to) CODAS dbs list
Output: L1 processed ADCP data and meta-data into transects or point timeseries
'''


import numpy as np
import os
import fnmatch
import argparse
from pycurrents.codas import get_profiles
from pycurrents.data.navcalc import lonlat_inside_km_radius
from pycurrents.data.navcalc import (great_circle_distance, diffxy_from_lonlat)
from pycurrents.system import Bunch
from pycurrents.file import npzfile
from scipy.stats import mode as Mode
from pysadcp import read_meta_from_bft
from pysadcp import read_meta_from_dbinfo
from pysadcp import find_most_common_position


class RunParams:
    def __init__(self, dbs_list, out_dir, out_fname=None, mas=3., tst=2.,
                 mtl=50., lts=6., rts=.2):
        self.mas = mas  # minimum average ship speed during segment in m/s
        self.tst = tst  # tolarated stop time in hrs (longer will be split)
        self.mtl = mtl  # minimum segment/transect length in km
        self.lts = lts  # minimum length of a point time series in hrs
        self.rts = rts  # max radious of a point time series in hrs
        self.dbslist = load_dbs_list(dbs_list)
        print("\nThere are", len(self.dbslist), " dbs to process\n")
        if out_fname is None:
            if isinstance(dbs_list, (str, bytes)):
                if dbs_list[-4:] == '.npz':
                    out_fname = os.path.split(dbs_list)[-1][:-4]
                elif dbs_list[-1] == '/':
                    pathend = os.path.normpath(dbs_list).split(os.sep)
                    out_fname = 'dbs_in_dir_' + pathend[-1]
                elif dbs_list[-1] == '*':
                    pathend = os.path.normpath(dbs_list).split(os.sep)
                    out_fname = 'dbs_in_dir_' + pathend[-2]
                else:
                    out_fname = 'db_' + os.path.split(dbs_list)[-1]
            else:
                out_fname = 'unknown_dbs'
        self.output_files_ids = prep_out_dir(out_dir, out_fname)


def load_dbs_list(dbs_list):
    '''
        Reads a string input with path(s) to CODAS databases or list/arrays of
        paths. The string can be a path to a directory containing many
        databses, in which case it must end w/ either * or /.
        Returns a list with said path(s).
    '''
    if isinstance(dbs_list, (str, bytes)):
        if dbs_list[-4:] == '.npz':
            return np.load(dbs_list, allow_pickle=True)['dbslist'].tolist()
        elif dbs_list[-1] == '*' or dbs_list[-1] == '/':
            print("A directory was provided, will walk it to form list of dbs")
            dbslist = []
            parent_dir = os.path.split(dbs_list)[0]
            for root, dirnames, filenames in os.walk(parent_dir):
                for filename in fnmatch.filter(filenames, '*dir.blk'):
                    dbslist.append(os.path.join(root, filename[:-7]))
            return dbslist
        else:
            print('Interpreting string input as a path to a single db')
            return list(dbs_list)
    elif isinstance(dbs_list, np.array):
        return dbs_list.tolist()
    elif isinstance(dbs_list, (list, tuple)):
        return dbs_list
    else:
        error_message = ("\nPath to CODAS database(s) must be a " +
                         "single str or a list of strs or a special npz file")
        raise TypeError(error_message)


def prep_out_dir(out_dir, out_fname):
    toutfilename = 'transects_' + out_fname
    poutfilename = 'point_timeseries_' + out_fname

    if out_dir[-1] == '/':
        output_file_id = out_dir + toutfilename + '.npz'
        output2_file_id = out_dir + poutfilename + '.npz'
    else:
        output_file_id = out_dir + '/' + toutfilename + '.npz'
        output2_file_id = out_dir + '/' + poutfilename + '.npz'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(output_file_id):
        print('Transect output file already exists, this will overwrite it!')
    if os.path.exists(output2_file_id):
        print('Timeseries output file already exists, this will overwrite it!')

    print("Output directory and file for transects is " + output_file_id)
    print("Output directory and file for timeseries is " + output2_file_id)
    return output_file_id, output2_file_id


def save_lut(output_file_id, lut):
    '''save the look-up table and segment database'''
    # (should PICKLE? if class/masked)
    try:
        npzfile.savez(output_file_id, seg_dbase=lut)
    except:  # can't remember the error that requires use of generic save
        np.save(output_file_id, lut)
    print("Database saved to " + output_file_id)
    return


def read_codas_db_wrap(db):
    try:
        data = get_profiles(db, diagnostics=True)  # get all data
        return data
    except ValueError as e:
        if "has 2 block directories" in str(e):
            print('\nThere was a problem reading this db (2 block dirs), skipping')
            print('This db should not be in the list!')
            return None
        elif 'has no block directory' in str(e):
            print('\nNo codas blk data in path of db, skipping')
            print('This db should not be in the list!')
            return None
        else:
            print('\nCould not read this db path for unknown reason, skipping')
            return None


def read_metadata_wrap(data, db):
    bftfile = db + '.bft'
    dbinfo_file = os.path.split(os.path.split(db)[0])[0] + '/dbinfo.txt'
    if os.path.exists(bftfile):
        cruise_id, instru_id, vessel_id, sac_id = read_meta_from_bft(bftfile)
    elif os.path.exists(dbinfo_file):
        cruise_id, instru_id, vessel_id = read_meta_from_dbinfo(dbinfo_file)
        sac_id = 'None; UH repo?'
    else:
        print('No meta data file found!')
        cruise_id = 'unknown_no_metafile'
        instru_id = 'unknown_no_metafile'
        vessel_id = 'unknown_no_metafile'
        sac_id = 'unknown_no_metafile'

    return cruise_id, instru_id, vessel_id, sac_id


def find_stations_restarts(data, mas, tst, lts, rts):
    svel = data.spd  # ship speed timeseries, need to ensure nav masks are same
    gids = svel > mas

    dtp = np.diff(data.dday[gids])  # time intervals when moving

    breaks = np.where(dtp > tst / 24.)[0]  # indices of start/end times of stop
    dts = round(np.ma.median(dtp) * 3600. * 24)

    restarts = np.empty_like(breaks)
    for n, idx in enumerate(breaks):
        if len(svel[gids][idx + 1:].compressed()) != 0:
            restarts[n] = np.where(svel[gids] ==
                                   svel[gids][idx + 1:].compressed()[0])[0][0]
        else:
            restarts[n] = len(svel[gids]) - 1
    breaks = np.sort(np.concatenate((breaks, restarts)))

    print("\nnumber of stops is ", len(breaks))

    if len(svel[gids]) == 0 or np.all(data.spd.mask):
        print("No transects to be found here")
        g_dists = np.array([0])
        c = None
        dts = round(np.ma.median(np.diff(data.dday)) * 3600. * 24)
    elif len(breaks) != 0:
        # time and position of break points:
        bdday = data.dday[gids][breaks]
        blon = data.lon[gids][breaks]
        blat = data.lat[gids][breaks]

        # first working index:
        if np.ma.is_masked(data.lon):
            ind0 = np.where(~data.lon.mask)[0][0]
        else:
            ind0 = 0

        # get geo distance between the breakpoints:
        g_dists = np.ma.hstack((1e-3 * great_circle_distance(data.lon[ind0],
                                data.lat[ind0], blon[0], blat[0]),
                                1e-3 * great_circle_distance(blon[:-1],
                                blat[:-1], blon[1:], blat[1:]),
                                1e-3 * great_circle_distance(blon[-1],
                                blat[-1], data.lon[-1], data.lat[-1])))

        # get the indices of the original data where the break starts and ends:
        c = np.empty((g_dists.size + 1,), dtype=int)
        c[0], c[-1] = ind0, len(svel)-1
        for n in range(0, len(bdday)):
            c[n+1] = np.where(data.dday == bdday[n])[0][0]  # ought to add + 1?
    else:
        tmk = np.ma.masked_where(svel[gids].mask, data.dday[gids])
        bd = tmk.compressed()[0] - data.dday[0]
        be = data.dday[-1] - tmk.compressed()[-1]
        if bd > 0 and be > 0:
            # bslice = slice(np.where(data.dday == tmk.compressed()[0])[0][0],
            #                np.where(data.dday == tmk.compressed()[-1])[0][0])
            bslice = np.where(np.logical_or(data.dday == tmk.compressed()[0],
                                            data.dday == tmk.compressed()[-1]))[0]
            blat = data.lat[bslice]
            blon = data.lon[bslice]
            bdday = data.dday[bslice]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon[0], blat[0]),
                                1e-3*great_circle_distance(blon[:-1],
                                                           blat[:-1], blon[1:],
                                                           blat[1:]),
                                1e-3 * great_circle_distance(blon[-1],
                                                             blat[-1],
                                                             data.lon[-1],
                                                             data.lat[-1])))

            # get the indices of the original data
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[-1] = 0, len(svel)-1
            for n in range(0, len(bdday)):
                c[n+1] = np.where(data.dday == bdday[n])[0][0]
        elif bd > 0 and be == 0:
            b1 = np.where(data.dday == tmk.compressed()[0])[0][0]
            blat = data.lat[b1]
            blon = data.lon[b1]
            bdday = data.dday[b1]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon, blat),
                                1e-3 * great_circle_distance(blon, blat,
                                                             data.lon[-1],
                                                             data.lat[-1])))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[1], c[-1] = 0, b1, len(svel) - 1
        elif bd == 0 and be > 0:
            b1 = np.where(data.dday == tmk.compressed()[-1])[0][0]
            blat = data.lat[b1]
            blon = data.lon[b1]
            bdday = data.dday[b1]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon, blat),
                                1e-3 * great_circle_distance(blon, blat,
                                                             data.lon[-1],
                                                             data.lat[-1])))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[1], c[-1] = 0, b1, -1
        else:
            g_dists = np.array((1e-3 * great_circle_distance(data.lon[0],
                                                             data.lat[0],
                                                             data.lon[-1],
                                                             data.lat[-1]), ))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[-1] = 0, len(svel) - 1

    # time series processing:
    gids = svel > .5
    dto = np.diff(data.dday[~gids])  # time intervals when stopping
    # mov = np.where(dto > (1.02 * dts / 3600. / 24))[0]  # iterative points??
    mov = np.where(dto > (1. / 24))[0] + 1  # iterative points??

    if not np.isin(0, mov) and np.isin(len(dto), mov):
        mov = np.concatenate(([0, ], mov))
    elif np.isin(0, mov) and not np.isin(len(dto), mov):
        mov = np.concatenate((mov, [len(dto), ]))
    elif not np.isin(0, mov) and not np.isin(len(dto), mov):
        mov = np.concatenate(([0, ], mov, [len(dto), ]))
    raw_inds = np.arange(0, len(~gids))

    TSL = []
    tind = []
    for m in range(0, len(mov)-1):
        lons = data.lon[~gids][mov[m]:mov[m+1]]
        lats = data.lat[~gids][mov[m]:mov[m+1]]
        allddays = data.dday[~gids][mov[m]:mov[m+1]]
        t_raw_inds = raw_inds[~gids][mov[m]:mov[m+1]]
        if len(lons.compressed()) > 0:
            tlon, tlat = find_most_common_position(lons.compressed(),
                                                   lats.compressed())
            noid = lonlat_inside_km_radius(lons, lats, (tlon, tlat), rts)
            tsdays = allddays[noid]
            if len(tsdays) > 0:
                noids, tsdaysl = check_time_continuity(tsdays, allddays)
                for anoid, tsdays in zip(noids, tsdaysl):
                    TSL.append(tsdays[-1] - tsdays[0])
                    tind.append(t_raw_inds[anoid])
            counter = 0
            while np.count_nonzero(~noid) > lts * 3600 // dts:
                tlon, tlat = find_most_common_position(lons[~noid].compressed(),
                                                       lats[~noid].compressed())
                noid2 = lonlat_inside_km_radius(lons[~noid], lats[~noid],
                                                (tlon, tlat), rts)
                tsdays2 = allddays[~noid][noid2]
                if len(tsdays2) > 0:
                    noids, tsdaysl = check_time_continuity(tsdays2,
                                                           allddays[~noid])
                    for anoid, tsdays in zip(noids, tsdaysl):
                        TSL.append(tsdays[-1] - tsdays[0])
                        tind.append(t_raw_inds[~noid][anoid])
                lons = lons[~noid]
                lats = lats[~noid]
                allddays = allddays[~noid]
                t_raw_inds = t_raw_inds[~noid]
                noid = noid2
                counter += 1
                if counter > 100:
                    break
    return g_dists, c, np.asarray(TSL), np.asarray(tind), dts


def check_time_continuity(tsdays, allddays, jump_len=3.):
    '''
    Check that a transect or point timeseries has continuous data acquisition
     defined by the optional argument jump_len [in hrs]
    '''
    dtsdays = np.diff(tsdays)
    jumps = np.where(dtsdays >= jump_len / 24.)[0] + 1
    noids = []
    tsdays_list = []
    if not np.isin(0, jumps) and np.isin(len(dtsdays), jumps):
        jumps = np.concatenate(([0, ], jumps))
    elif np.isin(0, jumps) and not np.isin(len(dtsdays), jumps):
        jumps = np.concatenate((jumps, [len(dtsdays) + 1, ]))
    elif not np.isin(0, jumps) and not np.isin(len(dtsdays), jumps):
        jumps = np.concatenate(([0, ], jumps, [len(dtsdays)+1, ]))
    for jm in range(len(jumps)-1):
        new_tsdays = tsdays[jumps[jm]:jumps[jm+1]]
        __, x_ind, __ = np.intersect1d(allddays, new_tsdays,
                                       return_indices=True)
        new_noid = np.zeros_like(allddays, dtype=bool)
        new_noid[x_ind] = True
        noids.append(new_noid)
        tsdays_list.append(new_tsdays)
    return noids, tsdays_list


def eval_proc_transects(data, g_dists, c, nsegs, dts, mtl, mas, cruise_id,
                        instru_id, vessel_id, sac_id, lut, d):
    svel = data.spd
    counter = 0
    for n in range(0, nsegs):
        ndp = np.ma.count(svel[c[n]:c[n+1] + 1])  # num of valid nav pts
        if ndp < 20:
            print('Not enough points in this chunk, skipping to the next')
        else:
            g_dist = g_dists[n]  # great circle distance between start/end pts
            a_spd = svel[c[n]:c[n+1] + 1].mean()
            dcover = 1e-3 * np.sum(svel[c[n]:c[n+1] + 1] * dts)  # should ~ g_dist
            seg_len_days = data["dday"][c[n+1]] - data["dday"][c[n]]
            lons = data["lon"][c[n]:c[n+1]+1]
            lats = data["lat"][c[n]:c[n+1]+1]
            dx, dy = diffxy_from_lonlat(lons, lats)
            dl = 1e-3 * np.ma.sqrt(dx**2 + dy**2)
            dcover_l = dl.sum()
            dlg = great_circle_distance(lons[:-1], lats[:-1],
                                        lons[1:], lats[1:]) / 1e3
            dcover_g = dlg.sum()
            trans_orient = np.rad2deg(np.arctan(np.abs(dy.sum() / dx.sum())))

            # some tests must be made to know if its worth saving the seg data
            gndp = int(round(mtl / (dts * a_spd / 1e3)))
            dacond = (dcover >= mtl and g_dist >= mtl and
                      a_spd > mas and ndp >= gndp)

            if dacond:
                # figure out number of gaps and size of gaps (rms and max)
                nmdpt = np.ma.count_masked(data.u[c[n]:c[n+1]+1], axis=0)
                ngaps = 100. * nmdpt / len(data.u[c[n]:c[n+1]+1])
                gap_max = np.zeros((data.dep.shape))
                gap_tip = np.zeros((data.dep.shape))
                for k in range(0, len(data.dep)):
                    gaps = np.ma.clump_masked(data.u[c[n]:c[n+1]+1, k])
                    # gap_sizes = [len(np.arange(p.start, p.stop+1)) for p in gaps]
                    gap_sizes = [np.ma.sum(dl[p]) for p in gaps]
                    if len(gaps) > 0:
                        gap_max[k] = np.ma.max(gap_sizes)
                        # gap_tip[k] = np.ma.median(gap_sizes)
                        gap_tip[k] = Mode(gap_sizes)[0][0]

                seg_data = Bunch()
                seg_data.headings = data["heading"][c[n]:c[n+1]+1]
                seg_data.cogs = data["cog"][c[n]:c[n+1]+1]
                seg_data.lon, seg_data.lat, seg_data.dl = lons, lats, dl
                seg_data.svel = svel[c[n]:c[n+1]+1]
                seg_data.u = data["u"][c[n]:c[n+1]+1]
                seg_data.v = data["v"][c[n]:c[n+1]+1]
                seg_data.pg = data["pg"][c[n]:c[n+1]+1]
                seg_data.amp = data["amp"][c[n]:c[n+1]+1]
                seg_data.amp1 = data["amp1"][c[n]:c[n+1]+1]
                seg_data.amp2 = data["amp2"][c[n]:c[n+1]+1]
                seg_data.amp3 = data["amp3"][c[n]:c[n+1]+1]
                seg_data.amp4 = data["amp4"][c[n]:c[n+1]+1]
                seg_data.dday = data["dday"][c[n]:c[n+1]+1]
                seg_data.uship = data["uship"][c[n]:c[n+1]+1]
                seg_data.vship = data["vship"][c[n]:c[n+1]+1]
                seg_data.depth = data["depth"][c[n]:c[n+1]+1]
                seg_data.errs = data["e"][c[n]:c[n+1]+1]
                seg_data.ymdhms = data["ymdhms"][c[n]:c[n+1]+1]
                month = Mode(data.ymdhms[c[n]:c[n+1]+1, 1], axis=None)[0][0]
                year = Mode(data.ymdhms[c[n]:c[n+1]+1, 0], axis=None)[0][0]
                datuple = (instru_id, cruise_id, vessel_id, sac_id, d,
                           data.yearbase, year, month, lats.min(), lats.max(),
                           lons.min(), lons.max(), g_dist, dcover,
                           seg_len_days, trans_orient, a_spd, data.dep, dts,
                           np.ma.median(dl), ngaps, gap_max, gap_tip, seg_data)
                lut.append(datuple)
                counter = counter + 1
    print("final number of usable transects for this db is " + str(counter))
    return lut


def eval_proc_timeseries(data, ts_len, tinds, nts, dts, lts, rts, cruise_id,
                         instru_id, vessel_id, sac_id, ts_lut, d):
    gndp = int(round(lts / dts) * .9)
    counter = 0
    for n in range(0, nts):
        ndp = np.ma.count(data.spd[tinds[n]])  # num of valid nav pts
        if ndp < 36:
            print('Not enough points in this series, skipping to the next')
        else:
            if (ts_len[n] * 24 >= lts and ndp >= gndp):
                # addtional tests must be made to know if its worth saving data
                a_spd = np.ma.median(data.spd[tinds[n]])
                lons = data["lon"][tinds[n]]
                lats = data["lat"][tinds[n]]
                clon, clat = find_most_common_position(lons.compressed(),
                                                       lats.compressed())
                # t_dist = get_distmatrix(lons, lats)  # distance matrix
                # dacond = (ts_len >= lts and a_spd < mas and ndp >= gndp)

                nmdpt = np.ma.count_masked(data.u[tinds[n]], axis=0)
                ngaps = 100. * nmdpt / len(data.u[tinds[n]])
                gap_max = np.zeros((data.dep.shape))
                gap_tip = np.zeros((data.dep.shape))
                for k in range(0, len(data.dep)):
                    gaps = np.ma.clump_masked(data.u[tinds[n], k])
                    # gap_sizes = [len(np.arange(p.start, p.stop+1)) for p in gaps]
                    gap_sizes = [dts * (p.stop - p.start + 1) for p in gaps]
                    if len(gaps) > 0:
                        gap_max[k] = np.ma.max(gap_sizes)
                        # gap_tip[k] = np.ma.median(gap_sizes)
                        gap_tip[k] = Mode(gap_sizes)[0][0]

                ts_data = Bunch()
                ts_data.headings = data["heading"][tinds[n]]
                ts_data.cogs = data["cog"][tinds[n]]
                ts_data.lon = lons
                ts_data.lat = lats
                ts_data.svel = data["spd"][tinds[n]]
                ts_data.u = data["u"][tinds[n]]
                ts_data.v = data["v"][tinds[n]]
                ts_data.pg = data["pg"][tinds[n]]
                ts_data.amp = data["amp"][tinds[n]]
                ts_data.amp1 = data["amp1"][tinds[n]]
                ts_data.amp2 = data["amp2"][tinds[n]]
                ts_data.amp3 = data["amp3"][tinds[n]]
                ts_data.amp4 = data["amp4"][tinds[n]]
                ts_data.dday = data["dday"][tinds[n]]
                ts_data.uship = data["uship"][tinds[n]]
                ts_data.vship = data["vship"][tinds[n]]
                ts_data.depth = data["depth"][tinds[n]]
                ts_data.errs = data["e"][tinds[n]]
                ts_data.ymdhms = data["ymdhms"][tinds[n]]
                month = Mode(data.ymdhms[tinds[n], 1], axis=None)[0][0]
                year = Mode(data.ymdhms[tinds[n], 0], axis=None)[0][0]
                tstuple = (instru_id, cruise_id, vessel_id, sac_id, d,
                           data.yearbase, year, month, clon, clat,
                           ts_len[n], a_spd, data.dep, dts,
                           ngaps, gap_max, gap_tip, ts_data)
                ts_lut.append(tstuple)
                counter = counter + 1
    print("final number of usable timeseries for this db is " + str(counter))
    return ts_lut


def loop_proc_dbs(dbslist, mas, tst, mtl, lts, rts):
    # iterate and segment:
    lut = []  # this list will be appended with each useful transect
    ts_lut = []  # this list will be appended with each useful timeseries

    for m, d in enumerate(dbslist):
        print("doing database: ", d)
        data = read_codas_db_wrap(d)
        if data is None:
            continue
        elif np.count_nonzero(data.dday == 0) > 1:
            print("weird database, skipping")
            continue

        # get meta-data (depends if JAS or UHDAS)
        cruise_id, instru_id, vessel_id, sac_id = read_metadata_wrap(data, d)

        # start processing here:

        # check if there are breaks:
        g_dists, c, ts_len, tinds, dts = find_stations_restarts(data, mas,
                                                                tst, lts, rts)

        nsegs = len(g_dists)
        print("DB " + d + " has", nsegs, "transects to evaluate")

        nts = len(ts_len)
        print("DB " + d + " has", nts, "point timeseries to evaluate")

        if nsegs > 0:
            lut = eval_proc_transects(data, g_dists, c, nsegs, dts, mtl, mas,
                                      cruise_id, instru_id, vessel_id, sac_id,
                                      lut, d)
        if nts > 0:
            ts_lut = eval_proc_timeseries(data, ts_len, tinds, nts, dts, lts,
                                          rts, cruise_id, instru_id, vessel_id,
                                          sac_id, ts_lut, d)

    lut = np.array(lut, dtype=[("inst_id", '<U19'), ("cruise_id", '<U19'),
                               ("vessel_id", '<U19'), ("sac_id", '<U19'),
                               ("db_path", '<U19'),
                               ('yearbase', 'int32'), ('year', 'int32'),
                               ('month', 'int32'), ('lat_min', 'float32'),
                               ('lat_max', 'float32'), ('lon_min', 'float32'),
                               ('lon_max', 'float32'), ('g_dist', 'float32'),
                               ('dcover', 'float32'), ('seg_days', 'float32'),
                               ('trans_orient', 'float32'),
                               ('avg_spd', 'float32'),
                               ('dep', 'O'),
                               ('dt', 'float16'), ('dlm', 'float16'),
                               ('ngaps', 'O'), ('gap_max', 'O'),
                               ('gap_tipical', 'O'),
                               ('seg_data', 'O')])

    ts_lut = np.array(ts_lut, dtype=[("inst_id", '<U19'),
                                     ("cruise_id", '<U19'),
                                     ("vessel_id", '<U19'), ("sac_id", '<U19'),
                                     ("db_path", '<U19'),
                                     ('yearbase', 'int32'), ('year', 'int32'),
                                     ('month', 'int32'), ('lon', 'float32'),
                                     ('lat', 'float32'),
                                     ('duration', 'float32'),
                                     ('avg_spd', 'float32'), ('dep', 'O'),
                                     ('dt', 'float16'), ('ngaps', 'O'),
                                     ('gap_max', 'O'), ('gap_tipical', 'O'),
                                     ('ts_data', 'O')])
    return lut, ts_lut


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("listfile",
                        help="file or list/tuple with CODAS databses or path",
                        type=str)
    parser.add_argument("outdir",
                        help="directory where output will go",
                        type=str)
    parser.add_argument("-out_fname",
                        help="output file name; optional",
                        type=str)
    parser.add_argument("-mas",
                        help="minimum average ship speed in transect (m/s)",
                        type=float)
    parser.add_argument("-tst",
                        help="tolerated stop time during transect (hrs)",
                        type=float)
    parser.add_argument("-mtl",
                        help="minimum usable transect length (km)",
                        type=float)
    parser.add_argument("-lts",
                        help="minimum length of usable pt timeseries (hrs)",
                        type=float)
    parser.add_argument("-rts",
                        help="maximum radious for a pt timeseries (km)",
                        type=float)
    args = parser.parse_args()
    arg_dict = vars(args)
    noarg_list = list({el for el in arg_dict.keys() if arg_dict[el] is None})
    kw_args = {k: v for k, v in arg_dict.items() if v is not None and k != 'listfile' and k != 'outdir'}

    if len(noarg_list) == 6:
        configs = RunParams(args.listfile, args.outdir)
        print('\nExecuting the processing code w/ default settings\n')
    else:
        configs = RunParams(args.listfile, args.outdir, **kw_args)
    return configs


def _configure(arg1, arg2, arg3=None, arg4=None, arg5=None, arg6=None,
               arg7=None, arg8=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("listfile",
                        help="file or list/tuple with CODAS databses or path",
                        type=str)
    parser.add_argument("outdir",
                        help="directory where output will go",
                        type=str)
    parser.add_argument("-out_fname",
                        help="output file name; optional",
                        type=str)
    parser.add_argument("-mas",
                        help="minimum average ship speed in transect (m/s)",
                        type=float)
    parser.add_argument("-tst",
                        help="tolerated stop time during transect (hrs)",
                        type=float)
    parser.add_argument("-mtl",
                        help="minimum usable transect length (km)",
                        type=float)
    parser.add_argument("-lts",
                        help="minimum length of usable pt timeseries (hrs)",
                        type=float)
    parser.add_argument("-rts",
                        help="maximum radious for a pt timeseries (km)",
                        type=float)
    # args = parser.parse_args([arg1, arg2])
    args = parser.parse_args([arg1, arg2, "-out_fname", arg3, "-mas", arg4,
                              "-tst", arg5, "-mtl", arg6,
                              "-lts", arg7, "-rts", arg8])
    arg_dict = vars(args)
    print(arg_dict)
    noarg_list = list({el for el in arg_dict.keys() if arg_dict[el] is None})
    kw_args = {k: v for k, v in arg_dict.items() if v is not None and k != 'listfile' and k != 'outdir'}

    if len(noarg_list) == 6:
        configs = RunParams(args.listfile, args.outdir)
        print('Executing the processing code w/ default settings\n')
    else:
        configs = RunParams(args.listfile, args.outdir, **kw_args)
    return configs, args


def main():
    # initialize - w/ or w/o argeparse capabilities
    # run main loop
    # write to file
    configs = configure()
    lut, ts_lut = loop_proc_dbs(configs.dbslist, configs.mas, configs.tst,
                                configs.mtl, configs.lts, configs.rts)
    save_lut(configs.output_files_ids[0], lut)
    save_lut(configs.output_files_ids[1], ts_lut)


if __name__ == '__main__':
    main()
