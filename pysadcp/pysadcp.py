#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main module of pySADCP."""

import numpy as np
import os
import re
from pycurrents.data.navcalc import lonlat_inside_km_radius
from pycurrents.data.navcalc import great_circle_distance
from pycurrents.codas import get_txy


def read_meta_from_bft(bftfile):
    '''
    Reads a CODAS meta data bft file (only JAS-ADCP repo) to determine cruise,
    sonar, vessel and SAC ID.
    '''
    pattern = re.compile("CRUISE_NAME", re.IGNORECASE)
    sac_pattern = re.compile("SAC_CRUISE_ID", re.IGNORECASE)
    plat_pattern = re.compile("#PLATFORM_NAME", re.IGNORECASE)
    with open(bftfile, 'rt', encoding="utf8", errors='ignore') as bft:
        cruise_id, instru_id, vessel_id, sac_id = None, None, None, None
        for line in bft:
            if plat_pattern.search(line):
                vessel_id = line.split(':')[-1].rstrip('\n').lstrip()
            if sac_pattern.search(line):
                sac_id = line.split(':')[-1].rstrip(' \n').lstrip()
            if pattern.search(line):
                if len(line.split(':')) == 3:
                    cruise_id = line.split(':')[-2].rstrip(' ').lstrip()
                    instru_id = line.split(':')[-1].rstrip('\n').lstrip()
                    # need to check and accomodate ship tag?
                elif len(line.split(':')) == 2:
                    cruise_id = line.split(':')[-1].rstrip(' ').rstrip('\n').lstrip()
                    # instru_id = 'nb150-assumed'  # need the special read/dict
                    instru_id = translate_inst_str(read_bad_meta(bft))
                elif len(line.split(':')) == 4:
                    instru_id, crid2 = line.split(':')[-2].split(' ')
                    cruise_id = line.split(':')[-3].lstrip() + ' ' + crid2
                else:
                    print('Found CRUISE_NAME but cannot understand!')
                    cruise_id = 'unknown_not_understood'
                    instru_id = 'unknown_not_understood'
                break
        if cruise_id is None:
            print('CRUISE_NAME meta data not found in bft!')
            cruise_id = 'unknown_read_fail'
            instru_id = 'unknown_read_fail'
        if vessel_id is None:
            print('PLATFORM_NAME meta data not found in bft!')
            vessel_id = 'unknown_read_fail'
        if sac_id is None:
            print('SAC_CRUISE_ID meta data not found in bft!')
            sac_id = 'unknown_read_fail'
    return cruise_id, instru_id, vessel_id, sac_id


def read_meta_from_dbinfo(dbinfo_file):
    '''
    Reads a CODAS meta data text file to determine cruise, sonar and vessel
    '''

    pattern_cruise = re.compile("cruisename", re.IGNORECASE)
    pattern_sonar = re.compile("sonar", re.IGNORECASE)
    with open(dbinfo_file, 'rt') as dbinfo:
        cruise_id, instru_id = None, None
        for line in dbinfo:
            if pattern_cruise.search(line):
                cruise_id = line.lstrip().rstrip().split(' ')[-1]
            elif pattern_sonar.search(line):
                instru_id = line.lstrip().rstrip().split(' ')[-1]
        if cruise_id is None:
            print('CRUISENAME meta data not found in dbinfo.txt!')
            cruise_id = 'unknown_read_fail'
        if instru_id is None:
            print('sonar meta data not found in dbinfo.txt!')
            instru_id = 'unknown_read_fail'
    if instru_id != 'unknown_read_fail':
        basedir = os.path.split(os.path.split(dbinfo_file)[0])[0]
        cruise_sonar_file = basedir + '/' + instru_id + '.txt'
        pattern_vessel = re.compile("ship name", re.IGNORECASE)
        if os.path.exists(cruise_sonar_file):
            with open(cruise_sonar_file, 'rt') as txtinfo:
                vessel_id = 'unknown_read_fail'
                for line in txtinfo:
                    if pattern_vessel.search(line):
                        vessel_id = line.split(':')[-1].rstrip('\n').lstrip()
        elif os.path.exists(cruise_sonar_file[:-6] + '.txt'):
            with open(cruise_sonar_file[:-6] + '.txt', 'rt') as txtinfo:
                vessel_id = 'unknown_read_fail'
                for line in txtinfo:
                    if pattern_vessel.search(line):
                        vessel_id = line.split(':')[-1].rstrip('\n').lstrip()
        else:
            vessel_id = 'unknown_read_fail'
    else:
        vessel_id = 'unknown_read_fail'
    return cruise_id, instru_id, vessel_id


def read_bad_meta(inF):
    '''
    Helps determining sonar characteristics from the meta data text files
    '''

    pt = ('narrowband', 'broadband', 'broad band', 'narrow band', 'Narrowband',
          'Broadband', 'Broad band', 'Narrow band', 'NarrowBand', 'BroadBand',
          'Broad Band', 'Narrow Band')
    for index, line in enumerate(inF):
        if ' MANUFACTURER ' in line:
            sonar_line = line[line.find(":")+1:].strip()
        if ' HARDWARE MODEL ' in line:
            sonar_line = sonar_line + line[line.find(":")+1:-1].strip()
        if ' TRANSMIT FREQUENCY ' in line and any([p in line for p in pt]):
            cl = line[line.find(":")+1:-1]
            return (sonar_line + cl).strip()
        if ' COMMENTS ' in line and any([p in line for p in pt]):
            cl = line[line.find(":")+1:-1]
            return (sonar_line + cl).strip()
    return sonar_line


def translate_inst_str(astr):
    '''
    Convert long string sequences with sonar information into standard acronyms
    '''

    good_strs = ('os38nb', 'os38bb', 'os75nb', 'os75bb', 'os150nb', 'os150bb',
                 'wh300', 'nb150', 'bb150')
    if astr not in good_strs:
        if (
            'NB 150' in astr or 'VM-150' in astr or 'VM150' in astr or
            'NB-150' in astr or 'NB150' in astr or
            '150 kHz Narrowband' in astr or 'Narrowband 150' in astr or
            'Narrow Band 150Khz' in astr or 'Narrow Band 150' in astr
             ):
            return 'nb150'
        elif (
            'Broad Band 150' in astr or 'BB 150' in astr or
            'Broadband 150' in astr
             ):
            return 'bb150'
        elif (
            'RD-VM300' in astr or 'VM-300' in astr
             ):
            return '??300'
        elif (
            'Ocean Surveyor 75' in astr and 'narrowband' in astr
             ) or ('OS75' in astr and 'narrowband' in astr) or (
             '75KHz Ocean Surveyor' in astr and 'narrowband' in astr) or (
             'Ocean Surveyor 75' in astr and 'Narrowband' in astr
             ):
            return 'os75nb'
        elif (
            'Ocean Surveyor 75' in astr and 'broadband' in astr
             ) or ('OS75' in astr and 'broadband' in astr) or (
             '75KHz Ocean Surveyor' in astr and 'broadband' in astr) or (
             'Ocean Surveyor 75' in astr and 'Broadband' in astr
             ):
            return 'os75bb'
        elif (
            'Ocean Surveyor 75' in astr and 'broadband' not in astr
             ) or ('OS75' in astr and 'broadband' not in astr) or (
             '75KHz Ocean Surveyor' in astr and 'broadband' not in astr or
            'Ocean Surveyor 75' in astr and 'narrowband' not in astr
             ) or ('OS75' in astr and 'narrowband' not in astr) or (
             '75KHz Ocean Surveyor' in astr and 'narrowband' not in astr) or (
             'Ocean Surveyer 75' in astr and 'narrowband' not in astr) or (
             'Ocean Surveyer 75' in astr and 'broadband' not in astr
             ):
            return 'os75??'
        elif (
            'Ocean Surveyor 38' in astr and 'narrowband' in astr
             ) or ('OS38' in astr and 'narrowband' in astr) or (
             '38KHz Ocean Surveyor' in astr and 'narrowband' in astr) or (
             'Ocean Surveyer 38' in astr and 'narrowband' in astr) or (
             '38KHz Ocean Surveyer' in astr and 'narrowband' in astr):
            return 'os38nb'
        elif (
            'Ocean Surveyor 38' in astr and 'broadband' in astr
             ) or ('OS38' in astr and 'broadband' in astr) or (
             '38KHz Ocean Surveyor' in astr and 'broadband' in astr) or (
             'Ocean Surveyer 38' in astr and 'broadband' in astr) or (
             '38KHz Ocean Surveyer' in astr and 'broadband' in astr
             or 'OS38NB ' in astr):
            return 'os38bb'
        else:
            return astr
    else:
        return astr


def find_most_common_position(lons, lats, darad=.05):
    '''
    Find the most common position (lon, lat pair) in a sequence of coordinates.
    Inputs are longitude and latitude arrays in degrees.
    Outputs the lon, lat pair of the most common location.
    Optional input is the radius in km to determine the "location", use very
    small radii make location selective.
    '''
    pos_list = []
    for alon, alat, in zip(lons, lats):
        pos_c = np.count_nonzero(lonlat_inside_km_radius(lons, lats,
                                                         (alon, alat), darad))
        pos_list.append(pos_c)
    pos_list = np.asarray(pos_list)
    dalon = lons[pos_list == max(pos_list)][0]  # may not be unique
    dalat = lats[pos_list == max(pos_list)][0]
    return dalon, dalat


def convert_lonEW_to_lonE(lon):
    if np.any(lon < 0.):
        if np.isscalar(lon) is True:
            return lon + 360.
        else:
            lon[lon < 0.] = lon[lon < 0.] + 360.
            return lon
    else:
        return lon


def convert_lonE_to_lonEW(lon):
    if np.any(lon > 180e0):
        if np.isscalar(lon) is True:
            return lon - 360.
        else:
            lon[lon > 180e0] = lon[lon > 180e0] - 360.
            return lon
    else:
        return lon


def read_nav_from_db_list(databases, dlims=None, N=1, Delta=0.):

    if dlims is not None:
        xmin, xmax, ymin, ymax = dlims[0], dlims[1], dlims[2], dlims[-1]
        xmin = convert_lonEW_to_lonE(xmin)  # useful to go across pacific dateline
        xmax = convert_lonEW_to_lonE(xmax)

    dpassed = []
    dfailed = []
    alllats = []
    alllons = []
    years = []
    months = []
    cruise_ids = []
    instru_ids = []
    vessel_ids = []
    sac_ids = []
    for d in databases:

        try:
            dataxy = get_txy(d)

        except ValueError as e:
            if "has 2 block directories" in str(e):
                print('There was a problem reading this db (2 block dirs), skipping')
                dfailed.append(d)
                pass
            elif 'has no block directory' in str(e):
                print('No codas blk data in path of db, skipping')
                dfailed.append(d)
                pass
            else:
                print('Could not read this db path for unknown reason, aborting')
                break
        else:
            lats = dataxy["lat"]
            lons = convert_lonEW_to_lonE(dataxy["lon"])
            # lons = dataxy["lon"]

            if dlims is not None:
                idx = np.logical_and(lons >= xmin, lons <= xmax)
                idy = np.logical_and(lats >= ymin, lats <= ymax)
                ids = np.logical_and(idx, idy)

                # check if lat_final is same size as lon_final (maybe should be assert)
                if len(lats[ids]) != len(lons[ids]):
                    raise Exception('Indexed lat and lon not of same size!')
            else:
                ids = slice(0, -1)

            if len(lons[ids]) >= N and '.ignore' not in d:
                # check if earth distance covered is at least equal to Delta:
                dl = great_circle_distance(dataxy.lon[ids][:-1], lats[ids][:-1],
                                           dataxy.lon[ids][1:], lats[ids][1:]) / 1e3
                if dl.sum() >= Delta:
                    dpassed.append(d)
                    alllons.append(dataxy.lon)
                    alllats.append(lats)
                    print(d + ' passed, dist covered: ' + str(np.round(dl.sum())) + ' km')

                    years.append(dataxy.yearbase)
                    months.append(dataxy.ymdhms[ids][len(dataxy.ymdhms[ids]) // 2, 1])

                    bftfile = d + '.bft'
                    dbinfo_file = os.path.split(os.path.split(d)[0])[0] + '/dbinfo.txt'
                    if os.path.exists(bftfile):
                        cruise_id, instru_id, vessel_id, sac_id = read_meta_from_bft(bftfile)
                        cruise_ids.append(cruise_id)
                        instru_ids.append(instru_id)
                        vessel_ids.append(instru_id)
                        sac_ids.append(instru_id)
                    elif os.path.exists(dbinfo_file):
                        cruise_id, instru_id, vessel_id = read_meta_from_dbinfo(dbinfo_file)
                        cruise_ids.append(cruise_id)
                        instru_ids.append(instru_id)
                        vessel_ids.append(instru_id)
                        sac_ids.append('None/UH?')
                    else:
                        print('No meta data file found!')
                        cruise_ids.append('unknown_no_metafile')
                        instru_ids.append('unknown_no_metafile')
                        vessel_ids.append('unknown_no_metafile')
                        sac_ids.append('unknown_no_metafile')

    nav_data = np.rec.fromarrays((dpassed, cruise_ids, instru_id, vessel_ids,
                                  sac_ids, years, months, alllons, alllats),
                                 names=('db path', 'cruise id', 'sonar id',
                                        'vessel id', 'SAC id', 'yearbase',
                                        'month', 'longitude', 'latitude')
                                 )

    return nav_data, dfailed


def read_gps_txt(gpsfile, use_encoding=False):
    if use_encoding:
        gpsdat = np.genfromtxt(gpsfile, encoding='utf-8')
        # dday = gpsdat[:, 0]
        lon = gpsdat[:, 1]
        lat = gpsdat[:, 2]
        return lon, lat
    else:
        gpsdat = np.genfromtxt(gpsfile)
        try:
            # gpsdat = np.genfromtxt(gpsfile)
            lon = gpsdat[:, 1]
            lat = gpsdat[:, 2]
            return lon, lat
        except IndexError:
            # if 'Some errors were detected' in e:
            return None, None


def read_nav_from_gps_list(gpsfiles, dlims=None, N=1, Delta=0.):

    if dlims is not None:
        xmin, xmax, ymin, ymax = dlims[0], dlims[1], dlims[2], dlims[-1]
        xmin = convert_lonEW_to_lonE(xmin)  # useful to go across pacific dateline
        xmax = convert_lonEW_to_lonE(xmax)

    dpassed = []
    alllats = []
    alllons = []
    cruise_ids = []
    instru_ids = []
    vessel_ids = []
    dfailed = []

    for gpsfile in gpsfiles:

        try:
            lons, lats = read_gps_txt(gpsfile)
        except UnicodeDecodeError:
            # lons, lats = read_gps_txt(gpsfile, use_encoding=True)
            dfailed.append(gpsfile)
            pass
        except ValueError:
            # if 'Some errors were detected' in e:
            dfailed.append(gpsfile)
            pass

        if lons is None or lats is None:
            dfailed.append(gpsfile)
            continue

        lons = convert_lonEW_to_lonE(lons)

        if dlims is not None:
            idx = np.logical_and(lons >= xmin, lons <= xmax)
            idy = np.logical_and(lats >= ymin, lats <= ymax)
            ids = np.logical_and(idx, idy)

            # check if lat_final is same size as lon_final (maybe should be assert)
            if len(lats[ids]) != len(lons[ids]):
                raise Exception('Indexed lat and lon not of same size!')
        else:
            ids = slice(0, -1)

        if len(lons[ids]) >= N:
            # check if earth distance covered is at least equal to Delta:
            dl = great_circle_distance(lons[ids][:-1], lats[ids][:-1],
                                       lons[ids][1:], lats[ids][1:]) / 1e3
            if dl.sum() >= Delta:
                dpassed.append(gpsfile)
                alllons.append(lons)
                alllats.append(lats)
                print(gpsfile + ' passed, dist covered: ' + str(np.round(dl.sum())) + ' km')

                cruise_ids.append(gpsfile.split('/')[-5])
                instru_ids.append(gpsfile.split('/')[-3])
                vessel_ids.append(gpsfile.split('/')[-6])

    nav_data = np.rec.fromarrays((dpassed, cruise_ids, instru_ids, vessel_ids,
                                  alllons, alllats),
                                 names=('db_path', 'cruise_id', 'sonar_id',
                                        'vessel_id', 'longitude', 'latitude')
                                 )

    return nav_data, dfailed


def split_transect_by_heading(transects, heading_diff_crit=15., len_crit=10):
    '''split transects by abrupt heading changes'''
    new_transects = []

    for atran in transects:
        # meta_vars = [k for k in atran.dtype.fields.keys() if k != 'seg_data']
        # ndat = [atran[k] for k in atran.dtype.fields.keys() if k != 'seg_data']
        # ndat = tuple(ndat)
        atrandata = atran['seg_data']
        inds = fixed_heading_inds(atrandata, heading_diff_crit, len_crit)
        for n in range(0, len(inds)-1):
            if inds[n+1]+1 - inds[n]+1 >= len_crit:
                # print('inside')
                ndat = [atran[k] for k in atran.dtype.fields.keys() if k != 'seg_data']
                new_tran_data = Bunch()
                for key in atrandata.keys():
                    new_tran_data[key] = atrandata[key][inds[n]+1:inds[n+1]+1]
                ndat.append(new_tran_data)
                new_transects.append(tuple(ndat))
    new_data = np.array(new_transects, dtype=transects.dtype.descr)
    return new_data


def fixed_heading_inds(atrandata, heading_diff_crit=15., len_crit=10):
    '''find indices of constant heading transects by abrupt heading changes'''
    transect_headings = atrandata.headings
    heading_diffs = np.diff(transect_headings)
    heading_changes = np.where(np.abs(heading_diffs) > heading_diff_crit)

    # heading_changes_bool = np.abs(heading_diffs) > heading_diff_crit

    inds = np.hstack(([-1, ], heading_changes[0], len(transect_headings)))
    # split_headings = [transect_headings[inds[n]+1:inds[n+1]+1] for n in range(0, len(inds)-1)]
    # split_heading_len = np.asarray([len(shd) for shd in split_headings])

    return inds


def main():
    print('This would execute the pysadcp package')


if __name__ == '__main__':
    main()
