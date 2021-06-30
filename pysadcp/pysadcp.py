#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main module of pySADCP."""

import numpy as np
import os
import re
from pycurrents.data.navcalc import lonlat_inside_km_radius


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


def main():
    print('This would execute the pysadcp package')


if __name__ == '__main__':
    main()
