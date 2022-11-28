#!/usr/bin/env python

'''
Workhorse script to search several CODAS dbs for criteria:

samples within an area, and/or
samples within a period or in certain months or years, and/or
uses a particular ADCP.

Input: (path to) CODAS dbs list
Output: data struc with select dbs/cruises and their metadata
'''


import numpy as np
import os
import fnmatch
import argparse
from pycurrents.codas import get_profiles
from pycurrents.codas import get_txy
from pycurrents.data.navcalc import lonlat_inside_km_radius
from pycurrents.data.navcalc import (great_circle_distance, diffxy_from_lonlat)
from pycurrents.system import Bunch
from pycurrents.file import npzfile
from pycurrents.adcp.panelplotter import get_netCDF_data
from scipy.stats import mode as Mode
from pysadcp.pysadcp import read_meta_from_bft
from pysadcp.pysadcp import read_meta_from_dbinfo
from pysadcp import find_most_common_position
from pysadcp import convert_lonEW_to_lonE
from pysadcp import convert_lonE_to_lonEW
from pysadcp.pysadcp import read_nav_from_db_list
from pysadcp.process_codas_dbs_L1 import load_dbs_list
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# define bounding box / area:
lons_vect = [-119.18, -119.35, -117.89, -117.68, -119.18]
lats_vect = [34.1, 33.85, 33.078, 33.43, 34.1]

# custom functions?
def check_transect_within_pol(lons_vect, lats_vect, nav_data, crit=100):
    '''
    Check if db samples within an area defined by a polygon
    '''

    # form polygon
    lons_lats_vect = np.column_stack((lons_vect, lats_vect))
    polygon = Polygon(lons_lats_vect)

    # iterate points (only valid nav)
    lons, lats = nav_data.lon.compressed(), nav_data.lat.compressed()
    points = [Point(x, y) for x, y in zip(lons, lats)]
    inpol = np.array([polygon.contains(pt) for pt in points])

    # if more pts than crit add to keeper
    if np.count_nonzero(inpol) > crit:
        return True
    else:
        return False


def read_nav_from_db_list(database):
    return nav_data


def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    import sys
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, u"#"*x, "."*(size-x), j, count),
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def search_db_list(databases, N=1):

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

    for d in progressbar(databases):

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
                pass
        else:
            year = dataxy.yearbase
            month = dataxy.ymdhms[0, 1]

            check1 = check_transect_within_pol(lons_vect, lats_vect,
                                               dataxy, crit=N)
            checks = check1

            if checks and  '.ignore' not in d:
                print(d + ' sampled within region')

                dpassed.append(d)
                alllons.append(dataxy.lon)
                alllats.append(dataxy.lat)

                years.append(dataxy.ymdhms[0, 0])
                months.append(dataxy.ymdhms[0, 1])

                # bftfile = d + '/' + os.path.split(d)[1] + '.bft'
                bftfile = d + '.bft'
                dbinfo_file = os.path.split(os.path.split(d)[0])[0] + '/dbinfo.txt'  # fix!
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
            else:
                print(d + ' failed checks')
                dfailed.append(d)

    nav_data = np.rec.fromarrays((dpassed, cruise_ids, instru_ids, vessel_ids,
                                  sac_ids, years, months, alllons, alllats),
                                 names=('db path', 'cruise id', 'sonar id',
                                        'vessel id', 'SAC id', 'yearbase',
                                        'month', 'longitude', 'latitude')
                                 )

    return nav_data, dfailed



keepers = check_transect_within_pol(lons_vect, lats_vect, transect_data_socal, crit=25)


# input of databases: (to be arg parsed)


# crawl (iterate dbs and perform checks)
