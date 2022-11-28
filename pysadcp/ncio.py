#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Input/output handling for pySADCP.
Class for making shipboard adcp netcdf files: a short form
for analysis, and a long form to approximately match the
CODAS database. The class can also be used from a script.
"""

import numpy as np
import argparse
import datetime
import logging
from netCDF4 import Dataset
from .process_transects_L2 import TransectsDB


# Standard logging
log = logging.getLogger(__name__)

description = """Generate shipboard adcp transect netcdf file from a transect npz file.
Works only for NETCDF4 for now."""
usage = """
io.py inputfile outfilebase
           [ -h] [--compressed]
           [--configs ??_?? [?? ...]]
eg.
    io.py oleander_L1_transects.npz  oleander_L1_transects

Test adcp_nc.py output with the standard netcdf command-line tool:

    ncdump -h outfile"""


# these dicts missing float16, int32 and <U str code [useful only for NETCDF3]
dvtypes = {'float64': 'f8',
           'float32': 'f4',
           'uint8': 'i2',
           'int16': 'i2',
           'uint16': 'i2',
           'int8': 'i1'}

# For netcdf4 we can use uint8.
dvtypes4 = {'float64': 'f8',
            'float32': 'f4',
            'uint8': 'u1',
            'int16': 'i2',
            'uint16': 'u2',
            'int8': 'i1'}

# Data structure used in the make_short and make_compressed methods:
                            # long_name  units                format
meta_vars = [
                ('inst_id',   ('Sonar model, frequency and ping type',
                               None,              '%str')),
                ('cruise_id',   ('Cruise identifier',
                                 None,              '%str')),
                ('vessel_id',   ('Vessel identifier',
                                 None,              '%str')),
                ('sac_id',   ('JAS repository id number',
                              None,              '%str')),
                ('yearbase',     ('year base for time',
                                  None,                  '%d')),
                ('year',     ('year at the start of transect',
                              None,                  '%d')),
                ('month',     ('month at the start of transect',
                               None,                  '%d')),
                ('lon_min',     ('minimum longitude of transect',
                                 'degrees_east',     '%9.4f')),
                ('lat_min',     ('minimum latitude of transect',
                                 'degrees_north',     '%9.4f')),
                ('lon_max',     ('maximum longitude of transect',
                                 'degrees_east',     '%9.4f')),
                ('lat_max',     ('maximum latitude of transect',
                                 'degrees_north',     '%9.4f')),
                ('g_dist',     ('Great circle distance between start and end',
                                'km',                '%8.2f')),
                ('dcover',     ('distance covered by transect',
                                'km',                '%8.2f')),
                ('avg_spd',   ('average ship speed over ground',
                               'meter second-1',     '%9.4f')),
                ('trans_orient',     ('orientation of transect',
                                      'degrees cw from east',   '%8.2f')),
                ('dlm',     ('Mean distance between profiles',
                             'km',                '%8.2f')),
                ('seg_days',     ('duration of transect',
                                  'days',                '%8.2f')),
                ('dt',     ('Mean time between each profile',
                            'seconds',                '%8.2f')),
                ('meta_dump',     ('entire codas meta data file',
                                   None,                '%str')),
            ]

txy_vars = [('dday',  ('Decimal day',
                       'days since YEARBASE-01-01 00:00:00',
                       '%12.5f')),
            ('lon',   ('Longitude',   'degrees_east',   '%9.4f')),
            ('lat',   ('Latitude',    'degrees_north',  '%9.4f')),
            ('headings', ('Ship heading', 'degrees',     '%6.1f')),
            ('uship',   ('Ship zonal velocity component',
                         'meter second-1',     '%9.4f')),
            ('vship',   ('Ship meridional velocity component',
                         'meter second-1',     '%9.4f')),
            ('dl',   ('Distance between profiles',
                      'meter',                 '%8.2f')),
            # ('tr_temp', ('ADCP transducer temperature',
            #              'Celsius',            '%4.1f')),
            ]

oneD_vars = [
                ('dep',   ('Depth of bins for first profile',
                           'meter',              '%8.2f')),
                ('ngaps',   ('Gapness of transect per bin',
                             'percent',          '%8.2f')),
                ('gap_max',   ('Size of largest gap in transect per bin',
                               'km',             '%8.2f')),
                ('gap_tipical',   ('Most frequent gap size in transect / bin',
                                   'km',          '%8.2f')),
            ]

twoD_vars = [
                ('depth',   ('Depth',
                                        'meter',              '%8.2f')),
                ('u',       ('Zonal velocity component',
                                        'meter second-1',     '%7.2f')),
                ('v',      ('Meridional velocity component',
                                        'meter second-1',     '%7.2f')),
                ('errs',      ('Error velocity',
                                        'meter second-1',     '%7.2f')),
                ('amp',     ('Received signal strength',
                                        None,                  '%d')),
                ('amp1',     ('Received signal strength beam 1',
                                        None,                  '%d')),
                ('amp2',     ('Received signal strength beam 2',
                                        None,                  '%d')),
                ('amp3',     ('Received signal strength beam 3',
                                        None,                  '%d')),
                ('amp4',     ('Received signal strength beam 4',
                                        None,                  '%d')),
                ('pg',      ('Percent good pings',
                                        None,                  '%d')), ]


meta_vars_names = [aname[0] for aname in meta_vars]
txy_vars_names = [aname[0] for aname in txy_vars]
oneD_vars_names = [aname[0] for aname in oneD_vars]
twoD_vars_names = [aname[0] for aname in twoD_vars]


class transects_nc:
    """
    Class to make netCDF files from a transects db.

    methods:

        make_nc(ncfilename)

    """

    def __init__(self, data, attrlist=None):
        '''
        data: transect data structured array or TransectsDB class
        optionally add global attributes via attrlist, a list of tuples,
           where each tuple is (attr_name , contents)
        '''

        if isinstance(data, TransectsDB):
            self.ntransects = len(data)
            self.nbins = data.get_nbin_max()  # useful for netCDF3 support
            self.nprofs = data.get_nprof_max()
            self.data = data.transects
        elif isinstance(data, np.ndarray):
            self.ntransects = len(data)
            self.nbins = self.get_alltran_nbin_max(data)
            self.nprofs = self.get_alltran_nprof_max(data)
            self.data = data
        else:
            raise ValueError("Incorrect data type")

        if attrlist is None:
            self.attrlist = []
        else:
            self.attrlist = attrlist

        # self.varkw = dict()

    def get_alltran_nbin_max(self, data):
        nbins = [len(dep) for dep in data['dep']]
        return max(nbins)

    def get_alltran_nprof_max(self, data):
        nprofs = [len(aseg.dday) for aseg in data['seg_data']]
        return max(nprofs)

    def create_nc(self, filename, ntransects=None, nbins=None,
                  nprofs=None, netcdf4=True):
        '''
        create the netCDF writer for filename (specify ntransects, nbins, nprofs)
        '''

        format = 'NETCDF4' if netcdf4 else 'NETCDF3_CLASSIC'
        nf = Dataset(filename, 'w', format=format)
        # format is stored in nf.data_model

        # We could make the time dimension unlimited by setting it
        # to None, but this doubles the run time for making the long
        # files.
        if ntransects is None:
            ntransects = self.ntransects

        nf.createDimension('transect', size=ntransects)

        if format != 'NETCDF4':
            if nbins is None:
                nbins = self.nbins

            if nprofs is None:
                nprofs = self.nprofs

            nf.createDimension('n_profs', nprofs)
            nf.createDimension('n_bins', nbins)

        # tp = nf.createVariable('transect', 'i4', tuple())  # scalar
        # tp.standard_name = "transect_number"
        # tp.assignValue(hash(self.cruise_id + ' ' + self.inst_ping))

        # groups of variables:

        if format == 'NETCDF4':
            nb = nf.createVariable('nbins', 'i2', ('transect',), fill_value=False)
            npr = nf.createVariable('nprofs', 'i2', ('transect',), fill_value=False)
            metagrp = nf.createGroup("metadata")
            navgrp = nf.createGroup("nav_data")
            diaggrp = nf.createGroup("qc_diag_data")
            twoDgrp = nf.createGroup("twoD_data")
            nb[:] = [len(dep) for dep in self.data['dep']]
            npr[:] = [len(dat['dday']) for dat in self.data['seg_data']]
            nb.long_name = 'Number of depth bins used in each transect'
            npr.long_name = 'Number of profile ensembles in each transect'
            metagrp.description = 'Meta data variables for transects'
            diaggrp.description = 'Diagnostic variables (transect, nbins)'
            navgrp.description = 'Navigation variables (transect, nprofs)'
            twoDgrp.description = '2D variables (transect, nprofs, nbins)'

        # global attributes:

        nf.featureType = "transectsDB"
        utc = datetime.datetime.utcnow()
        nf.history = 'Created: %s' % utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        nf.Conventions = 'COARDS'
        nf.software = 'pysadcp'

        n_codas_dbs = len(np.unique(self.data['cruise_id']))
        nf.title = 'Shipboard ADCP transects database'
        nf.description = ('Shipboard ADCP transect data ' +
                          'assembled from %s CODAS databases'
                          % (str(n_codas_dbs)))

        # add optional global attributes [run configs?]
        for attr_tup in self.attrlist:
            setattr(nf, attr_tup[0], attr_tup[1])

        self.nf = nf

    def add_array_var_netcdf4(self, vname, var, vdat, **varkw):
        # dvdict = dvtypes4  # depracted: netcdf4 handles numpy dtypes
        # vtype = dvdict[str(var.dtype)]  # this is only needed for NETCFD3
        if vname == 'dday':
            ncvname = 'time'
        else:
            ncvname = vname
        vtype = var.dtype
        if vtype.type == np.float16:
            vtype = 'f4'
        # elif vtype == 'O' or isinstance(vtype.type, np.object): 2 is bad!
        elif vtype == 'O':
            vtype = var[0].dtype

        # dims = {1: ('transect',), 2: ('transect', 'nbins')}[var.ndim]
        # x = self.nf.createVariable(ncvname, vtype, dims, **varkw)

        if vname in twoD_vars_names:
            print(ncvname, vtype)
            vl_t = self.nf.groups['twoD_data'].createVLType(vtype, ncvname + '_type')
            x = self.nf.groups['twoD_data'].createVariable(ncvname, vl_t, ('transect',), **varkw)
        elif vname in oneD_vars_names:
            print(ncvname, vtype)
            vl_t = self.nf.groups['qc_diag_data'].createVLType(vtype, ncvname + '_type')
            x = self.nf.groups['qc_diag_data'].createVariable(ncvname, vl_t, ('transect',), **varkw)
        elif vname in txy_vars_names:
            print(ncvname, vtype)
            vl_t = self.nf.groups['nav_data'].createVLType(vtype, ncvname + '_type')
            x = self.nf.groups['nav_data'].createVariable(ncvname, vl_t, ('transect',), **varkw)
        elif vname in meta_vars_names:
            print(ncvname, vtype)
            x = self.nf.groups['metadata'].createVariable(ncvname, vtype, ('transect',), **varkw)

        # need to hanle missing values for the object variables:
        if np.ma.isMaskedArray(var):
            if (self.nf.data_model != 'NETCDF4'
                    and var.dtype.itemsize == 1
                    and var.dtype.kind == 'u'):
                var = np.ma.array(var, dtype=np.int16, fill_value=32767)
            x.missing_value = var.dtype.type(var.fill_value)
            x[:] = var
        else:
            if var.dtype == 'O':
                masked_els = np.array([np.ma.isMaskedArray(el) for el in var])
                if any(masked_els):
                    x.missing_value = var.dtype.type(var[masked_els][0].fill_value)
            x[:] = var

        # attributes:
        x.long_name = vdat[0]
        if vdat[1] is not None:
            x.units = vdat[1]
        x.C_format = vdat[2]

        if vname == 'depth':
            x.positive = "down"

        if vname == 'lon':
            x.standard_name = 'longitude'
        elif vname == 'lat':
            x.standard_name = 'latitude'
        elif vname == 'dday':
            x.standard_name = 'time'

        # handling data_min/max attribiute for non NETCDF4
        if self.nf.data_model != 'NETCDF4':
            if np.ma.count(var) == 0:
                x.data_min = x.data_max = x.missing_value
            else:
                if var.dtype.kind == 'f' and not np.ma.isMaskedArray(var):
                    x.data_min = var.dtype.type(np.nanmin(var))
                    x.data_max = var.dtype.type(np.nanmax(var))
                else:
                    x.data_min = var.dtype.type(var.min())
                    x.data_max = var.dtype.type(var.max())

    def make_it(self, filename, netcdf4=True, nc_compressed=False):
        '''
        dump the transect database into a netcdf file
        '''
        # create nc file
        self.create_nc(filename, netcdf4=netcdf4)

        if netcdf4:
            for v in meta_vars + oneD_vars:
                vname = v[0]
                vdat = v[1]
                avar = self.data[vname]
                varkw = dict()
                if nc_compressed:
                    varkw['zlib'] = True
                    if avar.dtype.kind == 'f':
                        lsd = 4 if vname == 'depth' else 3
                        varkw['least_significant_digit'] = lsd
                self.add_array_var_netcdf4(vname, avar, vdat, **varkw)
            for v in txy_vars + twoD_vars:
                vname = v[0]
                vdat = v[1]
                avar = np.array([ada[vname] for ada in self.data['seg_data']])
                varkw = dict()
                if nc_compressed:
                    varkw['zlib'] = True
                    if avar.dtype.kind == 'f':
                        lsd = 4 if vname == 'depth' else 3
                        varkw['least_significant_digit'] = lsd
                self.add_array_var_netcdf4(vname, avar, vdat, **varkw)
        else:
            raise NotImplementedError("No netCDF3 support yet!")

        self.nf.close()


def make_nc4(da_input, file_name, attrlist=None, nc_compressed=False):
    """
    Convenience function to make a transectsDB nc file.

    input: transect npz file or data structured array or transectsDB class obj
    file_name: netCDF output file
    cruise_id: netCDF global attribute
    inst_ping: netCDF global attribute
    """
    if attrlist is None:
        attrlist = []

    if isinstance(input, str):
        da_input = np.load(da_input, allow_pickle=True)['seg_data']
    a = transects_nc(da_input, attrlist=attrlist)
    a.make_it(file_name)


def main():
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument("inputfile", help='L1 transect npz file')
    parser.add_argument("outfile",
                        help="output netcdf file path (without '.nc')")

    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("--long",
    #                     help='write full CODAS DB dump instead of short form',
    #                     action="store_true")
    # group.add_argument("--compressed",
    #                     help='write netcdf4 with lossy compression',
    #                     action="store_true")
    # parser.add_argument("--dday_range",
    #                     help='colon-separated decimal day interval, only for\n'
    #                          '        short and compressed forms.\n'
    #                          '        E.g., "278.2:278.7" (no spaces).',
    #                     )
    parser.add_argument("--compressed",
                        help='write netcdf4 with lossy compression',
                        action="store_true")
    parser.add_argument("--configs",
                        nargs='+',
                        help='Parameter list used for L1 or L2 processing',
                        default=None,
                        type=str
                        )

    args = parser.parse_args()
    inputfile = args.inputfile
    outfile = args.outfile
    if not outfile.endswith('.nc'):
        outfile += '.nc'

    # if args.dday_range:
    #     ddrange = [float(s) for s in args.dday_range.split(':')]
    # else:
    #     ddrange = None

    if args.configs:
        run_configs = " ".join(args.configs)
        attrlist = [("run_configs", run_configs), ]
    else:
        attrlist = None

    if args.compressed:
        make_nc4(inputfile, outfile, attrlist=attrlist, nc_compressed=True)
    else:
        make_nc4(inputfile, outfile, attrlist=attrlist)

    log.info('test your file by running \n\n  ncdump -h %s\n\n' % (outfile))
