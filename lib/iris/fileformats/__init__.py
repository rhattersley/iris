# (C) British Crown Copyright 2010 - 2016, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
A package for converting cubes to and from specific file formats.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

from iris.io.format_picker import (FileExtension, FormatAgent,
                                   FormatSpecification, MagicNumber,
                                   UriProtocol, LeadingLine)
from . import abf
from . import um
try:
    from . import grib
except ImportError:
    grib = None
from . import name
from . import netcdf
from . import nimrod
from . import pp


__all__ = ['FORMAT_AGENT']


FORMAT_AGENT = FormatAgent()
FORMAT_AGENT.__doc__ = "The FORMAT_AGENT is responsible for identifying the " \
                       "format of a given URI. New formats can be added " \
                       "with the **add_spec** method."


#
# PP files.
#
FORMAT_AGENT.add_spec(FormatSpecification('UM Post Processing file (PP)',
                                          MagicNumber(4),
                                          0x00000100,
                                          pp.load_cubes,
                                          priority=5,
                                          constraint_aware_handler=True))


FORMAT_AGENT.add_spec(
    FormatSpecification('UM Post Processing file (PP) little-endian',
                        MagicNumber(4),
                        0x00010000,
                        pp.load_cubes_little_endian,
                        priority=3,
                        constraint_aware_handler=True))


#
# GRIB files.
#
def _load_grib(*args, **kwargs):
    if grib is None:
        raise RuntimeError('Unable to load GRIB file - the ECMWF '
                           '`gribapi` package is not installed.')
    return grib.load_cubes(*args, **kwargs)


# NB. Because this is such a "fuzzy" check, we give this a very low
# priority to avoid collateral damage from false positives.
FORMAT_AGENT.add_spec(
    FormatSpecification('GRIB', MagicNumber(100),
                        lambda header_bytes: b'GRIB' in header_bytes,
                        _load_grib, priority=1))


#
# netCDF files.
#
FORMAT_AGENT.add_spec(FormatSpecification('NetCDF',
                                          MagicNumber(4),
                                          0x43444601,
                                          netcdf.load_cubes,
                                          priority=5))


FORMAT_AGENT.add_spec(FormatSpecification('NetCDF 64 bit offset format',
                                          MagicNumber(4),
                                          0x43444602,
                                          netcdf.load_cubes,
                                          priority=5))


# This covers both v4 and v4 classic model.
FORMAT_AGENT.add_spec(FormatSpecification('NetCDF_v4',
                                          MagicNumber(8),
                                          0x894844460D0A1A0A,
                                          netcdf.load_cubes,
                                          priority=5))


_nc_dap = FormatSpecification('NetCDF OPeNDAP',
                              UriProtocol(),
                              lambda protocol: protocol in ['http', 'https'],
                              netcdf.load_cubes,
                              priority=6)
FORMAT_AGENT.add_spec(_nc_dap)
del _nc_dap

#
# UM Fieldsfiles.
#
FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) pre v3.1',
                                          MagicNumber(8),
                                          0x000000000000000F,
                                          um.load_cubes,
                                          priority=3,
                                          constraint_aware_handler=True))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) post v5.2',
                                          MagicNumber(8),
                                          0x0000000000000014,
                                          um.load_cubes,
                                          priority=4,
                                          constraint_aware_handler=True))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) ancillary',
                                          MagicNumber(8),
                                          0xFFFFFFFFFFFF8000,
                                          um.load_cubes,
                                          priority=3,
                                          constraint_aware_handler=True))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) converted '
                                          'with ieee to 32 bit',
                                          MagicNumber(4),
                                          0x00000014,
                                          um.load_cubes_32bit_ieee,
                                          priority=3,
                                          constraint_aware_handler=True))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) ancillary '
                                          'converted with ieee to 32 bit',
                                          MagicNumber(4),
                                          0xFFFF8000,
                                          um.load_cubes_32bit_ieee,
                                          priority=3,
                                          constraint_aware_handler=True))


#
# NIMROD files.
#
def _nimrod_load(*args, **kwargs):
    # A simple wrapper for `iris.fileformats.nimrod.load_cubes` which
    # allows the loader to be registered without triggering the import
    # of `iris.fileformats.nimrod`.
    from .nimrod import load_cubes
    return load_cubes(*args, **kwargs)

FORMAT_AGENT.add_spec(FormatSpecification('NIMROD',
                                          MagicNumber(4),
                                          0x00000200,
                                          #nimrod.load_cubes,
                                          _nimrod_load,
                                          priority=3))

#
# NAME files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification('NAME III',
                        LeadingLine(),
                        lambda line: line.lstrip().startswith(b"NAME III"),
                        name.load_cubes,
                        priority=5))


#
# ABF/ABL
#
FORMAT_AGENT.add_spec(FormatSpecification('ABF', FileExtension(), '.abf',
                                          abf.load_cubes, priority=3))


FORMAT_AGENT.add_spec(FormatSpecification('ABL', FileExtension(), '.abl',
                                          abf.load_cubes, priority=3))
