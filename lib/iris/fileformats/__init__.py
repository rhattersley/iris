# (C) British Crown Copyright 2010 - 2013, Met Office
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

from iris.io.format_picker import (FileExtension, FormatAgent,
                                   FormatSpecification, MagicNumber,
                                   UriProtocol, LeadingLine)
import abf
import ff
import name
import nimrod
import pp


__all__ = ['FORMAT_AGENT']


def _pp_little_endian(filename, *args, **kwargs):
    msg = 'PP file {!r} contains little-endian data, ' \
          'please convert to big-endian with command line utility "bigend".'
    raise ValueError(msg.format(filename))


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
                                          priority=5))


FORMAT_AGENT.add_spec(
    FormatSpecification('UM Post Processing file (PP) little-endian',
                        MagicNumber(4),
                        0x00010000,
                        _pp_little_endian,
                        priority=3))


#
# GRIB files.
#
def _grib(filename, callback):
    import iris.fileformats.grib
    iris.fileformats.grib.load_cubes(filename, callback)

FORMAT_AGENT.add_spec(FormatSpecification(
    'GRIB', MagicNumber(4), 0x47524942, _grib, priority=5))

FORMAT_AGENT.add_spec(FormatSpecification(
    'WMO GRIB Bulletin (21 bytes)', MagicNumber(4, offset=21), 0x47524942,
    _grib, priority=3))

FORMAT_AGENT.add_spec(FormatSpecification(
    'WMO GRIB Bulletin (42 bytes)', MagicNumber(4, offset=42), 0x47524942,
    _grib, priority=3))


#
# netCDF files.
#
def _netcdf(filename, callback):
    import iris.fileformats.netcdf
    iris.fileformats.netcdf.load_cubes(filename, callback)

FORMAT_AGENT.add_spec(FormatSpecification(
    'NetCDF', MagicNumber(4), 0x43444601, _netcdf, priority=5))

FORMAT_AGENT.add_spec(FormatSpecification(
    'NetCDF 64 bit offset format', MagicNumber(4), 0x43444602, _netcdf,
    priority=5))

# This covers both v4 and v4 classic model.
FORMAT_AGENT.add_spec(FormatSpecification(
    'NetCDF_v4', MagicNumber(8), 0x894844460D0A1A0A, _netcdf, priority=5))

FORMAT_AGENT.add_spec(FormatSpecification(
    'NetCDF OPeNDAP', UriProtocol(),
    lambda protocol: protocol in ['http', 'https'],
    _netcdf, priority=6))


#
# UM Fieldsfiles.
#
FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) pre v3.1',
                                          MagicNumber(8),
                                          0x000000000000000F,
                                          ff.load_cubes,
                                          priority=3))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) post v5.2',
                                          MagicNumber(8),
                                          0x0000000000000014,
                                          ff.load_cubes,
                                          priority=4))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) ancillary',
                                          MagicNumber(8),
                                          0xFFFFFFFFFFFF8000,
                                          ff.load_cubes,
                                          priority=3))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) converted '
                                          'with ieee to 32 bit',
                                          MagicNumber(4),
                                          0x00000014,
                                          ff.load_cubes_32bit_ieee,
                                          priority=3))


FORMAT_AGENT.add_spec(FormatSpecification('UM Fieldsfile (FF) ancillary '
                                          'converted with ieee to 32 bit',
                                          MagicNumber(4),
                                          0xFFFF8000,
                                          ff.load_cubes_32bit_ieee,
                                          priority=3))


#
# NIMROD files.
#
FORMAT_AGENT.add_spec(FormatSpecification('NIMROD',
                                          MagicNumber(4),
                                          0x00000200,
                                          nimrod.load_cubes,
                                          priority=3))

#
# NAME files.
#
FORMAT_AGENT.add_spec(
    FormatSpecification('NAME III',
                        LeadingLine(),
                        lambda line: line.lstrip().startswith("NAME III"),
                        name.load_cubes,
                        priority=5))


#
# ABF/ABL
#
FORMAT_AGENT.add_spec(FormatSpecification('ABF', FileExtension(), '.abf',
                                          abf.load_cubes, priority=3))


FORMAT_AGENT.add_spec(FormatSpecification('ABL', FileExtension(), '.abl',
                                          abf.load_cubes, priority=3))
