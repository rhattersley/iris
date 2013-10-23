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
"""Unit tests for iris.fileformats.grib_save_rules"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import unittest
import warnings

try:
    import gribapi
except ImportError:
    gribapi = None
import mock
import numpy as np
import numpy.ma as ma

import iris.cube
import iris.coords
if gribapi is not None:
    import iris.fileformats.grib.grib_save_rules as grib_save_rules


@unittest.skipIf(gribapi is None, 'The "gribapi" module is not available.')
class Test_non_hybrid_surfaces(tests.IrisTest):
    # Test grib_save_rules.non_hybrid_surfaces()

    @mock.patch.object(gribapi, "grib_set_long")
    def test_altitude_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1,2,3,4,5])
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "altitude", units="m"))

        grib_save_rules.non_hybrid_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 102)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_height_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1,2,3,4,5])
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "height", units="m"))

        grib_save_rules.non_hybrid_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 103)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_no_vertical(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1,2,3,4,5])
        grib_save_rules.non_hybrid_surfaces(cube, grib)
        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)


@unittest.skipIf(gribapi is None, 'The "gribapi" module is not available.')
class Test_data(tests.IrisTest):
    # Test grib_save_rules.data()

    @mock.patch.object(gribapi, "grib_set_double_array")
    @mock.patch.object(gribapi, "grib_set_double")
    @mock.patch.object(gribapi, "grib_set")
    def test_masked_array(self, mock_set, mock_set_double, grib_set_double_array):
        grib = None
        cube = iris.cube.Cube(ma.MaskedArray([1,2,3,4,5], fill_value=54321))

        grib_save_rules.data(cube, grib)

        mock_set_double.assert_any_call(grib, "missingValue", float(54321))

    @mock.patch.object(gribapi, "grib_set_double_array")
    @mock.patch.object(gribapi, "grib_set_double")
    def test_numpy_array(self, mock_set_double, grib_set_double_array):
        grib = None
        cube = iris.cube.Cube(np.array([1,2,3,4,5]))

        grib_save_rules.data(cube, grib)

        mock_set_double.assert_any_call(grib, "missingValue", float(-1e9))

    @mock.patch.object(gribapi, "grib_set_double_array")
    @mock.patch.object(gribapi, "grib_set_double")
    def test_scaling(self, mock_set_double, mock_set_double_array):
        # Show that data type known to be stored as %ge gets scaled
        grib = None
        cube = iris.cube.Cube(np.array([0.0, 0.25, 1.0]),
                              standard_name='cloud_area_fraction',
                              units=iris.unit.Unit('0.5'))
        grib_save_rules.data(cube, grib)
        callargs = mock_set_double_array.call_args_list
        self.assertEqual(len(callargs), 1)
        self.assertEqual(callargs[0][0][:2], (None, "values"))
        self.assertArrayAlmostEqual(callargs[0][0][2],
                                    np.array([0.0, 12.5, 50.0]))


@unittest.skipIf(gribapi is None, 'The "gribapi" module is not available.')
class Test_phenomenon(tests.IrisTest):
    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_unknown(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        # Force reset of warnings registry to avoid suppression of
        # repeated warnings. warnings.resetwarnings() does not do this.
        if hasattr(grib_save_rules, '__warningregistry__'):
            grib_save_rules.__warningregistry__.clear()
        with warnings.catch_warnings():
            # This should issue a warning about unrecognised data
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                grib_save_rules.param_code(cube, grib)
        # do it all again, and this time check the results
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        grib_save_rules.param_code(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 0)
        mock_set_long.assert_any_call(grib, "parameterCategory", 0)
        mock_set_long.assert_any_call(grib, "parameterNumber", 0)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_known_standard_name(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]),
                              standard_name='sea_surface_temperature')
        grib_save_rules.param_code(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 10)
        mock_set_long.assert_any_call(grib, "parameterCategory", 3)
        mock_set_long.assert_any_call(grib, "parameterNumber", 0)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_known_long_name(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]),
                              long_name='cloud_mixing_ratio')
        grib_save_rules.param_code(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 0)
        mock_set_long.assert_any_call(grib, "parameterCategory", 1)
        mock_set_long.assert_any_call(grib, "parameterNumber", 22)


@unittest.skipIf(gribapi is None, 'The "gribapi" module is not available.')
class Test_type_of_statistical_processing(tests.IrisTest):
    @mock.patch.object(gribapi, "grib_set_long")
    def test_stats_type_min(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        time_unit = iris.unit.Unit('hours since 1970-01-01 00:00:00')
        time_coord = iris.coords.DimCoord([0.0],
                                          bounds=[0.0, 1],
                                          standard_name='time',
                                          units=time_unit)
        cube.add_aux_coord(time_coord, ())
        cube.add_cell_method(iris.coords.CellMethod('maximum', time_coord))
        grib_save_rules.type_of_statistical_processing(cube, grib, time_coord)
        mock_set_long.assert_any_call(grib, "typeOfStatisticalProcessing", 2)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_stats_type_max(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        time_unit = iris.unit.Unit('hours since 1970-01-01 00:00:00')
        time_coord = iris.coords.DimCoord([0.0],
                                          bounds=[0.0, 1],
                                          standard_name='time',
                                          units=time_unit)
        cube.add_aux_coord(time_coord, ())
        cube.add_cell_method(iris.coords.CellMethod('minimum', time_coord))
        grib_save_rules.type_of_statistical_processing(cube, grib, time_coord)
        mock_set_long.assert_any_call(grib, "typeOfStatisticalProcessing", 3)


if __name__ == "__main__":
    tests.main()
