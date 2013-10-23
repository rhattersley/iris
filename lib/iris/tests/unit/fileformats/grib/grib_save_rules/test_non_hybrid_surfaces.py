# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for module-level functions."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest

try:
    import gribapi
except ImportError:
    gribapi = None
import numpy as np

import iris
import iris.cube
import iris.coords
if gribapi is not None:
    from iris.fileformats.grib.grib_save_rules import non_hybrid_surfaces


@unittest.skipIf(gribapi is None, 'The "gribapi" module is not available.')
class Test_non_hybrid_surfaces(tests.IrisTest):
    def test_bounded_altitude_feet(self):
        cube = iris.cube.Cube([0])
        cube.add_aux_coord(iris.coords.AuxCoord(
            1500.0, long_name='altitude', units='ft',
            bounds=np.array([1000.0, 2000.0])))
        grib = gribapi.grib_new_from_samples("GRIB2")
        non_hybrid_surfaces(cube, grib)
        self.assertEqual(
            gribapi.grib_get_double(grib, "scaledValueOfFirstFixedSurface"),
            304.0)
        self.assertEqual(
            gribapi.grib_get_double(grib, "scaledValueOfSecondFixedSurface"),
            609.0)


if __name__ == "__main__":
    tests.main()
