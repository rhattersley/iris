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
"""
Integration tests for time handling.

"""
# Import `iris.tests` first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import copy
from datetime import timedelta

import numpy as np

from iris.time import Time360, time360, timedelta_dtype


class TestTime360(tests.IrisTest):
    def test_copy(self):
        t = Time360(2013, 6, 26)
        self.assertEqual(t, copy(t))


class TestTimedeltaArray(tests.IrisTest):
    def test_auto_dtype_scalar(self):
        # Check that an array of datetime.timedelta objects is given
        # the `timedelta_dtype` dtype.
        delta = timedelta(1)
        a = np.array(delta)
        self.assertEqual(a.dtype, timedelta_dtype)

    def test_auto_dtype_list(self):
        # Check that an array of datetime.timedelta objects is given
        # the `timedelta_dtype` dtype.
        delta = timedelta(1)
        a = np.array([delta])
        self.assertEqual(a.dtype, timedelta_dtype)

    def test_zeros(self):
        n = 3
        zero_delta = timedelta()
        a = np.zeros(n, dtype=timedelta_dtype)
        for delta in a:
            self.assertEqual(delta, zero_delta)

    def test_getitem(self):
        delta = timedelta(1)
        a = np.array([delta], dtype=timedelta_dtype)
        self.assertIsInstance(a[0], timedelta)
        self.assertEqual(a[0], delta)

    def test_setitem(self):
        delta = timedelta(1)
        a = np.zeros(3, dtype=timedelta_dtype)
        a[1] = delta
        self.assertNotEqual(a[0], delta)
        self.assertEqual(a[1], delta)
        self.assertNotEqual(a[2], delta)

    def test_argmax(self):
        a = np.array([timedelta(3, 12, 100), timedelta(0, 15, 4), timedelta(-5),
                      timedelta(2, 4), timedelta(0, 1)], dtype=timedelta_dtype)
        self.assertEqual(np.argmax(a), 0)

        a = np.array([timedelta(2), timedelta(0), timedelta(-5),
                      timedelta(2, 4), timedelta(0, 1)], dtype=timedelta_dtype)
        self.assertEqual(np.argmax(a), 3)

        a = np.array([timedelta(-2), timedelta(0), timedelta(-5)],
                     dtype=timedelta_dtype)
        self.assertEqual(np.argmax(a), 1)

        a = np.array([timedelta(-2), timedelta(-1), timedelta(-5)],
                     dtype=timedelta_dtype)
        self.assertEqual(np.argmax(a), 1)

    def test_sign(self):
        a = np.array([timedelta(2), timedelta(0), timedelta(-5),
                      timedelta(2, 4), timedelta(0, 1)], dtype=timedelta_dtype)
        self.assertArrayEqual(np.sign(a), np.array([1, 0, -1, 1, 1]))

    def XXX_test_subtract(self):
        a = np.array([Time360(2013, 6, 20), Time360(2013, 6, 24),
                      Time360(2014, 2, 30)], dtype=time360)
        r = a[1:] - a[:-1]
        self.assertArrayEqual(r, np.array([timedelta(4), timedelta(246)]))


class TestTime360Array(tests.IrisTest):
    def test_auto_dtype_scalar(self):
        # Check that an array of Time360 objects is given the `time360`
        # dtype.
        t = Time360(2013, 6, 20)
        a = np.array(t)
        self.assertEqual(a.dtype, time360)

    def test_auto_dtype_list(self):
        # Check that an array of Time360 objects is given the `time360`
        # dtype.
        t = Time360(2013, 6, 20)
        a = np.array([t])
        self.assertEqual(a.dtype, time360)

    def test_getitem(self):
        t = Time360(2013, 6, 20)
        a = np.array([t], dtype=time360)
        self.assertIsInstance(a[0], Time360)
        self.assertEqual(a[0], t)

    def test_subtract(self):
        a = np.array([Time360(2013, 6, 20), Time360(2013, 6, 24),
                      Time360(2014, 2, 30)], dtype=time360)
        r = a[1:] - a[:-1]
        self.assertArrayEqual(r, np.array([timedelta(4), timedelta(246)],
                                          dtype=timedelta_dtype))


if __name__ == "__main__":
    tests.main()
