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
Unit tests for the `iris.time.Time360` class.

"""
# Import `iris.tests` first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from datetime import datetime, timedelta

import numpy as np

from iris.time import Time360


# TODO: Add, subtract, compare, hash, attribute get/set.


class Test_repr(tests.IrisTest):
    def test_ymd(self):
        self.assertEqual(repr(Time360(2013, 6, 20)),
                         'iris.time.Time360(2013, 6, 20)')

    def test_ymdHM(self):
        self.assertEqual(repr(Time360(2013, 6, 20, 9, 34)),
                         'iris.time.Time360(2013, 6, 20, 9, 34)')

    def test_ymdHMS(self):
        self.assertEqual(repr(Time360(2013, 6, 20, 9, 34, 10)),
                         'iris.time.Time360(2013, 6, 20, 9, 34, 10)')

    def test_ymdHMSf(self):
        self.assertEqual(repr(Time360(2013, 6, 20, 9, 34, 10, 123)),
                         'iris.time.Time360(2013, 6, 20, 9, 34, 10, 123)')


class Test_str(tests.IrisTest):
    def test_ymd(self):
        self.assertEqual(str(Time360(2013, 6, 20)), '2013-06-20 00:00')

    def test_ymdHM(self):
        self.assertEqual(str(Time360(2013, 6, 20, 9, 34)), '2013-06-20 09:34')

    def test_ymdHMS(self):
        self.assertEqual(str(Time360(2013, 6, 20, 9, 34, 10)),
                         '2013-06-20 09:34:10')

    def test_ymdHMSf(self):
        self.assertEqual(str(Time360(2013, 6, 20, 9, 34, 10, 123)),
                         '2013-06-20 09:34:10.000123')


class Test_cmp(tests.IrisTest):
    def _test(self, ta, tb, less, equal):
        self.assertEqual(ta < tb, less)
        self.assertEqual(ta <= tb, less or equal)
        self.assertEqual(ta == tb, equal)
        self.assertEqual(ta >= tb, not less)
        self.assertEqual(ta > tb, not(less or equal))

    def test_pos(self):
        t1 = Time360(2013, 6, 24)
        t1_b = Time360(2013, 6, 24)
        t2 = Time360(2013, 6, 25)
        # Deliberately choose a year value whose little-endian
        # representation will flush out problems with a naive
        # byte-by-byte comparison.
        t3 = Time360(2053, 6, 25)

        self._test(t1, t1, False, True)
        self._test(t1, t1_b, False, True)
        self._test(t1, t2, True, False)
        self._test(t2, t1, False, False)
        self._test(t1, t3, True, False)
        self._test(t3, t1, False, False)

    def test_neg_pos(self):
        # Deliberately choose a negative value whose little-endian,
        # twos-complement representation starts with a byte which is
        # less than the first byte of the positive value.
        t1 = Time360(-2054, 6, 24)
        t2 = Time360(2013, 6, 24)

        self._test(t1, t2, True, False)
        self._test(t2, t1, False, False)

    def test_neg(self):
        # Deliberately choose two year values whose little-endian
        # representations will flush out problems with a naive
        # byte-by-byte comparison.
        t1 = Time360(-2054, 6, 24)
        t2 = Time360(-2013, 6, 24)

        self._test(t1, t2, True, False)
        self._test(t2, t1, False, False)

    def test_cmp_calendar_specific(self):
        t_360 = Time360(2013, 6, 24)
        t_std = datetime(2013, 6, 24)

        self.assertNotEqual(t_360, t_std)
        self.assertNotEqual(t_std, t_360)

        with self.assertRaises(TypeError):
            t_360 < t_std
        with self.assertRaises(TypeError):
            t_std < t_360

        with self.assertRaises(TypeError):
            t_360 <= t_std
        with self.assertRaises(TypeError):
            t_std <= t_360

        with self.assertRaises(TypeError):
            t_360 >= t_std
        with self.assertRaises(TypeError):
            t_std >= t_360

        with self.assertRaises(TypeError):
            t_360 > t_std
        with self.assertRaises(TypeError):
            t_std > t_360


class Test_add(tests.IrisTest):
    def test_simple(self):
        t_360 = Time360(2013, 6, 26)
        self.assertEqual(Time360(2013, 6, 26) + timedelta(1),
                         Time360(2013, 6, 27))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(0, 1),
                         Time360(2013, 6, 26, 0, 0, 1))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(0, 0, 1),
                         Time360(2013, 6, 26, 0, 0, 0, 1))

    def test_wrap_days(self):
        self.assertEqual(Time360(2013, 6, 26) + timedelta(30),
                         Time360(2013, 7, 26))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(360),
                         Time360(2014, 6, 26))

        self.assertEqual(Time360(2013, 6, 26) + timedelta(6),
                         Time360(2013, 7, 2))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(246),
                         Time360(2014, 3, 2))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(606),
                         Time360(2015, 3, 2))

    def test_wrap_seconds(self):
        self.assertEqual(Time360(2013, 6, 26) + timedelta(0, 60),
                         Time360(2013, 6, 26, 0, 1))
        self.assertEqual(Time360(2013, 6, 26) + timedelta(0, 3600),
                         Time360(2013, 6, 26, 1))

        self.assertEqual(Time360(2013, 12, 30, 23, 59, 59) + timedelta(0, 1),
                         Time360(2014, 1, 1))

    def test_wrap_microseconds(self):
        self.assertEqual(Time360(2013, 12, 30, 23, 59, 59, 999999) +
                            timedelta(0, 0, 1),
                         Time360(2014, 1, 1))

    def test_invalid_types(self):
        t_360 = Time360(2013, 6, 25)
        t_std = datetime(2013, 6, 25)
        with self.assertRaises(TypeError):
            t_360 + t_std
        with self.assertRaises(TypeError):
            t_std + t_360


class Test_sub(tests.IrisTest):
    def _test(self, t1, t2, expected):
        r = t2 - t1
        self.assertIsInstance(r, timedelta)
        self.assertEqual(r, expected)

        r = t1 - t2
        self.assertIsInstance(r, timedelta)
        self.assertEqual(r, -expected)

    def test_equal(self):
        t1 = Time360(2013, 6, 24)
        t2 = Time360(2013, 6, 24)
        self._test(t1, t2, timedelta())

    def test_single_year(self):
        t1 = Time360(2013, 6, 24)
        t2 = Time360(2014, 6, 24)
        self._test(t1, t2, timedelta(360))

    def test_single_month(self):
        t1 = Time360(2013, 6, 24)
        t2 = Time360(2013, 7, 24)
        self._test(t1, t2, timedelta(30))

    def test_single_day(self):
        t1 = Time360(2013, 6, 24)
        t2 = Time360(2013, 6, 25)
        self._test(t1, t2, timedelta(1))

    def test_single_hour(self):
        t1 = Time360(2013, 6, 24, 13)
        t2 = Time360(2013, 6, 24, 14)
        self._test(t1, t2, timedelta(hours=1))

    def test_single_minute(self):
        t1 = Time360(2013, 6, 24, 13, 22)
        t2 = Time360(2013, 6, 24, 13, 23)
        self._test(t1, t2, timedelta(minutes=1))

    def test_single_second(self):
        t1 = Time360(2013, 6, 24, 13, 22, 26)
        t2 = Time360(2013, 6, 24, 13, 22, 27)
        self._test(t1, t2, timedelta(seconds=1))

    def test_mixed_calendar(self):
        t_360 = Time360(2013, 6, 25)
        t_std = datetime(2013, 6, 25)
        with self.assertRaises(TypeError):
            t_360 - t_std
        with self.assertRaises(TypeError):
            t_std - t_360

if __name__ == "__main__":
    tests.main()
