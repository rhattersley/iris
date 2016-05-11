# (C) British Crown Copyright 2016, Met Office
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
"""Unit tests for :func:`iris.sample_data_path` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import os
import os.path
import shutil
import tempfile

import mock

from iris import sample_data_path


class Test(tests.IrisTest):
    def test_file_ok(self):
        try:
            sample_dir = tempfile.mkdtemp()
            sample_handle, sample_path = tempfile.mkstemp(dir=sample_dir)
            os.close(sample_handle)
            with mock.patch('iris.config.SAMPLE_DATA_DIR', sample_dir):
                result = sample_data_path(os.path.basename(sample_path))
            self.assertEqual(result, sample_path)
        finally:
            shutil.rmtree(sample_dir)

    def test_file_not_found(self):
        try:
            sample_dir = tempfile.mkdtemp()
            with mock.patch('iris.config.SAMPLE_DATA_DIR', sample_dir):
                with self.assertRaisesRegexp(ValueError,
                                             "Sample data .* not found"):
                    sample_data_path('foo')
        finally:
            shutil.rmtree(sample_dir)

    def test_glob_ok(self):
        try:
            sample_dir = tempfile.mkdtemp()
            sample_handle, sample_path = tempfile.mkstemp(dir=sample_dir)
            os.close(sample_handle)
            with mock.patch('iris.config.SAMPLE_DATA_DIR', sample_dir):
                sample_glob = '?' + os.path.basename(sample_path)[1:]
                result = sample_data_path(sample_glob)
            self.assertEqual(result, os.path.join(sample_dir, sample_glob))
        finally:
            shutil.rmtree(sample_dir)

    def test_glob_not_found(self):
        try:
            sample_dir = tempfile.mkdtemp()
            with mock.patch('iris.config.SAMPLE_DATA_DIR', sample_dir):
                with self.assertRaisesRegexp(ValueError,
                                             "Sample data .* not found"):
                    sample_data_path('foo.*')
        finally:
            shutil.rmtree(sample_dir)


if __name__ == '__main__':
    tests.main()
