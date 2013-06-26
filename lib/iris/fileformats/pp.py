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
Provides UK Met Office Post Process (PP) format specific capabilities.

"""

import abc
import collections
from copy import deepcopy
import itertools
import operator
import os
import re
import struct

import numpy as np
import numpy.ma as ma
import netcdftime

import iris.config
import iris.fileformats.rules
import iris.io
import iris.time
import iris.unit
import iris.fileformats.manager
import iris.coord_systems
import iris.proxy
iris.proxy.apply_proxy('iris.fileformats.pp_packing', globals())


__all__ = ['load', 'save', 'PPField', 'add_load_rules', 'reset_load_rules', 
           'add_save_rules', 'reset_save_rules', 'STASH', 'EARTH_RADIUS']


EARTH_RADIUS = 6371229.0


# PP->Cube and Cube->PP rules are loaded on first use
_load_rules = None
_cross_reference_rules = None
_save_rules = None


PP_HEADER_DEPTH = 256
PP_WORD_DEPTH = 4
NUM_LONG_HEADERS = 45
NUM_FLOAT_HEADERS = 19

# The header definition for header release 2.
UM_HEADER_2 = [
        ('lbyr',   (1, )),
        ('lbmon',  (2, )),
        ('lbdat',  (3, )),
        ('lbhr',   (4, )),
        ('lbmin',  (5, )),
        ('lbday',  (6, )),
        ('lbyrd',  (7, )),
        ('lbmond', (8, )),
        ('lbdatd', (9, )),
        ('lbhrd',  (10, )),
        ('lbmind', (11, )),
        ('lbdayd', (12, )),
        ('lbtim',  (13, )),
        ('lbft',   (14, )),
        ('lblrec', (15, )),
        ('lbcode', (16, )),
        ('lbhem',  (17, )),
        ('lbrow',  (18, )),
        ('lbnpt',  (19, )),
        ('lbext',  (20, )),
        ('lbpack', (21, )),
        ('lbrel',  (22, )),
        ('lbfc',   (23, )),
        ('lbcfc',  (24, )),
        ('lbproc', (25, )),
        ('lbvc',   (26, )),
        ('lbrvc',  (27, )),
        ('lbexp',  (28, )),
        ('lbegin', (29, )),
        ('lbnrec', (30, )),
        ('lbproj', (31, )),
        ('lbtyp',  (32, )),
        ('lblev',  (33, )),
        ('lbrsvd', (34, 35, 36, 37, )),
        ('lbsrce', (38, )),
        ('lbuser', (39, 40, 41, 42, 43, 44, 45, )),
        ('brsvd',  (46, 47, 48, 49, )),
        ('bdatum', (50, )),
        ('bacc',   (51, )),
        ('blev',   (52, )),
        ('brlev',  (53, )),
        ('bhlev',  (54, )),
        ('bhrlev', (55, )),
        ('bplat',  (56, )),
        ('bplon',  (57, )),
        ('bgor',   (58, )),
        ('bzy',    (59, )),
        ('bdy',    (60, )),
        ('bzx',    (61, )),
        ('bdx',    (62, )),
        ('bmdi',   (63, )),
        ('bmks',   (64, )),
    ]
"""A list of (header_name, position_in_header(tuple of)) pairs for header release 2 - using the one-based UM/FORTRAN indexing convention."""

# The header definition for header release 3.
UM_HEADER_3 = [
        ('lbyr',   (1, )),
        ('lbmon',  (2, )),
        ('lbdat',  (3, )),
        ('lbhr',   (4, )),
        ('lbmin',  (5, )),
        ('lbsec',  (6, )),
        ('lbyrd',  (7, )),
        ('lbmond', (8, )),
        ('lbdatd', (9, )),
        ('lbhrd',  (10, )),
        ('lbmind', (11, )),
        ('lbsecd', (12, )),
        ('lbtim',  (13, )),
        ('lbft',   (14, )),
        ('lblrec', (15, )),
        ('lbcode', (16, )),
        ('lbhem',  (17, )),
        ('lbrow',  (18, )),
        ('lbnpt',  (19, )),
        ('lbext',  (20, )),
        ('lbpack', (21, )),
        ('lbrel',  (22, )),
        ('lbfc',   (23, )),
        ('lbcfc',  (24, )),
        ('lbproc', (25, )),
        ('lbvc',   (26, )),
        ('lbrvc',  (27, )),
        ('lbexp',  (28, )),
        ('lbegin', (29, )),
        ('lbnrec', (30, )),
        ('lbproj', (31, )),
        ('lbtyp',  (32, )),
        ('lblev',  (33, )),
        ('lbrsvd', (34, 35, 36, 37, )),
        ('lbsrce', (38, )),
        ('lbuser', (39, 40, 41, 42, 43, 44, 45, )),
        ('brsvd',  (46, 47, 48, 49, )),
        ('bdatum', (50, )),
        ('bacc',   (51, )),
        ('blev',   (52, )),
        ('brlev',  (53, )),
        ('bhlev',  (54, )),
        ('bhrlev', (55, )),
        ('bplat',  (56, )),
        ('bplon',  (57, )),
        ('bgor',   (58, )),
        ('bzy',    (59, )),
        ('bdy',    (60, )),
        ('bzx',    (61, )),
        ('bdx',    (62, )),
        ('bmdi',   (63, )),
        ('bmks',   (64, )),
    ]
"""A list of (header_name, position_in_header(tuple of)) pairs for header release 3 - using the one-based UM/FORTRAN indexing convention."""

# A map from header-release-number to header definition
UM_HEADERS = {2: UM_HEADER_2, 3: UM_HEADER_3}

# Offset value to convert from UM_HEADER positions to PP_HEADER offsets.
UM_TO_PP_HEADER_OFFSET = 1

EXTRA_DATA = {
                 1: 'x',
                 2: 'y',
                 3: 'lower_y_domain',
                 4: 'lower_x_domain',
                 5: 'upper_y_domain',
                 6: 'upper_x_domain',
                 7: 'lower_z_domain',
                 8: 'upper_z_domain',
                 10: 'field_title',
                 11: 'domain_title',
                 12: 'x_lower_bound',
                 13: 'x_upper_bound',
                 14: 'y_lower_bound',
                 15: 'y_upper_bound',
             }
"""A dictionary mapping IB values to their names."""


LBUSER_DTYPE_LOOKUP = {1 :np.dtype('>f4'), 
                       2 :np.dtype('>i4'), 
                       3 :np.dtype('>i4'),
                       -1:np.dtype('>f4'),
                       -2:np.dtype('>i4'),
                       -3:np.dtype('>i4'),
                       'default': np.dtype('>f4'),
                       }
"""Maps lbuser[0] to numpy data type. "default" will be interpreted if no match is found, providing a warning in such a case.""" 

# LBPROC codes and their English equivalents
LBPROC_PAIRS = ((1, "Difference from another experiment"),
                (2, "Difference from zonal (or other spatial) mean"),
                (4, "Difference from time mean"),
                (8, "X-derivative (d/dx)"),
                (16, "Y-derivative (d/dy)"),
                (32, "Time derivative (d/dt)"),
                (64, "Zonal mean field"),
                (128, "Time mean field"),
                (256, "Product of two fields"),
                (512, "Square root of a field"),
                (1024, "Difference between fields at levels BLEV and BRLEV"),
                (2048, "Mean over layer between levels BLEV and BRLEV"),
                (4096, "Minimum value of field during time period"),
                (8192, "Maximum value of field during time period"),
                (16384, "Magnitude of a vector, not specifically wind speed"),
                (32768, "Log10 of a field"),
                (65536, "Variance of a field"),
                (131072, "Mean over an ensemble of parallel runs"))

# lbproc_map is dict mapping lbproc->English and English->lbproc essentially a one to one mapping
lbproc_map = {x : y for x,y in itertools.chain(LBPROC_PAIRS, ((y,x) for x,y in LBPROC_PAIRS))}


class STASH(collections.namedtuple('STASH', 'model section item')):
    """
    A class to hold a single STASH code.

    Create instances using:
        >>> model = 1
        >>> section = 2
        >>> item = 3
        >>> my_stash = iris.fileformats.pp.STASH(model, section, item)

    Access the sub-components via:
        >>> my_stash.model
        1
        >>> my_stash.section
        2
        >>> my_stash.item
        3

    String conversion results in the MSI format:
        >>> print iris.fileformats.pp.STASH(1, 16, 203)
        m01s16i203

    """

    __slots__ = ()

    def __new__(cls, model, section, item):
        """

        Args:
        
        * model
            A positive integer less than 100, or None.
        * section
            A non-negative integer less than 100, or None.
        * item
            A positive integer less than 1000, or None.

        """
        model = cls._validate_member('model', model, 1, 99)
        section = cls._validate_member('section', section, 0, 99)
        item = cls._validate_member('item', item, 1, 999)
        return super(STASH, cls).__new__(cls, model, section, item)

    @staticmethod
    def from_msi(msi):
        """Convert a STASH code MSI string to a STASH instance."""
        if not isinstance(msi, basestring):
            raise TypeError('Expected STASH code MSI string, got %r' % msi)

        msi_match = re.match('^\s*m(.*)s(.*)i(.*)\s*$', msi, re.IGNORECASE)

        if msi_match is None:
            raise ValueError('Expected STASH code MSI string "mXXsXXiXXX", got %r' % msi)

        return STASH(*msi_match.groups())

    @staticmethod
    def _validate_member(name, value, lower_limit, upper_limit):
        # Returns a valid integer or None.
        try:
            value = int(value)
            if not lower_limit <= value <= upper_limit:
                value = None
        except (TypeError, ValueError):
            value = None
        return value

    def __str__(self):
        model = self._format_member(self.model, 2)
        section = self._format_member(self.section, 2)
        item = self._format_member(self.item, 3)
        return 'm{}s{}i{}'.format(model, section, item)

    def _format_member(self, value, num_digits):
        if value is None:
            result = '?' * num_digits
        else:
            format_spec = '0' + str(num_digits)
            result = format(value, format_spec)
        return result

    @property
    def is_valid(self):
        return '?' not in str(self)

    def __eq__(self, other):
        if isinstance(other, basestring):
            return super(STASH, self).__eq__(STASH.from_msi(other))
        else:
            return super(STASH, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class SplittableInt(object):
    """
    A class to hold integers which can easily get each decimal digit individually.
    
    >>> three_six_two = SplittableInt(362)
    >>> print three_six_two
    362
    >>> print three_six_two[0]
    2
    >>> print three_six_two[2]
    3
    
    .. note:: No support for negative numbers
        
    """
    def __init__(self, value, name_mapping_dict=None):
        """
        Build a SplittableInt given the positive integer value provided.
        
        Kwargs:
        
        * name_mapping_dict - (dict) 
            A special mapping to provide name based access to specific integer positions:
            
                >>> a = SplittableInt(1234, {'hundreds': 2})
                >>> print a.hundreds
                2
                >>> a.hundreds = 9
                >>> print a.hundreds
                9
                >>> print a
                1934
        
        """
        if value < 0:
            raise ValueError('Negative numbers not supported with splittable integers object')

        # define the name lookup first (as this is the way __setattr__ is plumbed)
        self._name_lookup = name_mapping_dict or {}
        """A dictionary mapping special attribute names on this object to the slices/indices required to access them."""
        
        self._value = value
        
        self._calculate_str_value_from_value()
    
    def __int__(self):
        return int(self._value)
            
    def _calculate_str_value_from_value(self):
        # Reverse the string to get the appropriate index when getting the sliced value
        self._strvalue = [int(c) for c in str(self._value)[::-1]]
        
        # Associate the names in the lookup table to attributes
        for name, index in self._name_lookup.items():
            object.__setattr__(self, name, self[index])
        
    def _calculate_value_from_str_value(self):
        self._value = np.sum( [ 10**i * val for i, val in enumerate(self._strvalue)] )
    
    def __len__(self):
        return len(self._strvalue)
    
    def __getitem__(self, key):
        try:
            val = self._strvalue[key]
        except IndexError:
            val = 0
        
        # if the key returns a list of values, then combine them together to an integer
        if isinstance(val, list):
            val = sum([ 10**i * val for i, val in enumerate(val)])
            
        return val        
        
    def __setitem__(self, key, value):
        # The setitem method has been overridden so that assignment using ``val[0] = 1`` style syntax updates
        # the entire object appropriately.
        
        if (not isinstance(value, int) or value < 0):
            raise ValueError('Can only set %s as a positive integer value.' % key)
        
        if isinstance(key, slice):
            if ((key.start is not None and key.start < 0) or
                (key.step is not None and key.step < 0) or
                (key.stop is not None and key.stop < 0)):
                raise ValueError('Cannot assign a value with slice objects containing negative indices.')
            
            # calculate the current length of the value of this string
            current_length = len(range(*key.indices(len(self))))
            
            # get indices for as many digits as have been requested. Putting the upper limit on the number of digits at 100.
            indices = range(*key.indices(100))
            if len(indices) < len(str(value)):
                raise ValueError('Cannot put %s into %s as it has too many digits.' % (value, key))
            
            # Iterate over each of the indices in the slice, zipping them together with the associated digit
            for index, digit in zip(indices, str(value).zfill(current_length)[::-1]):
                # assign each digit to the associated index 
                self.__setitem__(index, int(digit))
        
        else:
            # If we are trying to set to an index which does not currently exist in _strvalue then extend it to the
            # appropriate length
            if (key + 1) > len(self):
                new_str_value = [0] * (key + 1)
                new_str_value[:len(self)] = self._strvalue
                self._strvalue = new_str_value
            
            self._strvalue[key] = value
            
            for name, index in self._name_lookup.items():
                if index == key:
                    object.__setattr__(self, name, value)
                
            self._calculate_value_from_str_value()
        
    def __setattr__(self, name, value):
        # if the attribute is a special value, update the index value which will in turn update the attribute value
        if (name != '_name_lookup' and name in self._name_lookup.keys()):
            self[self._name_lookup[name]] = value
        else:
            object.__setattr__(self, name, value)
    
    def __str__(self):
        return str(self._value)
    
    def __repr__(self):
        return 'SplittableInt(%r, name_mapping_dict=%r)' % (self._value, self._name_lookup)

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, SplittableInt):
            result = self._value == other._value
        elif isinstance(other, int):
            result = self._value == other
        return result

    def __ne__(self, other):
        result = self == other
        if result != NotImplemented:
            result = not result
        return result

    def _compare(self, other, op):
        result = NotImplemented
        if isinstance(other, SplittableInt):
            result = op(self._value, other._value)
        elif isinstance(other, int):
            result = op(self._value, other)
        return result

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

            
class BitwiseInt(SplittableInt):
    """
    A class to hold an integer, of fixed bit-length, which can easily get/set each bit individually.
    
    .. note::

        Uses a fixed number of bits.
        Will raise an Error when attempting to access an out-of-range flag.
    
    >>> a = BitwiseInt(511)
    >>> a.flag1
    1
    >>> a.flag8
    1
    >>> a.flag128
    1
    >>> a.flag256
    1
    >>> a.flag512
    AttributeError: 'BitwiseInt' object has no attribute 'flag512'
    >>> a.flag512 = 1
    AttributeError: Cannot set a flag that does not exist: flag512
    
    """
    
    def __init__(self, value, num_bits=None):
        """ """ # intentionally empty docstring as all covered in the class docstring.
        
        SplittableInt.__init__(self, value)
        self.flags = ()
        
        #do we need to calculate the number of bits based on the given value?
        self._num_bits = num_bits
        if self._num_bits is None:
            self._num_bits = 0
            while((value >> self._num_bits) > 0):
                self._num_bits += 1
        else:
            #make sure the number of bits is enough to store the given value.
            if (value >> self._num_bits) > 0:
                raise ValueError("Not enough bits to store value")
        
        self._set_flags_from_value()
    
    def _set_flags_from_value(self):
        all_flags = []
        
        # Set attributes "flag[n]" to 0 or 1
        for i in range(self._num_bits):
            flag_name = 1 << i
            flag_value = ((self._value >> i) & 1)
            object.__setattr__(self, 'flag%d' % flag_name, flag_value)

            # Add to list off all flags
            if flag_value:
                all_flags.append(flag_name)
            
            self.flags = tuple(all_flags)

    def _set_value_from_flags(self):
        self._value = 0
        for i in range(self._num_bits):
            bit_value = pow(2, i)
            flag_name = "flag%i" % bit_value
            flag_value = object.__getattribute__(self, flag_name)
            self._value += flag_value * bit_value

    def __iand__(self, value):
        """Perform an &= operation."""
        self._value &= value
        self._set_flags_from_value()
        return self

    def __ior__(self, value):
        """Perform an |= operation."""
        self._value |= value
        self._set_flags_from_value()
        return self

    def __iadd__(self, value):
        """Perform an inplace add operation"""
        self._value += value
        self._set_flags_from_value()
        return self
        
    def __setattr__(self, name, value):
        # Allow setting of the attribute flags
        # Are we setting a flag?
        if name.startswith("flag") and name != "flags":
            #true and false become 1 and 0
            if not isinstance(value, bool):
                raise TypeError("Can only set bits to True or False")
            
            # Setting an existing flag?
            if hasattr(self, name):
                #which flag?
                flag_value = int(name[4:])
                #on or off?
                if value:
                    self |= flag_value
                else:
                    self &= ~flag_value
            
            # Fail if an attempt has been made to set a flag that does not exist
            else:
                raise AttributeError("Cannot set a flag that does not exist: %s" % name)
            
        # If we're not setting a flag, then continue as normal
        else:
            SplittableInt.__setattr__(self, name, value)


class PPDataProxy(object):
    """A reference to the data payload of a single PP field."""
    
    __slots__ = ('path', 'offset', 'data_len', 'lbpack')
    
    def __init__(self, path, offset, data_len, lbpack):
        self.path = path
        self.offset = offset
        self.data_len = data_len
        self.lbpack = lbpack

    # NOTE:
    # "__getstate__" and "__setstate__" functions are defined here to provide a custom interface for Pickle 
    #  : Pickle "normal" behaviour is just to save/reinstate the object dictionary 
    #  : that won't work here, because the use of __slots__ means **there is no object dictionary**
    def __getstate__(self):
        # object state capture method for Pickle.dump()
        #  - return the instance data values needed to reconstruct the PPDataProxy object
        return dict([(k,getattr(self,k)) for k in PPDataProxy.__slots__])

    def __setstate__(self, state):
        # object reconstruction method for Pickle.load()
        # reinitialise the object state from the serialised values (using setattr, as there is no object dictionary)
        for (key, val) in state.items():
            setattr(self, key, val)

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % \
                (self.__class__.__name__, self.path, self.offset, self.data_len, self.lbpack)

    def load(self, data_shape, data_type, mdi, deferred_slice):
        """
        Load the corresponding proxy data item and perform any deferred slicing.
        
        Args:
        
        * data_shape (tuple of int):
            The data shape of the proxy data item.
        * data_type (:class:`numpy.dtype`):
            The data type of the proxy data item.
        * mdi (float):
            The missing data indicator value.
        * deferred_slice (tuple):
            The deferred slice to be applied to the proxy data item.
        
        Returns:
            :class:`numpy.ndarray`
        
        """
        # Load the appropriate proxy data conveniently with a context manager.
        with open(self.path, 'rb') as pp_file:
            pp_file.seek(self.offset, os.SEEK_SET)
            data = _read_data(pp_file, self.lbpack, self.data_len, data_shape, data_type, mdi)
                
        # Identify which index items in the deferred slice are tuples. 
        tuple_dims = [i for i, value in enumerate(deferred_slice) if isinstance(value, tuple)]
        
        # Whenever a slice consists of more than one tuple index item, numpy does not slice the
        # data array as we want it to. We therefore require to split the deferred slice into 
        # multiple slices and consistently slice the data with one slice per tuple.
        if len(tuple_dims) > 1:
            # Identify which index items in the deferred slice are single scalar values.
            # Such dimensions will collapse in the sliced data shape.
            collapsed_dims = [i for i, value in enumerate(deferred_slice) if isinstance(value, int)]

            # Equate the first slice to be the original deferred slice.
            tuple_slice = list(deferred_slice)
            # Replace all tuple index items in the slice, except for the first,
            # to be full slices over their dimension.
            for dim in tuple_dims[1:]:
                tuple_slice[dim] = slice(None)
            
            # Perform the deferred slice containing only the first tuple index item.
            payload = data[tuple_slice]
            
            # Re-slice the data consistently with the next single tuple index item. 
            for dim in tuple_dims[1:]:
                # Identify all those pre-sliced collapsed dimensions less than
                # the dimension of the current slice tuple index item.
                ndims_collapsed = len(filter(lambda x: x < dim, collapsed_dims))
                # Construct the single tuple slice.
                tuple_slice = [slice(None)] * payload.ndim
                tuple_slice[dim - ndims_collapsed] = deferred_slice[dim]
                # Slice the data with this single tuple slice.
                payload = payload[tuple_slice]
        else:
            # The deferred slice contains no more than one tuple index item, so
            # it's safe to slice the data directly.
            payload = data[deferred_slice]
        
        return payload

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, PPDataProxy):
            result = True
            for attr in self.__slots__:
                if getattr(self, attr) != getattr(other, attr):
                    result = False
                    break
        return result

    def __ne__(self, other):
        result = self == other
        if result != NotImplemented:
            result = not result
        return result


def _read_data(pp_file, lbpack, data_len, data_shape, data_type, mdi):
    """Read the data from the given file object given its precise location in the file."""
    if lbpack.n1 == 0:
        data = np.fromfile(pp_file, dtype=data_type, count=data_len / data_type.itemsize)
    elif lbpack.n1 == 1:
        data = pp_file.read(data_len)
        data = pp_packing.wgdos_unpack(data, data_shape[0], data_shape[1], mdi)
    elif lbpack.n1 == 2:
        data = np.fromfile(pp_file, dtype=data_type, count=data_len / data_type.itemsize)
    elif lbpack.n1 == 4:
        data = np.fromfile(pp_file, dtype=data_type, count=data_len / data_type.itemsize)
        data = pp_packing.rle_decode(data, data_shape[0], data_shape[1], mdi)
    else:
        raise iris.exceptions.NotYetImplementedError('PP fields with LBPACK of %s are not supported.' % lbpack)

    # Ensure the data is in the native byte order
    if not data.dtype.isnative:
        data.byteswap(True)
        data.dtype = data.dtype.newbyteorder('=')
        
    # Reform in row-column order
    data.shape = data_shape

    # Mask the array? 
    if mdi in data:
        data = ma.masked_values(data, mdi, copy=False)
    
    return data


# The special headers of the PPField classes which get some improved functionality
_SPECIAL_HEADERS = ('lbtim', 'lbcode', 'lbpack', 'lbproc',
                    'data', 'data_manager', 'stash', 't1', 't2')

def _header_defn(release_number):
    """
    Returns the zero-indexed header definition for a particular release of a PPField.

    """
    um_header = UM_HEADERS[release_number]
    offset = UM_TO_PP_HEADER_OFFSET
    return [(name, tuple(position - offset for position in positions)) for name, positions in um_header]


def _pp_attribute_names(header_defn):
    """
    Returns the allowed attributes of a PPField:
        all of the normal headers (i.e. not the _SPECIAL_HEADERS),
        the _SPECIAL_HEADERS with '_' prefixed,
        the possible extra data headers.

    """
    normal_headers = list(name for name, positions in header_defn if name not in _SPECIAL_HEADERS)
    special_headers = list('_' + name for name in _SPECIAL_HEADERS)
    extra_data = EXTRA_DATA.values()
    return normal_headers + special_headers + extra_data


class PPField(object):
    """
    A generic class for PP fields - not specific to a particular header release number.

    A PPField instance can easily access the PP header "words" as attributes with some added useful capabilities::
        
        for field in iris.fileformats.pp.load(filename):
            print field.lbyr
            print field.lbuser
            print field.lbuser[0]
            print field.lbtim
            print field.lbtim.ia
            print field.t1
    
    """

    # NB. Subclasses must define the attribute HEADER_DEFN to be their
    # zero-based header definition. See PPField2 and PPField3 for examples.

    __metaclass__ = abc.ABCMeta

    __slots__ = ()

    def __init__(self):
        """
        PPField instances are always created empty, and attributes are added subsequently.
        
        .. seealso::
            For PP field loading see :func:`load`.
        
        """
    
    @abc.abstractproperty
    def t1(self):
        pass

    @abc.abstractproperty
    def t2(self):
        pass

    def __repr__(self):
        """Return a string representation of the PP field."""

        # Define an ordering on the basic header names
        attribute_priority_lookup = {name: loc[0] for name, loc in self.HEADER_DEFN}
        
        # With the attributes sorted the order will remain stable if extra attributes are added.
        public_attribute_names =  attribute_priority_lookup.keys() + EXTRA_DATA.values()
        self_attrs = [(name, getattr(self, name, None)) for name in public_attribute_names]
        self_attrs = filter(lambda pair: pair[1] is not None, self_attrs)

        if hasattr(self, '_data_manager'):
            if self._data_manager is None:
                self_attrs.append( ('data', self.data) )
            else:
                self_attrs.append( ('unloaded_data_manager', self._data_manager) )
                self_attrs.append( ('unloaded_data_proxy', self._data) )
            
        # sort the attributes by position in the pp header followed, then by alphabetical order.
        attributes = sorted(self_attrs, key=lambda pair: (attribute_priority_lookup.get(pair[0], 999), pair[0]) )
        
        return 'PP Field' + ''.join(['\n   %s: %s' % (k, v) for k, v in attributes]) + '\n'

    @property
    def stash(self):
        """A stash property giving access to the associated STASH object, now supporting __eq__"""
        if not hasattr(self, '_stash'):
            self._stash = STASH(self.lbuser[6], self.lbuser[3] / 1000, self.lbuser[3] % 1000)
        return self._stash

    # lbtim
    def _lbtim_setter(self, new_value):
        if not isinstance(new_value, SplittableInt):
            # add the ia/ib/ic values for lbtim
            new_value = SplittableInt(new_value, {'ia':slice(2, None), 'ib':1, 'ic':0})
        self._lbtim = new_value

    lbtim = property(lambda self: self._lbtim, _lbtim_setter)

    # lbcode
    def _lbcode_setter(self, new_value):
        if not isinstance(new_value, SplittableInt):
            # add the ix/iy values for lbcode
            new_value = SplittableInt(new_value, {'iy':slice(0, 2), 'ix':slice(2, 4)})
        self._lbcode = new_value

    lbcode = property(lambda self: self._lbcode, _lbcode_setter)
    
    # lbpack
    def _lbpack_setter(self, new_value):
        if not isinstance(new_value, SplittableInt):
            # add the n1/n2/n3/n4/n5 values for lbpack
            name_mapping = dict(n5=slice(4, None), n4=3, n3=2, n2=1, n1=0)
            new_value = SplittableInt(new_value, name_mapping)
        self._lbpack = new_value

    lbpack = property(lambda self: self._lbpack, _lbpack_setter)

    # lbproc
    def _lbproc_setter(self, new_value):
        if not isinstance(new_value, BitwiseInt):
            new_value = BitwiseInt(new_value, num_bits=18)
        self._lbproc = new_value    

    lbproc = property(lambda self: self._lbproc, _lbproc_setter)
    
    @property
    def data(self):
        """The :class:`numpy.ndarray` representing the multidimensional data of the pp file"""
        # Cache the real data on first use
        if self._data_manager is not None:
            self._data = self._data_manager.load(self._data)
            self._data_manager = None
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._data_manager = None

    def _time_class(self):
        # TODO: Handle the other calendars (i.e. not 360-day)
        return {2: iris.time.Time360}.get(self.lbtim.ic, netcdftime.datetime)

    @property
    def calendar(self):
        """Return the calendar of the field."""
        # TODO #577 What calendar to return when ibtim.ic in [0, 3]
        return iris.unit.CALENDAR_GREGORIAN if self.lbtim.ic != 2 else iris.unit.CALENDAR_360_DAY

    def _read_extra_data(self, pp_file, file_reader, extra_len):
        """Read the extra data section and update the self appropriately."""
        
        # While there is still extra data to decode run this loop
        while extra_len > 0:
            extra_int_code = struct.unpack_from('>L', file_reader(PP_WORD_DEPTH))[0]
            extra_len -= PP_WORD_DEPTH
            
            ib = extra_int_code % 1000
            ia = extra_int_code // 1000
            
            data_len = ia * PP_WORD_DEPTH

            if ib == 10:
                self.field_title = ''.join(struct.unpack_from('>%dc' % data_len, file_reader(data_len))).rstrip('\00')
            elif ib == 11:
                self.domain_title = ''.join(struct.unpack_from('>%dc' % data_len, file_reader(data_len))).rstrip('\00')
            elif ib in EXTRA_DATA:
                attr_name = EXTRA_DATA[ib]
                values = np.fromfile(pp_file, dtype=np.dtype('>f%d' % PP_WORD_DEPTH), count=ia)
                # Ensure the values are in the native byte order
                if not values.dtype.isnative:
                    values.byteswap(True)
                    values.dtype = values.dtype.newbyteorder('=')
                setattr(self, attr_name, values)
            else:
                raise ValueError('Unknown IB value for extra data: %s' % ib)
            
            extra_len -= data_len
            
    @property
    def x_bounds(self):
        if hasattr(self, "x_lower_bound") and hasattr(self, "x_upper_bound"):
            return np.column_stack((self.x_lower_bound, self.x_upper_bound))

    @property
    def y_bounds(self):
        if hasattr(self, "y_lower_bound") and hasattr(self, "y_upper_bound"):
            return np.column_stack((self.y_lower_bound, self.y_upper_bound))
        
    def save(self, file_handle):
        """
        Save the PPField to the given file object (typically created with :func:`open`).
        
        ::
        
            # to append the field to a file
            a_pp_field.save(open(filename, 'ab'))
            
            # to overwrite/create a file
            a_pp_field.save(open(filename, 'wb'))
        
        
        .. note::

            The fields which are automatically calculated are: 'lbext',
            'lblrec' and 'lbuser[0]'. Some fields are not currently
            populated, these are: 'lbegin', 'lbnrec', 'lbuser[1]'.
            
        """

        # Before we can actually write to file, we need to calculate the header elements.
        # First things first, make sure the data is big-endian
        data = self.data
        if isinstance(data, ma.core.MaskedArray):
            data = data.filled(fill_value=self.bmdi)
        
        if data.dtype.newbyteorder('>') != data.dtype:
            # take a copy of the data when byteswapping
            data = data.byteswap(False)
            data.dtype = data.dtype.newbyteorder('>')

        # Create the arrays which will hold the header information
        lb = np.empty(shape=NUM_LONG_HEADERS, dtype=np.dtype(">u%d" % PP_WORD_DEPTH))
        b = np.empty(shape=NUM_FLOAT_HEADERS, dtype=np.dtype(">f%d" % PP_WORD_DEPTH))

        # Populate the arrays from the PPField
        for name, pos in self.HEADER_DEFN:
            try:
                header_elem = getattr(self, name)
            except AttributeError:
                raise AttributeError("PPField.save() could not find %s" % name)
            if pos[0] <= NUM_LONG_HEADERS - UM_TO_PP_HEADER_OFFSET:
                index = slice(pos[0], pos[-1] + 1)
                if isinstance(header_elem, SplittableInt):
                    header_elem = int(header_elem)
                lb[index] = header_elem
            else:
                index = slice(pos[0] - NUM_LONG_HEADERS, pos[-1] - NUM_LONG_HEADERS + 1)
                b[index] = header_elem

        # Although all of the elements are now populated, we still need to update some of the elements in case
        # things have changed (for example, the data length etc.)

        # Set up a variable to represent the datalength of this PPField in WORDS.
        len_of_data_payload = 0

        # set up a list to hold the extra data which will need to be encoded at the end of the data
        extra_items = []
        # iterate through all of the possible extra data fields
        for ib, extra_data_attr_name in EXTRA_DATA.iteritems():
            # try to get the extra data field, returning None if it doesn't exist
            extra_elem = getattr(self, extra_data_attr_name, None)
            if extra_elem is not None:
                # The special case of character extra data must be caught
                if isinstance(extra_elem, basestring):
                    ia = len(extra_elem)
                    # pad any strings up to a multiple of PP_WORD_DEPTH (this length is # of bytes)
                    ia = (PP_WORD_DEPTH - (ia-1) % PP_WORD_DEPTH) + (ia-1)
                    extra_elem = extra_elem.ljust(ia, '\00')

                    # ia is now the datalength in WORDS of the string
                    ia /= PP_WORD_DEPTH
                else:
                    # ia is the datalength in WORDS
                    ia = np.product(extra_elem.shape)
                    # flip the byteorder if the data is not big-endian
                    if extra_elem.dtype.newbyteorder('>') != extra_elem.dtype:
                        # take a copy of the extra data when byte swapping
                        extra_elem = extra_elem.byteswap(False)
                        extra_elem.dtype = extra_elem.dtype.newbyteorder('>')

                # add the number of bytes to the len_of_data_payload variable + the extra integer which will encode ia/ib
                len_of_data_payload += PP_WORD_DEPTH * ia + PP_WORD_DEPTH
                integer_code = 1000 * ia + ib
                extra_items.append( [integer_code, extra_elem] )

                if ia >= 1000:
                    raise IOError('PP files cannot write extra data with more than '
                                  '1000 elements. Tried to write "%s" which has %s '
                                  'elements.' % (extra_data_attr_name, ib)
                                  )

        HEADER_DICT = dict(self.HEADER_DEFN)

        # populate lbext in WORDS
        lb[HEADER_DICT['lbext'][0]] = len_of_data_payload / PP_WORD_DEPTH

        # Put the data length of pp.data into len_of_data_payload (in BYTES)
        len_of_data_payload += data.size * PP_WORD_DEPTH

        # populate lbrec in WORDS
        lb[HEADER_DICT['lblrec'][0]] = len_of_data_payload / PP_WORD_DEPTH

        # populate lbuser[0] to have the data's datatype
        if data.dtype == np.dtype('>f4'):
            lb[HEADER_DICT['lbuser'][0]] = 1
        elif data.dtype == np.dtype('>i4'):
            # NB: there is no physical difference between lbuser[0] of 2 or 3 so we encode just 2
            lb[HEADER_DICT['lbuser'][0]] = 2
        else:
            raise IOError('Unable to write data array to a PP file. The datatype was %s.' % data.dtype)

        # NB: lbegin, lbnrec, lbuser[1] not set up

        # Now that we have done the manouvering required, write to the file...
        if not isinstance(file_handle, file):
            raise TypeError('The file_handle argument must be an instance of a Python file object, but got %r. \n'
                             'e.g. open(filename, "wb") to open a binary file with write permission.' % type(file_handle))

        pp_file = file_handle

        # header length
        pp_file.write(struct.pack(">L", PP_HEADER_DEPTH))

        # 49 integers
        lb.tofile(pp_file)
        # 16 floats
        b.tofile(pp_file)

        #Header length (again)
        pp_file.write(struct.pack(">L", PP_HEADER_DEPTH))

        # Data length (including extra data length)
        pp_file.write(struct.pack(">L", int(len_of_data_payload)))

        # the data itself
        if lb[HEADER_DICT['lbpack'][0]] == 0:
            data.tofile(pp_file)
        else:
            raise NotImplementedError('Writing packed pp data with lbpack of %s '
                                      'is not supported.' % lb[HEADER_DICT['lbpack'][0]])

        # extra data elements
        for int_code, extra_data in extra_items:
            pp_file.write(struct.pack(">L", int(int_code)))
            if isinstance(extra_data, basestring):
                pp_file.write(struct.pack(">%sc" % len(extra_data), *extra_data))
            else:
                extra_data = extra_data.astype(np.dtype('>f4'))
                extra_data.tofile(pp_file)

        # Data length (again)
        pp_file.write(struct.pack(">L", int(len_of_data_payload)))

    ##############################################################
    #
    # From here on define helper methods for PP -> Cube conversion.
    #

    def regular_points(self, xy):
        """Return regular points from the PPField, or fail if not regular.
        
        Args:
        
            * xy - a string, "x" or "y" to specify the dimension for which to return points.
            
        """ 
        if xy.lower() == "x":
            bz = self.bzx
            bd = self.bdx
            count = self.lbnpt
        elif xy.lower() == "y":
            bz = self.bzy
            bd = self.bdy
            count = self.lbrow
        else:
            raise ValueError("'x' or 'y' not supplied")
         
        return (bz + bd) + bd * np.arange(count, dtype=np.float32)

    def regular_bounds(self, xy): 
        """Return regular bounds from the PPField, or fail if not regular.
        
        Args:
        
            * xy - a string, "x" or "y" to specify the dimension for which to return points.
            
        """ 
        if xy.lower() == "x":
            delta = 0.5 * self.bdx
        elif xy.lower() == "y":
            delta = 0.5 * self.bdy
        else:
            raise ValueError("'x' or 'y' not supplied")

        points = self.regular_points(xy) 
        return np.concatenate([[points - delta], [points + delta]]).T 

    def time_unit(self, time_unit, epoch='epoch'):
        return iris.unit.Unit('%s since %s' % (time_unit, epoch), calendar=self.calendar)

    def coord_system(self):
        """Return a CoordSystem for this PPField.
    
        Returns:
            Currently, a :class:`~iris.coord_systems.GeogCS` or :class:`~iris.coord_systems.RotatedGeogCS`.
    
        """
        geog_cs =  iris.coord_systems.GeogCS(EARTH_RADIUS)
        if self.bplat != 90.0 or self.bplon != 0.0:
            geog_cs = iris.coord_systems.RotatedGeogCS(self.bplat, self.bplon, ellipsoid=geog_cs)
        
        return geog_cs

    def _x_coord_name(self):
        # TODO: Remove once we have the ability to derive this in the rules.
        x_name = "longitude"
        if isinstance(self.coord_system(), iris.coord_systems.RotatedGeogCS):
            x_name = "grid_longitude"
        return x_name

    def _y_coord_name(self):
        # TODO: Remove once we have the ability to derive this in the rules.
        y_name = "latitude"
        if isinstance(self.coord_system(), iris.coord_systems.RotatedGeogCS):
            y_name = "grid_latitude"
        return y_name
    
    def read_data(self, ff_file, pp_data_depth, pp_data_shape, pp_data_type):
        return _read_data(ff_file, self.lbpack, pp_data_depth, pp_data_shape, pp_data_type, self.bmdi)

    def copy(self):
        """
        Returns a deep copy of this PPField.

        Returns:
            A copy instance of the :class:`PPField`.

        """
        return self._deepcopy({})

    def __deepcopy__(self, memo):
        return self._deepcopy(memo)

    def _deepcopy(self, memo):
        field = self.__class__()
        for attr in self.__slots__:
            if hasattr(self, attr):
                value = getattr(self, attr)
                # Cope with inability to deepcopy a 0-d NumPy array.
                if attr == '_data' and value is not None and value.ndim == 0:
                    setattr(field, attr, np.array(deepcopy(value[()], memo)))
                else:
                    setattr(field, attr, deepcopy(value, memo))
        return field

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, PPField):
            result = True
            for attr in self.__slots__:
                attrs = [hasattr(self, attr), hasattr(other, attr)]
                if all(attrs):
                    if not np.all(getattr(self, attr) == getattr(other, attr)):
                        result = False
                        break
                elif any(attrs):
                    result = False
                    break
        return result

    def __ne__(self, other):
        result = self == other
        if result != NotImplemented:
            result = not result
        return result


class PPField2(PPField):
    """
    A class to hold a single field from a PP file, with a header release number of 2.
    
    """
    HEADER_DEFN = _header_defn(2)

    __slots__ = _pp_attribute_names(HEADER_DEFN)

    def _get_t1(self):
        if not hasattr(self, '_t1'):
            time_class = self._time_class()
            self._t1 = time_class(self.lbyr, self.lbmon, self.lbdat, self.lbhr, self.lbmin)
        return self._t1
    
    def _set_t1(self, dt):
        self.lbyr = dt.year
        self.lbmon = dt.month
        self.lbdat = dt.day
        self.lbhr = dt.hour
        self.lbmin = dt.minute
        self.lbday = int(dt.strftime('%j'))
        if hasattr(self, '_t1'):
            delattr(self, '_t1')
        
    t1 = property(_get_t1, _set_t1, None,
        "A netcdftime.datetime object consisting of the lbyr, lbmon, lbdat, lbhr, and lbmin attributes.")

    def _get_t2(self):
        if not hasattr(self, '_t2'):
            time_class = self._time_class()
            self._t2 = time_class(self.lbyrd, self.lbmond, self.lbdatd, self.lbhrd, self.lbmind)
        return self._t2

    def _set_t2(self, dt):
        self.lbyrd = dt.year
        self.lbmond = dt.month
        self.lbdatd = dt.day
        self.lbhrd = dt.hour
        self.lbmind = dt.minute
        self.lbdayd = int(dt.strftime('%j'))
        if hasattr(self, '_t2'):
            delattr(self, '_t2')

    t2 = property(_get_t2, _set_t2, None,
        "A netcdftime.datetime object consisting of the lbyrd, lbmond, lbdatd, lbhrd, and lbmind attributes.")


class PPField3(PPField):
    """
    A class to hold a single field from a PP file, with a header release number of 3.
    
    """
    HEADER_DEFN = _header_defn(3)

    __slots__ = _pp_attribute_names(HEADER_DEFN)

    def _get_t1(self):
        if not hasattr(self, '_t1'):
            time_class = self._time_class()
            self._t1 = time_class(self.lbyr, self.lbmon, self.lbdat, self.lbhr, self.lbmin, self.lbsec)
        return self._t1
    
    def _set_t1(self, dt):
        self.lbyr = dt.year
        self.lbmon = dt.month
        self.lbdat = dt.day
        self.lbhr = dt.hour
        self.lbmin = dt.minute
        self.lbsec = dt.second
        if hasattr(self, '_t1'):
            delattr(self, '_t1')
        
    t1 = property(_get_t1, _set_t1, None,
        "A netcdftime.datetime object consisting of the lbyr, lbmon, lbdat, lbhr, lbmin, and lbsec attributes.")

    def _get_t2(self):
        if not hasattr(self, '_t2'):
            time_class = self._time_class()
            self._t2 = time_class(self.lbyrd, self.lbmond, self.lbdatd, self.lbhrd, self.lbmind, self.lbsecd)
        return self._t2

    def _set_t2(self, dt):
        self.lbyrd = dt.year
        self.lbmond = dt.month
        self.lbdatd = dt.day
        self.lbhrd = dt.hour
        self.lbmind = dt.minute
        self.lbsecd = dt.second
        if hasattr(self, '_t2'):
            delattr(self, '_t2')

    t2 = property(_get_t2, _set_t2, None,
        "A netcdftime.datetime object consisting of the lbyrd, lbmond, lbdatd, lbhrd, lbmind, and lbsecd attributes.")

PP_CLASSES = {
    2: PPField2,
    3: PPField3
}


def make_pp_field(header_values):
    # Choose a PP field class from the value of LBREL
    lbrel = header_values[21]
    if lbrel not in PP_CLASSES:
        raise ValueError('Unsupported header release number: {}'.format(lbrel))
    pp_field = PP_CLASSES[lbrel]()
    for name, loc in pp_field.HEADER_DEFN:
        if len(loc) == 1:
            value = header_values[loc[0]]
        else:
            value = header_values[loc[0]:loc[-1]+1]
        setattr(pp_field, name, value)
    return pp_field


def load(filename, read_data=False):
    """
    Return an iterator of PPFields given a filename.
    
    Args:
    
    * filename - string of the filename to load.
    
    Kwargs:
    
    * read_data - boolean 
        Flag whether or not the data should be read, if False an empty data manager
        will be provided which can subsequently load the data on demand. Default False.
        
    To iterate through all of the fields in a pp file::
    
        for field in iris.fileformats.pp.load(filename):
            print field
    
    """
    
    pp_file = open(filename, 'rb')
        
    # Get a reference to the seek method on the file
    # (this is accessed 3* #number of headers so can provide a small performance boost)
    pp_file_seek = pp_file.seek
    pp_file_read = pp_file.read
        
    # Keep reading until we reach the end of file
    while True:
        # Move past the leading header length word
        pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)
        # Get the LONG header entries
        header_longs = np.fromfile(pp_file, dtype='>i%d' % PP_WORD_DEPTH, count=NUM_LONG_HEADERS)
        # Nothing returned => EOF
        if len(header_longs) == 0:
            break
        # Get the FLOAT header entries
        header_floats = np.fromfile(pp_file, dtype='>f%d' % PP_WORD_DEPTH, count=NUM_FLOAT_HEADERS)
        header = tuple(header_longs) + tuple(header_floats)

        # Make a PPField of the appropriate sub-class (depends on header release number)
        pp_field = make_pp_field(header)
        
        # Skip the trailing 4-byte word containing the header length
        pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)
        
        # Read the word telling me how long the data + extra data is
        # This value is # of bytes
        len_of_data_plus_extra = struct.unpack_from('>L', pp_file_read(PP_WORD_DEPTH))[0]
        if len_of_data_plus_extra != pp_field.lblrec * PP_WORD_DEPTH:
            raise ValueError('LBLREC has a different value to the integer recorded after the '
                             'header in the file (%s and %s).' % (pp_field.lblrec * PP_WORD_DEPTH, 
                                                                  len_of_data_plus_extra))
        
        # calculate the extra length in bytes
        extra_len = pp_field.lbext * PP_WORD_DEPTH
        
        # Derive size and datatype of payload
        data_len = len_of_data_plus_extra - extra_len
        data_type = LBUSER_DTYPE_LOOKUP.get(pp_field.lbuser[0], LBUSER_DTYPE_LOOKUP['default'])
        
        data_shape = (pp_field.lbrow, pp_field.lbnpt)

        if read_data:
            pp_field._data = pp_field.read_data(pp_file, data_len, data_shape, data_type)
            pp_field._data_manager = None
        else:
            # NB. This makes a 0-dimensional array
            pp_field._data = np.array(PPDataProxy(filename, pp_file.tell(), data_len, pp_field.lbpack))
            pp_field._data_manager = iris.fileformats.manager.DataManager(data_shape, data_type, pp_field.bmdi)

            # Skip the data
            pp_file_seek(data_len, os.SEEK_CUR)

        # Do we have any extra data to deal with?
        if extra_len:
            pp_field._read_extra_data(pp_file, pp_file_read, extra_len)

        # Skip that last 4 byte record telling me the length of the field I have already read
        pp_file_seek(PP_WORD_DEPTH, os.SEEK_CUR)

        yield pp_field
        
    pp_file.close()


def _ensure_load_rules_loaded():
    """Makes sure the standard conversion and verification rules are loaded."""

    # Uses these module-level variables
    global _load_rules, _cross_reference_rules

    rules = iris.fileformats.rules

    if _load_rules is None:
        basepath = iris.config.CONFIG_PATH
        _load_rules = rules.RulesContainer(os.path.join(basepath, 'pp_rules.txt'))

    if _cross_reference_rules is None:
        basepath = iris.config.CONFIG_PATH
        _cross_reference_rules = rules.RulesContainer(os.path.join(basepath, 'pp_cross_reference_rules.txt'), 
                                                           rule_type=rules.ObjectReturningRule)


def add_load_rules(filename):
    """
    Registers a rules file for use during the PP load process.

    Registered files are processed after the standard conversion rules, and in
    the order they were registered.
    
    """
    _ensure_load_rules_loaded()
    _load_rules.import_rules(filename)


def reset_load_rules():
    """Resets the PP load process to use only the standard conversion rules."""
    
    # Uses this module-level variable
    global _load_rules
    
    _load_rules = None


def _ensure_save_rules_loaded():
    """Makes sure the standard save rules are loaded."""

    # Uses these module-level variables
    global _save_rules

    if _save_rules is None:
        # Load the pp save rules
        rules_filename = os.path.join(iris.config.CONFIG_PATH, 'pp_save_rules.txt')
        _save_rules = iris.fileformats.rules.RulesContainer(rules_filename, iris.fileformats.rules.ProcedureRule)


def add_save_rules(filename):
    """
    Registers a rules file for use during the PP save process.

    Registered files are processed after the standard conversion rules, and in
    the order they were registered.
    
    """
    _ensure_save_rules_loaded()
    _save_rules.import_rules(filename)


def reset_save_rules():
    """Resets the PP save process to use only the standard conversion rules."""
    
    # Uses this module-level variable
    global _save_rules
    
    _save_rules = None


def load_cubes(filenames, callback=None):
    """
    Loads cubes from a list of pp filenames.
    
    Args:
    
    * filenames - list of pp filenames to load
    
    Kwargs:
    
    * callback - a function which can be passed on to :func:`iris.io.run_callback`
    
    .. note::

        The resultant cubes may not be in the order that they are in the file (order 
        is not preserved when there is a field with orography references)
         
    """
    return _load_cubes_variable_loader(filenames, callback, load)


def _load_cubes_variable_loader(filenames, callback, loading_function,
                                loading_function_kwargs=None):
    _ensure_load_rules_loaded()
    rules = iris.fileformats.rules
    pp_loader = rules.Loader(loading_function, loading_function_kwargs or {},
                             _load_rules, _cross_reference_rules, 'PP_LOAD')
    return rules.load_cubes(filenames, callback, pp_loader)


def save(cube, target, append=False, field_coords=None):
    """
    Use the PP saving rules (and any user rules) to save a cube to a PP file.
    
    Args:

        * cube         - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or list of cubes.
        * target       - A filename or open file handle.

    Kwargs:
    
        * append       - Whether to start a new file afresh or add the cube(s) to the end of the file. 
                         Only applicable when target is a filename, not a file handle.
                         Default is False.

        * field_coords - list of 2 coords or coord names which are to be used for
                         reducing the given cube into 2d slices, which will ultimately 
                         determine the x and y coordinates of the resulting fields. 
                         If None, the final two  dimensions are chosen for slicing.

    See also :func:`iris.io.save`.
    
    """

    # Open issues
    # Could use rules in "sections" ... e.g. to process the extensive dimensions; ...?
    # Could pre-process the cube to add extra convenient terms?
    #    e.g. x-coord, y-coord ... but what about multiple coordinates on the dimension?

    # How to perform the slicing?
    #   Do we always slice in the last two dimensions?
    #   Not all source data will contain lat-lon slices.
    # What do we do about dimensions with multiple coordinates?

    # Deal with:
    #   LBLREC - Length of data record in words (incl. extra data)
    #       Done on save(*)
    #   LBUSER[0] - Data type
    #       Done on save(*)
    #   LBUSER[1] - Start address in DATA (?! or just set to "null"?)
    #   BLEV - Level - the value of the coordinate for LBVC

    # *) With the current on-save way of handling LBLREC and LBUSER[0] we can't
    # check if they've been set correctly without *actually* saving as a binary
    # PP file. That also means you can't use the same reference.txt file for
    # loaded vs saved fields (unless you re-load the saved field!).

    # Set to (or leave as) "null":
    #   LBEGIN - Address of start of field in direct access dataset
    #   LBEXP - Experiment identification
    #   LBPROJ - Fields file projection number
    #   LBTYP - Fields file field type code
    #   LBLEV - Fields file level code / hybrid height model level

    # Build confidence by having a PP object that records which header items
    # have been set, and only saves if they've all been set?
    #   Watch out for extra-data.

    # On the flip side, record which Cube metadata has been "used" and flag up
    # unused?

    _ensure_save_rules_loaded()

    # pp file
    if isinstance(target, basestring):
        pp_file = open(target, "ab" if append else "wb")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        pp_file = target
    else:
        raise ValueError("Can only save pp to filename or writable")

    n_dims = len(cube.shape)
    if n_dims < 2:
        raise ValueError('Unable to save a cube of fewer than 2 dimensions.')

    if field_coords is not None:
        # cast the given coord/coord names into cube coords
        field_coords = cube._as_list_of_coords(field_coords)
        if len(field_coords) != 2:
            raise ValueError('Got %s coordinates in field_coords, expecting exactly 2.' % len(field_coords))
    else:
        # default to the last two dimensions (if result of coords is an empty list, will
        # raise an IndexError)
        # NB watch out for the ordering of the dimensions
        field_coords = (cube.coords(dimensions=n_dims-2)[0], cube.coords(dimensions=n_dims-1)[0])

    # Save each named or latlon slice2D in the cube
    for slice2D in cube.slices(field_coords):
        # Start with a blank PPField
        pp_field = PPField3()

        # Set all items to 0 because we need lbuser, lbtim
        # and some others to be present before running the rules.
        for name, positions in pp_field.HEADER_DEFN:
            # Establish whether field name is integer or real
            default = 0 if positions[0] <= NUM_LONG_HEADERS - UM_TO_PP_HEADER_OFFSET else 0.0
            # Establish whether field position is scalar or composite
            if len(positions) > 1:
                default = [default] * len(positions)
            setattr(pp_field, name, default)

        # Some defaults should not be 0
        pp_field.lbrel = 3      # Header release 3.
        pp_field.lbcode = 1     # Grid code.
        pp_field.bmks = 1.0     # Some scaley thing.
        pp_field.lbproc = 0

        # Set the data
        pp_field.data = slice2D.data

        # Run the PP save rules on the slice2D, to fill the PPField,
        # recording the rules that were used
        rules_result = _save_rules.verify(slice2D, pp_field)
        verify_rules_ran = rules_result.matching_rules
        
        # Log the rules used
        iris.fileformats.rules.log('PP_SAVE', target if isinstance(target, basestring) else target.name, verify_rules_ran)

        # Write to file
        pp_field.save(pp_file)

    if isinstance(target, basestring):
        pp_file.close()
