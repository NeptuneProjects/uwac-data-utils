# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np


def create_empty_data_chunk(
    delta: int, dtype: np.dtype, fill_value: Optional[Union[int, float]] = None
) -> np.ndarray:
    """
    Creates an NumPy array depending on the given data type and fill value.

    If no ``fill_value`` is given a masked array will be returned.

    This function is adapted from the ObsPy library:
    https://docs.obspy.org/index.html

    Args:
        delta (int): Length of the array.
        dtype (np.dtype): Data type of the array.
        fill_value (Optional[Union[int, float]], optional): Value to fill the array with.
            Defaults to None.

    Returns:
        np.ndarray: NumPy array.

    Examples:
    >>> create_empty_data_chunk(3, 'int', 10)
    array([10, 10, 10])

    >>> create_empty_data_chunk(
    ...     3, 'f')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    masked_array(data = [-- -- --],
                 mask = ...,
                 ...)
    """
    if fill_value is None:
        temp = np.ma.masked_all(delta, dtype=np.dtype(dtype))
        # fill with nan if float number and otherwise with a very small number
        if issubclass(temp.data.dtype.type, np.integer):
            temp.data[:] = np.iinfo(temp.data.dtype).min
        else:
            temp.data[:] = np.nan
    elif (isinstance(fill_value, list) or isinstance(fill_value, tuple)) and len(
        fill_value
    ) == 2:
        # if two values are supplied use these as samples bordering to our data
        # and interpolate between:
        ls = fill_value[0]
        rs = fill_value[1]
        # include left and right sample (delta + 2)
        interpolation = np.linspace(ls, rs, delta + 2)
        # cut ls and rs and ensure correct data type
        temp = np.require(interpolation[1:-1], dtype=np.dtype(dtype))
    else:
        temp = np.ones(delta, dtype=np.dtype(dtype))
        temp *= fill_value
    return temp


def round_away(number: Union[int, float]) -> int:
    """Function to round a number away from zero to the nearest integer.
    This is potentially desired behavior in the trim functions.

    This function is adapted from the ObsPy library:
    https://docs.obspy.org/index.html

    Args:
        number (Union[int, float]): Number to round.

    Returns:
        int: Rounded number.

    Examples:
    >>> round_away(2.5)
    3
    >>> round_away(-2.5)
    -3

    >>> round_away(10.5)
    11
    >>> round_away(-10.5)
    -11

    >>> round_away(11.0)
    11
    >>> round_away(-11.0)
    -11
    """
    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))
