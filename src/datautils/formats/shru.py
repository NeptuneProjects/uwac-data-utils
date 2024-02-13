# -*- coding: utf-8 -*-

from array import array
from dataclasses import dataclass
import logging
from pathlib import Path
import struct
from typing import BinaryIO, Optional, Union
import warnings

import numpy as np

from datautils.util import db_to_linear

ADC_HALFSCALE = 2.5  # ADC half scale volts (+/- half scale is ADC i/p range)
ADC_MAXVALUE = 2**23  # ADC maximum halfscale o/p value, half the 2's complement range
BYTES_HDR = 1024
BYTES_PER_SAMPLE = 3


@dataclass(frozen=True)
class SHRUHeader:
    rhkey: str
    date: array
    time: array
    microsec: int
    rec: int
    ch: int
    npts: int
    rhfs: float
    unused: array
    rectime: int
    rhlat: str
    rhlng: str
    nav120: array
    nav115: array
    nav110: array
    POS: str
    unused2: array
    nav_day: int
    nav_hour: int
    nav_min: int
    nav_sec: int
    lblnav_flag: int
    unused3: array
    reclen: int
    acq_day: int
    acq_hour: int
    acq_min: int
    acq_sec: int
    acq_recnum: int
    ADC_tagbyte: int
    glitchcode: int
    bootflag: int
    internal_temp: str
    bat_voltage: str
    bat_current: str
    status: str
    proj: str
    shru_num: str
    vla: str
    hla: str
    filename: str
    record: str
    adate: str
    atime: str
    file_length: int
    total_records: int
    unused4: array
    adc_mode: int
    adc_clk_code: int
    unused5: array
    timebase: int
    unused6: array
    unused7: array
    rhkeyl: str


def _chk_read(count: int) -> int:
    """Check if the read count is correct.

    Currently unused.

    Args:
        count (int): Read count.

    Returns:
        int: 0 if count is 0, -1 otherwise.
    """
    if count == 0:
        return -1
    return 0


def condition_data(
    data: np.ndarray, fixed_gain: list[float], sensitivity: list[float]
) -> tuple[np.ndarray, str]:
    """Condition 24-bit data to pressure.

    Args:
        data (np.ndarray): 24-bit data.
        fixed_gain (list[float]): Fixed gain [dB].
        sensitivity (list[float]): Sensitivity [dB].

    Returns:
        tuple[np.ndarray, str]: Converted data and units.
    """
    linear_fixed_gain = db_to_linear(fixed_gain)
    linear_sensitivity = db_to_linear(sensitivity)
    data, _ = _convert_to_voltage(data, linear_fixed_gain)
    return _convert_to_pressure(data, linear_sensitivity)


def _convert_24bit_to_int(
    raw_data: bytes, nch: int, spts: int
) -> tuple[np.ndarray, int]:
    """Convert 24-bit data to 32-bit integers.

    Args:
        raw_data (bytes): 24-bit data.
        nch (int): Number of channels.
        spts (int): Number of samples per channel.

    Returns:
        tuple[np.ndarray, int]: Converted data and count.
    """
    # Convert the byte data to 24-bit integers (bit manipulation may be required)
    # Ensure we have a suitable type for bit manipulation
    data = (
        np.frombuffer(raw_data, dtype=np.uint8)
        .reshape(spts, nch, BYTES_PER_SAMPLE)
        .astype(np.int32)
    )
    # Combine bytes to form 24-bit integers
    data_24bit = (data[:, :, 0] << 16) | (data[:, :, 1] << 8) | data[:, :, 2]
    # Adjust for sign if necessary (assuming two's complement representation)
    data_24bit[data_24bit >= 2**23] -= 2**24
    return data_24bit.reshape(spts, nch), len(raw_data) // BYTES_PER_SAMPLE


def _convert_to_pressure(
    data: np.ndarray, linear_sensitivity: list[float]
) -> tuple[np.ndarray, str]:
    """Convert voltage data to pressure.

    Args:
        data (np.ndarray): Voltage data.

    Returns:
        tuple[np.ndarray, str]: Converted data and units.
    """
    return data * np.array(linear_sensitivity)[np.newaxis, :], "uPa"


def _convert_to_voltage(
    data: np.ndarray, linear_fixed_gain: list[float]
) -> tuple[np.ndarray, str]:
    """Convert 24-bit data to voltage.

    Args:
        data (np.ndarray): 24-bit data.
        fixed_linear_gain (list[float]): Fixed linear gain [V/count].

    Returns:
        tuple[np.ndarray, str]: Converted data and units.
    """
    norm_factor = ADC_HALFSCALE / ADC_MAXVALUE / np.array(linear_fixed_gain)
    return data * norm_factor[np.newaxis, :], "V"


def get_data_record(fid: BinaryIO, nch: int, spts: int) -> tuple[np.ndarray, int]:
    total_bytes = nch * spts * BYTES_PER_SAMPLE
    data_bytes = fid.read(total_bytes)
    if len(data_bytes) != total_bytes:
        raise ValueError("Failed to read all data bytes.")
    return _convert_24bit_to_int(data_bytes, nch, spts)


def get_num_channels(drhs: list[SHRUHeader]) -> int:
    for record in drhs:
        if record.ch != drhs[0].ch:
            warnings.warn("Number of channels varies across records.")
    return drhs[0].ch


def read_data(
    filename: Path,
    records: Union[int, list[int]],
    channels: Union[int, list[int]],
    fixed_gain: Union[float, list[float]] = 20.0,
    drhs: Optional[list[SHRUHeader]] = None,
) -> tuple[np.ndarray, SHRUHeader]:
    """Read 24-bit data from a SHRU file.

    Args:
        filename (Path): File to read.
        records (Union[int, list[int]]): Record number(s) to read.
        channels (Union[int, list[int]]): Channel number(s) to read.
        fixed_gain (Union[float, list[float]], optional): Fixed gain [dB]. Defaults to 20.0.
        drhs (Optional[list[SHRUHeader]], optional): Data record headers. Defaults to None.

    Returns:
        tuple[np.ndarray, SHRUHeader]: Data and header.

    Raises:
        ValueError: If the number of channels requested exceeds the number of channels in the file.
        ValueError: If the length of fixed_gain does not match the length of channels.
        ValueError: If the record is corrupted.
    """

    if not isinstance(records, list):
        records = [records]
    if not isinstance(channels, list):
        channels = [channels]
    if not isinstance(fixed_gain, list):
        fixed_gain = [fixed_gain] * len(channels)
    if isinstance(fixed_gain, list) and len(fixed_gain) != len(channels):
        raise ValueError("Length of fixed_gain must match length of channels.")

    if drhs is None:
        drhs = read_shru_headers(filename)

    nch = get_num_channels(drhs)
    if any((i > nch - 1) for i in channels):
        raise ValueError(
            f"Channel {len(channels)} requested but only got {nch} from header."
        )

    header1 = drhs[records[0]]

    # Calculate the number of bytes to skip to start reading data
    skip_bytes = sum([drh.reclen for drh in drhs[0 : records[0]]])

    with open(filename, "rb") as fid:
        fid.seek(skip_bytes, 0)  # Skip to selected record
        # Read the data
        spts = drhs[records[0]].npts
        data = np.nan * np.ones((len(channels), spts * len(records)))

        for i, recind in enumerate(records):
            if spts != drhs[recind].npts:
                raise ValueError(
                    f"Record {recind} corrupted in file '{filename.name}'."
                )

            # Skip record header
            fid.seek(BYTES_HDR, 1)
            # Read data
            data_block, count = get_data_record(fid, nch, spts)
            # data_block, count = read_bit24_to_int(fid, nch, spts)
            if count != nch * spts:
                raise ValueError(
                    f"Record {recind} corrupted in file '{filename.name}'."
                )

            # Keep only selected channels
            data_block = data_block[:, channels]

            # Store data
            data[:, i * spts : (i + 1) * spts] = data_block.T

    return data.T, header1


def _read_header(fid: BinaryIO) -> tuple[SHRUHeader, int]:
    """Read a SHRU data record header.

    Args:
        fid (BinaryIO): File to read.

    Returns:
        tuple[SHRUHeader, int]: Header and status.
    """
    status = 0

    rhkey = fid.read(4).decode("utf-8")
    # Check if end of file:
    if rhkey == "":
        status = -1
        return None, status

    date = array(
        "i", [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]]
    )
    time = array(
        "i", [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]]
    )
    microsec = struct.unpack(">H", fid.read(2))[0]
    rec = struct.unpack(">H", fid.read(2))[0]
    ch = struct.unpack(">H", fid.read(2))[0]
    npts = struct.unpack(">i", fid.read(4))[0]
    rhfs = struct.unpack(">f", fid.read(4))[0]
    unused = array("i", struct.unpack("BB", fid.read(2)))
    rectime = struct.unpack(">I", fid.read(4))[0]
    rhlat = fid.read(16).decode("utf-8")
    rhlng = fid.read(16).decode("utf-8")
    nav120 = array("f", struct.unpack("28I", fid.read(28 * 4)))
    nav115 = array("f", struct.unpack("28I", fid.read(28 * 4)))
    nav110 = array("f", struct.unpack("28I", fid.read(28 * 4)))
    POS = fid.read(128).decode("utf-8")
    unused2 = array("b", fid.read(208))
    nav_day = struct.unpack(">h", fid.read(2))[0]
    nav_hour = struct.unpack(">h", fid.read(2))[0]
    nav_min = struct.unpack(">h", fid.read(2))[0]
    nav_sec = struct.unpack(">h", fid.read(2))[0]
    lblnav_flag = struct.unpack(">h", fid.read(2))[0]
    unused3 = array("b", fid.read(2))
    reclen = struct.unpack(">I", fid.read(4))[0]
    acq_day = struct.unpack(">h", fid.read(2))[0]
    acq_hour = struct.unpack(">h", fid.read(2))[0]
    acq_min = struct.unpack(">h", fid.read(2))[0]
    acq_sec = struct.unpack(">h", fid.read(2))[0]
    acq_recnum = struct.unpack(">h", fid.read(2))[0]
    ADC_tagbyte = struct.unpack(">h", fid.read(2))[0]
    glitchcode = struct.unpack(">h", fid.read(2))[0]
    bootflag = struct.unpack(">h", fid.read(2))[0]
    internal_temp = fid.read(16).decode("utf-8")
    bat_voltage = fid.read(16).decode("utf-8")
    bat_current = fid.read(16).decode("utf-8")
    drh_status = fid.read(16).decode("utf-8")
    proj = fid.read(16).decode("utf-8")
    shru_num = fid.read(16).decode("utf-8")
    vla = fid.read(16).decode("utf-8")
    hla = fid.read(16).decode("utf-8")
    filename = fid.read(32).decode("utf-8")
    record = fid.read(16).decode("utf-8")
    adate = fid.read(16).decode("utf-8")
    atime = fid.read(16).decode("utf-8")
    file_length = struct.unpack(">I", fid.read(4))[0]
    total_records = struct.unpack(">I", fid.read(4))[0]
    unused4 = array("b", fid.read(2))
    adc_mode = struct.unpack(">h", fid.read(2))[0]
    adc_clk_code = struct.unpack(">h", fid.read(2))[0]
    unused5 = array("b", fid.read(2))
    timebase = struct.unpack(">i", fid.read(4))[0]
    unused6 = array("b", fid.read(12))
    unused7 = array("b", fid.read(12))
    rhkeyl = fid.read(4).decode("utf-8")

    if rhkey == "DATA" and rhkeyl == "ADAT":
        raise ValueError("Record header keys do not match!")

    return (
        SHRUHeader(
            rhkey,
            date,
            time,
            microsec,
            rec,
            ch,
            npts,
            rhfs,
            unused,
            rectime,
            rhlat,
            rhlng,
            nav120,
            nav115,
            nav110,
            POS,
            unused2,
            nav_day,
            nav_hour,
            nav_min,
            nav_sec,
            lblnav_flag,
            unused3,
            reclen,
            acq_day,
            acq_hour,
            acq_min,
            acq_sec,
            acq_recnum,
            ADC_tagbyte,
            glitchcode,
            bootflag,
            internal_temp,
            bat_voltage,
            bat_current,
            drh_status,
            proj,
            shru_num,
            vla,
            hla,
            filename,
            record,
            adate,
            atime,
            file_length,
            total_records,
            unused4,
            adc_mode,
            adc_clk_code,
            unused5,
            timebase,
            unused6,
            unused7,
            rhkeyl,
        ),
        status,
    )


def read_shru_headers(filename: Path) -> list[SHRUHeader]:
    """Read all data record headers from a SHRU file.

    Args:
        filename (Path): File to read.

    Returns:
        list[SHRUHeader]: Data record headers.
    """
    with open(filename, "rb") as fid:
        fid.seek(0, 0)  # Go to the beginning of the file
        drhs = []
        sav = 0

        while True:
            drh, status = _read_header(fid)

            if status < 0:
                logging.debug(f"End of file reached; found {len(drhs)} record(s).")
                return drhs

            if drh.rhkey != "DATA":
                logging.warning(f"Bad record found at #{sav}")
                return drhs

            # Skip over data
            bytes_rec = drh.ch * drh.npts * 3
            fid.seek(bytes_rec, 1)

            sav += 1
            drhs.append(drh)


def _swap_long(x: int) -> int:
    """Swap the byte order of a 32-bit integer.

    Not currently used.

    Args:
        x (int): Integer to swap.

    Returns:
        int: Swapped integer.
    """
    if x <= (2**16 - 1):
        x_hex = f"{x:04x}0000"
    else:
        x_hex = f"{x:08x}"
    return int(x_hex[6:8] + x_hex[4:6] + x_hex[2:4] + x_hex[0:2], base=16)


def _swap_short(x: int) -> int:
    """Swap the byte order of a 16-bit integer.

    Not currently used.

    Args:
        x (int): Integer to swap.

    Returns:
        int: Swapped integer.
    """
    x_hex = f"{x:04x}"
    return int(x_hex[2:4] + x_hex[0:2], base=16)


def _swap_string(x: str) -> str:
    """Swap the byte order of a string.

    Not currently used.

    Args:
        x (str): String to swap.

    Returns:
        str: Swapped string.
    """
    k = 2 * int(len(x) / 2)
    new_str = ""
    for i in range(0, k, 2):
        new_str += x[i + 1] + x[i]
    return new_str
