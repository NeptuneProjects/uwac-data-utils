# -*- coding: utf-8 -*-

from array import array
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import struct
from typing import BinaryIO, Optional, Union

import numpy as np
import scipy

# from datautils import query

ADC_HALFSCALE = 2.5  # ADC half scale volts (+/- half scale is ADC i/p range)
ADC_MAXVALUE = 2**23  # ADC maximum halfscale o/p value, half the 2's complement range
BYTES_HDR = 1024
BYTES_PER_SAMPLE = 3

logging.basicConfig(level=logging.INFO)


class DataRecordHeader:
    ...

def read_header():
    return



@dataclass
class FileInfo:
    filenames: list[Path]
    timestamps: list[list[np.datetime64]]
    timestamps_orig: list[list[np.datetime64]]
    rhfs_orig: float
    rhfs: float
    fixed_gain: list[float]
    hydrophone_sensitivity: list[float]
    hydrophone_SN: list[str]

    def save_to_json(self, savepath: Path):
        mdict = {
            "SHRU": {
                "filenames": [str(f) for f in self.filenames],
                "timestamps": [t.tolist() for t in self.timestamps],
                "timestamps_orig": [t.tolist() for t in self.timestamps_orig],
                "rhfs_orig": self.rhfs_orig,
                "rhfs": self.rhfs,
                "fixed_gain": self.fixed_gain,
                "hydrophone_sensitivity": self.hydrophone_sensitivity,
                "hydrophone_SN": self.hydrophone_SN,
            }
        }
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        with open(savepath, "w") as f:
            json.dump(mdict, f, indent=4)

    def save_to_mat(self, savepath: Path):
        mdict = {
            "SHRU": {
                "filenames": [str(f) for f in self.filenames],
                "timestamps": self._to_array(self.timestamps),
                "timestamps_orig": self._to_array(self.timestamps_orig),
                "rhfs_orig": self.rhfs_orig,
                "rhfs": self.rhfs,
                "fixed_gain": self.fixed_gain,
                "hydrophone_sensitivity": self.hydrophone_sensitivity,
                "hydrophone_SN": self.hydrophone_SN,
            }
        }
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(savepath, mdict)

    def _to_array(self, list_of_datetimes: list[list[np.datetime64]]) -> np.ndarray:
        # 2 x M x N
        # L = number of datetime elements
        # M = number of records
        # N = number of files
        L = 2
        M = max(len(dt) for dt in list_of_datetimes)
        N = len(list_of_datetimes)
        arr = np.zeros((L, M, N), dtype=np.float64)

        for i, dt in enumerate(list_of_datetimes):
            for j, d in enumerate(dt):
                year, yd_decimal = convert_timestamp_to_yyd(d)
                arr[0, j, i] = year
                arr[1, j, i] = yd_decimal

        return arr



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




def chk_read(count: int) -> int:
    """Check if the read count is correct."""
    if count == 0:
        return -1
    return 0


def convert_24bit_to_int(
    raw_data: bytes, nch: int, spts: int
) -> tuple[np.ndarray, int]:
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


def convert_timestamp_to_yyd(timestamp: np.datetime64) -> tuple[int, float]:
    Y, us = [timestamp.astype(f"datetime64[{t}]") for t in ["Y", "us"]]
    year = Y.astype(int) + 1970
    yd = (us - np.datetime64(f"{year - 1}-12-31")) / np.timedelta64(1, "D")
    return year, yd


def convert_to_datetime(
    year: int, yd: int, minute: int, millisec: int, microsec: int
) -> np.datetime64:
    base_date = np.datetime64(f"{year}-01-01")
    return (
        base_date
        + np.timedelta64(yd - 1, "D")
        + np.timedelta64(minute, "m")
        + np.timedelta64(millisec, "ms")
        + np.timedelta64(microsec, "us")
    )


def convert_to_voltage(data: np.ndarray, fixed_gain: list[float]) -> np.ndarray:
    norm_factor = ADC_HALFSCALE / ADC_MAXVALUE / np.array(fixed_gain)
    return data * norm_factor[np.newaxis, :]


# def correct_clock_drift(
#     timestamp: np.datetime64, clock: query.ClockParameters
# ) -> np.datetime64:
#     """Correct clock drift in a timestamp."""
#     days_diff = (timestamp - clock.time_check_0) / np.timedelta64(1, "D")
#     drift = clock.drift_rate * days_diff
#     return timestamp + np.timedelta64(int(1e6 * drift), "us")


# def get_data_record(fid: BinaryIO, nch: int, spts: int) -> tuple[np.ndarray, int]:
#     total_bytes = nch * spts * BYTES_PER_SAMPLE
#     data_bytes = fid.read(total_bytes)
#     if len(data_bytes) != total_bytes:
#         raise ValueError("Failed to read all data bytes.")
#     return convert_24bit_to_int(data_bytes, nch, spts)


# def get_num_channels(drhs: list[DataRecordHeader]) -> int:
#     for record in drhs:
#         if record.ch != drhs[0].ch:
#             raise Warning("Number of channels varies across records.")
#     return drhs[0].ch


# def get_timestamp(drh: DataRecordHeader) -> np.datetime64:
#     """Return the timestamp of a data record header."""
#     year = drh.date[0]
#     yd = drh.date[1]
#     minute = drh.time[0]
#     millisec = drh.time[1]
#     microsec = drh.microsec
#     return convert_to_datetime(year, yd, minute, millisec, microsec)


# def read_24bit_data(
#     filename: Path,
#     records: Union[int, list[int]],
#     channels: Union[int, list[int]],
#     fixed_gain: Union[int, list[int]] = 20.0,
#     drhs: Optional[list[DataRecordHeader]] = None,
# ) -> tuple[np.ndarray, DataRecordHeader]:

#     if not isinstance(records, list):
#         records = [records]
#     if not isinstance(channels, list):
#         channels = [channels]
#     if not isinstance(fixed_gain, list):
#         fixed_gain = [fixed_gain] * len(channels)
#     if isinstance(fixed_gain, list) and len(fixed_gain) != len(channels):
#         raise ValueError("Length of fixed_gain must match length of channels.")

#     if drhs is None:
#         drhs = read_all_drh(filename)

#     nch = get_num_channels(drhs)
#     if any((i > nch - 1) for i in channels):
#         raise ValueError(
#             f"Channel {len(channels)} requested but only got {nch} from header."
#         )

#     header1 = drhs[records[0]]

#     # Calculate the number of bytes to skip to start reading data
#     skip_bytes = sum([drh.reclen for drh in drhs[0 : records[0]]])

#     with open(filename, "rb") as fid:
#         fid.seek(skip_bytes, 0)  # Skip to selected record
#         # Read the data
#         spts = drhs[records[0]].npts
#         data = np.nan * np.ones((len(channels), spts * len(records)))

#         for i, recind in enumerate(records):
#             if spts != drhs[recind].npts:
#                 raise ValueError(
#                     f"Record {recind} corrupted in file '{filename.name}'."
#                 )

#             # Skip record header
#             fid.seek(BYTES_HDR, 1)
#             # Read data
#             data_block, count = get_data_record(fid, nch, spts)
#             # data_block, count = read_bit24_to_int(fid, nch, spts)
#             if count != nch * spts:
#                 raise ValueError(
#                     f"Record {recind} corrupted in file '{filename.name}'."
#                 )

#             # Keep only selected channels
#             data_block = data_block[:, channels]
#             # Convert data to voltage
#             data_v = convert_to_voltage(data_block, fixed_gain)
#             # Store data
#             data[:, i * spts : (i + 1) * spts] = data_v.T

#     return data.T, header1


# def read_all_drh(filename: Path) -> list[DataRecordHeader]:
#     with open(filename, "rb") as fid:
#         fid.seek(0, 0)  # Go to the beginning of the file
#         drhs = []
#         sav = 0

#         while True:
#             drh, status = read_header(fid)

#             if status < 0:
#                 logging.debug(f"End of file reached; found {len(drhs)} record(s).")
#                 return drhs

#             if drh.rhkey != "DATA":
#                 logging.warning(f"Bad record found at #{sav}")
#                 return drhs

#             # Skip over data
#             bytes_rec = drh.ch * drh.npts * 3
#             fid.seek(bytes_rec, 1)

#             sav += 1
#             drhs.append(drh)


# def read_header(fid: BinaryIO) -> tuple[DataRecordHeader, int]:
#     status = 0

#     rhkey = fid.read(4).decode("utf-8")
#     # Check if end of file:
#     if rhkey == "":
#         status = -1
#         return None, status

#     date = array(
#         "i", [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]]
#     )
#     time = array(
#         "i", [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]]
#     )
#     microsec = struct.unpack(">H", fid.read(2))[0]
#     rec = struct.unpack(">H", fid.read(2))[0]
#     ch = struct.unpack(">H", fid.read(2))[0]
#     npts = struct.unpack(">i", fid.read(4))[0]
#     rhfs = struct.unpack(">f", fid.read(4))[0]
#     unused = array("i", struct.unpack("BB", fid.read(2)))
#     rectime = struct.unpack(">I", fid.read(4))[0]
#     rhlat = fid.read(16).decode("utf-8")
#     rhlng = fid.read(16).decode("utf-8")
#     nav120 = array("f", struct.unpack("28I", fid.read(28 * 4)))
#     nav115 = array("f", struct.unpack("28I", fid.read(28 * 4)))
#     nav110 = array("f", struct.unpack("28I", fid.read(28 * 4)))
#     POS = fid.read(128).decode("utf-8")
#     unused2 = array("b", fid.read(208))
#     nav_day = struct.unpack(">h", fid.read(2))[0]
#     nav_hour = struct.unpack(">h", fid.read(2))[0]
#     nav_min = struct.unpack(">h", fid.read(2))[0]
#     nav_sec = struct.unpack(">h", fid.read(2))[0]
#     lblnav_flag = struct.unpack(">h", fid.read(2))[0]
#     unused3 = array("b", fid.read(2))
#     reclen = struct.unpack(">I", fid.read(4))[0]
#     acq_day = struct.unpack(">h", fid.read(2))[0]
#     acq_hour = struct.unpack(">h", fid.read(2))[0]
#     acq_min = struct.unpack(">h", fid.read(2))[0]
#     acq_sec = struct.unpack(">h", fid.read(2))[0]
#     acq_recnum = struct.unpack(">h", fid.read(2))[0]
#     ADC_tagbyte = struct.unpack(">h", fid.read(2))[0]
#     glitchcode = struct.unpack(">h", fid.read(2))[0]
#     bootflag = struct.unpack(">h", fid.read(2))[0]
#     internal_temp = fid.read(16).decode("utf-8")
#     bat_voltage = fid.read(16).decode("utf-8")
#     bat_current = fid.read(16).decode("utf-8")
#     drh_status = fid.read(16).decode("utf-8")
#     proj = fid.read(16).decode("utf-8")
#     shru_num = fid.read(16).decode("utf-8")
#     vla = fid.read(16).decode("utf-8")
#     hla = fid.read(16).decode("utf-8")
#     filename = fid.read(32).decode("utf-8")
#     record = fid.read(16).decode("utf-8")
#     adate = fid.read(16).decode("utf-8")
#     atime = fid.read(16).decode("utf-8")
#     file_length = struct.unpack(">I", fid.read(4))[0]
#     total_records = struct.unpack(">I", fid.read(4))[0]
#     unused4 = array("b", fid.read(2))
#     adc_mode = struct.unpack(">h", fid.read(2))[0]
#     adc_clk_code = struct.unpack(">h", fid.read(2))[0]
#     unused5 = array("b", fid.read(2))
#     timebase = struct.unpack(">i", fid.read(4))[0]
#     unused6 = array("b", fid.read(12))
#     unused7 = array("b", fid.read(12))
#     rhkeyl = fid.read(4).decode("utf-8")

#     if rhkey == "DATA" and rhkeyl == "ADAT":
#         raise ValueError("Record header keys do not match!")

#     return (
#         DataRecordHeader(
#             rhkey,
#             date,
#             time,
#             microsec,
#             rec,
#             ch,
#             npts,
#             rhfs,
#             unused,
#             rectime,
#             rhlat,
#             rhlng,
#             nav120,
#             nav115,
#             nav110,
#             POS,
#             unused2,
#             nav_day,
#             nav_hour,
#             nav_min,
#             nav_sec,
#             lblnav_flag,
#             unused3,
#             reclen,
#             acq_day,
#             acq_hour,
#             acq_min,
#             acq_sec,
#             acq_recnum,
#             ADC_tagbyte,
#             glitchcode,
#             bootflag,
#             internal_temp,
#             bat_voltage,
#             bat_current,
#             drh_status,
#             proj,
#             shru_num,
#             vla,
#             hla,
#             filename,
#             record,
#             adate,
#             atime,
#             file_length,
#             total_records,
#             unused4,
#             adc_mode,
#             adc_clk_code,
#             unused5,
#             timebase,
#             unused6,
#             unused7,
#             rhkeyl,
#         ),
#         status,
#     )


# def read_shru(fquery: query.FileInfoQuery) -> FileInfo:
#     """Load a SHRU file and return a SHRU_FileInfo object."""
#     files = sorted(Path(fquery.data.directory).glob(fquery.data.glob_pattern))

#     if len(files) == 0:
#         logging.error("No SHRU files found in directory.")
#         raise FileNotFoundError("No SHRU files found in directory.")

#     filenames = []
#     timestamps = []
#     timestamps_orig = []
#     for i, f in enumerate(files):
#         drhs = read_all_drh(f)
#         if len(drhs) == 0:
#             logging.warning(f"File {f} has no valid records.")
#             continue
#         logging.info(f"File {f} has {len(drhs)} valid record(s).")

#         filenames.append(f)

#         file_timestamps_orig = []
#         file_timestamps = []
#         for record in drhs:
#             ts_orig = get_timestamp(record)
#             ts = correct_clock_drift(ts_orig, fquery.clock)
#             file_timestamps_orig.append(ts_orig)
#             file_timestamps.append(ts)

#         # The time stamp of the first record in the 4-channel SHRU file
#         # seems to be rounded to seconds. The following corrects this:
#         if i == 0 and len(drhs) > 1:
#             offset_for_first_record = np.timedelta64(
#                 int(1e6 * drhs[0].npts / drhs[0].rhfs), "us"
#             )
#             file_timestamps[0] = file_timestamps[1] - offset_for_first_record
#             file_timestamps_orig[0] = file_timestamps_orig[1] - offset_for_first_record

#         timestamps.append(file_timestamps)
#         timestamps_orig.append(file_timestamps_orig)

#         logging.debug(f"Header extracted from {f}.")

#     # Extract sampling rate from the first record:
#     rhfs_orig = drhs[0].rhfs
#     rhfs = drhs[0].rhfs / (1 + fquery.clock.drift_rate / 24 / 3600)

#     return FileInfo(
#         filenames=filenames,
#         timestamps=timestamps,
#         timestamps_orig=timestamps_orig,
#         rhfs_orig=rhfs_orig,
#         rhfs=rhfs,
#         fixed_gain=fquery.hydrophones.fixed_gain,
#         hydrophone_sensitivity=fquery.hydrophones.sensitivity,
#         hydrophone_SN=fquery.hydrophones.serial_number,
#     )


# def swap_long(x: int) -> int:
#     if x <= (2**16 - 1):
#         x_hex = f"{x:04x}0000"
#     else:
#         x_hex = f"{x:08x}"
#     return int(x_hex[6:8] + x_hex[4:6] + x_hex[2:4] + x_hex[0:2], base=16)


# def swap_short(x: int) -> int:
#     x_hex = f"{x:04x}"
#     return int(x_hex[2:4] + x_hex[0:2], base=16)


# def swap_string(x: str) -> str:
#     k = 2 * int(len(x) / 2)
#     new_str = ""
#     for i in range(0, k, 2):
#         new_str += x[i + 1] + x[i]
#     return new_str

