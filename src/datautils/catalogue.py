# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
import polars as pl
import scipy

from datautils.data import DataStream, read
from datautils.formats.formats import FileFormat, validate_file_format
from datautils.formats.shru import read_shru_headers
from datautils.formats.sio import read_sio_headers
from datautils.formats.wav import read_wav_headers
from datautils.query import CatalogueQuery, FileInfoQuery
from datautils.time import (
    convert_timestamp_to_yyd,
    convert_to_datetime,
    correct_clock_drift,
)


class CatalogueFileFormat(Enum):
    CSV = "csv"
    JSON = "json"
    MAT = "mat"


class FileCallbackHandler:
    def __init__(self, file_format: FileFormat):
        self.file_format = file_format

    def format_records(self, records: list[Record]) -> list[Record]:
        if self.file_format == FileFormat.SHRU:
            return self.format_shru_records(records)
        return

    @staticmethod
    def format_shru_records(records: list[Record]) -> list[Record]:
        # Calculate the time offset between the first and second records
        if len(records) == 1:
            return records
        offset_for_first_record = np.timedelta64(
            int(1e6 * records[0].npts / records[0].sampling_rate_orig), "us"
        )
        records[0].timestamp = records[1].timestamp - offset_for_first_record
        records[0].timestamp_orig = records[1].timestamp_orig - offset_for_first_record
        return records

    @staticmethod
    def format_sio_records(records: list[Record]) -> list[Record]:
        return records

    @staticmethod
    def format_wav_records(records: list[Record]) -> list[Record]:
        return records


class Header(Protocol): ...


@dataclass
class Record:
    filename: Path
    record_number: int
    file_format: FileFormat
    npts: int
    timestamp: np.datetime64
    timestamp_orig: np.datetime64
    sampling_rate: float
    sampling_rate_orig: float
    fixed_gain: float
    hydrophone_sensitivity: float
    hydrophone_SN: str


class RecordCatalogue:
    def __init__(self, records: Optional[list[Record]] = None):
        self.records = records
        if records is not None:
            self.df = self._records_to_polars_df()

    def build(self, query: FileInfoQuery) -> RecordCatalogue:
        files = sorted(Path(query.data.directory).glob(query.data.glob_pattern))

        if len(files) == 0:
            logging.error("No SHRU files found in directory.")
            raise FileNotFoundError("No SHRU files found in directory.")

        records = []
        for f in files:
            headers, file_format = self._read_headers(
                f, file_format=query.data.file_format
            )
            records_from_file = []
            callback = FileCallbackHandler(file_format)
            for i, header in enumerate(headers):
                ts_orig = self._get_timestamp(header)
                ts = correct_clock_drift(ts_orig, query.clock)
                records_from_file.append(
                    Record(
                        filename=f,
                        record_number=i,
                        file_format=file_format,
                        npts=header.npts,
                        timestamp=ts,
                        timestamp_orig=ts_orig,
                        sampling_rate_orig=self._get_sampling_rate(
                            file_format, headers
                        ),
                        sampling_rate=self._get_sampling_rate(file_format, headers)
                        / (1 + query.clock.drift_rate / 24 / 3600),
                        fixed_gain=query.hydrophones.fixed_gain,
                        hydrophone_sensitivity=query.hydrophones.sensitivity,
                        hydrophone_SN=query.hydrophones.serial_number,
                    )
                )
            corrected_records = callback.format_records(records_from_file)
            records.extend(corrected_records)

        self.records = records
        self.df = self._records_to_polars_df()
        return self

    def _format_df_for_csv(self, df: pl.DataFrame) -> pl.DataFrame:
        def _to_list(lst: list):
            return ",".join([str(i) for i in lst])

        return df.with_columns(
            pl.col("fixed_gain").apply(_to_list),
            pl.col("hydrophone_sensitivity").apply(_to_list),
            pl.col("hydrophone_SN").apply(_to_list),
        )

    @staticmethod
    def _get_headers_reader(
        suffix: Optional[str] = None, file_format: Optional[str] = None
    ) -> tuple[callable, FileFormat]:
        """Factory to get header reader for file format."""
        file_format = validate_file_format(suffix, file_format)
        if file_format == FileFormat.SHRU:
            return read_shru_headers, file_format
        if file_format == FileFormat.SIO:
            return read_sio_headers, file_format
        if file_format == FileFormat.WAV:
            return read_wav_headers, file_format
        raise ValueError(f"File format {file_format} is not recognized.")

    @staticmethod
    def _get_sampling_rate(file_format: FileFormat, headers: list[Header]) -> float:
        """Get sampling rate from headers."""
        if file_format == FileFormat.SHRU:
            return headers[0].rhfs
        if file_format == FileFormat.SIO:
            return headers[0].rhfs
        if file_format == FileFormat.WAV:
            return headers[0].framerate

    @staticmethod
    def _get_timestamp(header: Header) -> np.datetime64:
        """Return the timestamp of a data record header."""
        year = header.date[0]
        yd = header.date[1]
        minute = header.time[0]
        millisec = header.time[1]
        microsec = header.microsec
        return convert_to_datetime(year, yd, minute, millisec, microsec)

    def _read_headers(
        self, filename: Path, file_format: str = None
    ) -> tuple[list[Header], FileFormat]:
        reader, file_format = self._get_headers_reader(
            suffix=filename.suffix, file_format=file_format
        )
        return reader(filename), file_format

    def _records_to_dfdict(self) -> dict:
        return {
            "filename": [str(record.filename) for record in self.records],
            "record_number": [record.record_number for record in self.records],
            "file_format": [record.file_format.name for record in self.records],
            "npts": [record.npts for record in self.records],
            "timestamp": [record.timestamp.astype("int64") for record in self.records],
            "timestamp_orig": [
                record.timestamp_orig.astype("int64") for record in self.records
            ],
            "sampling_rate": [record.sampling_rate for record in self.records],
            "sampling_rate_orig": [
                record.sampling_rate_orig for record in self.records
            ],
            "fixed_gain": [record.fixed_gain for record in self.records],
            "hydrophone_sensitivity": [
                record.hydrophone_sensitivity for record in self.records
            ],
            "hydrophone_SN": [record.hydrophone_SN for record in self.records],
        }

    def _records_to_mdict(self) -> dict:
        filenames = list(set([str(record.filename) for record in self.records]))

        timestamps = []
        timestamps_orig = []
        for filename in filenames:
            ts = (
                self.df.filter(pl.col("filename") == filename)
                .select("timestamp")
                .to_series()
                .cast(pl.Int64)
            )
            ts_orig = (
                self.df.filter(pl.col("filename") == filename)
                .select("timestamp_orig")
                .to_series()
                .cast(pl.Int64)
            )

            timestamps.append([np.datetime64(ts[i], "us") for i in range(len(ts))])
            timestamps_orig.append(
                [np.datetime64(ts_orig[i], "us") for i in range(len(ts_orig))]
            )

        return {
            self.records[0].file_format.name: {
                "filenames": filenames,
                "timestamps": self._to_ydarray(timestamps),
                "timestamps_orig": self._to_ydarray(timestamps_orig),
                "rhfs_orig": self.records[0].sampling_rate_orig,
                "rhfs": self.records[0].sampling_rate,
                "fixed_gain": self.records[0].fixed_gain,
                "hydrophone_sensitivity": self.records[0].hydrophone_sensitivity,
                "hydrophone_SN": self.records[0].hydrophone_SN,
            }
        }

    def _records_to_polars_df(self) -> pl.DataFrame:
        return (
            pl.DataFrame(self._records_to_dfdict())
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            .with_columns(pl.col("timestamp_orig").cast(pl.Datetime("us")))
        )

    def save(self, savepath: Path, fmt: Union[str, list[str]] = "csv"):
        if isinstance(fmt, str):
            fmt = [fmt]
        if not all(f in CatalogueFileFormat for f in fmt):
            raise ValueError(f"File format {fmt} is not recognized.")

        savepath.parent.mkdir(parents=True, exist_ok=True)

        if CatalogueFileFormat.CSV.name.lower() in fmt:
            self.write_csv(savepath.parent / (savepath.stem + ".csv"))
        if CatalogueFileFormat.JSON.name.lower() in fmt:
            self.write_json(savepath.parent / (savepath.stem + ".json"))
        if CatalogueFileFormat.MAT.name.lower() in fmt:
            self.write_mat(savepath.parent / (savepath.stem + ".mat"))

    @staticmethod
    def _to_ydarray(list_of_datetimes: list[list[np.datetime64]]) -> np.ndarray:
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

    def write_csv(self, savepath: Path):
        df_out = self._format_df_for_csv(self.df)
        df_out.write_csv(savepath)

    def write_json(self, savepath: Path):
        self.df.write_json(savepath, pretty=True)

    def write_mat(self, savepath: Path):
        mdict = self._records_to_mdict()
        scipy.io.savemat(savepath, mdict)


# def build_catalogues(queries: list[FileInfoQuery]) -> None:
#     for q in queries:
#         catalogue = _build_catalogue(q)
#         catalogue.save_to_mat(
#             savepath=Path(q.data.destination) / f"{q.serial}_FileInfo.mat"
#         )
#         catalogue.save_to_json(
#             savepath=Path(q.data.destination) / f"{q.serial}_FileInfo.json"
#         )


# def read_data_from_catalogue(query: CatalogueQuery) -> DataStream:
#     # TODO: Write function that takes a catalogue query and returns a DataStream object
#     """Loads data from file."""
#     # 1. Load catalogue:
#     catalogue = read_catalogue(query.catalogue)

#     # 2. Filter files by time:
#     selected_files = select_files_by_time(
#         catalogue.filenames, query.time_start, query.time_end
#     )
#     print(catalogue.filenames)
#     print(selected_files)

#     # 3. Load data from files:
#     # read(catalogue.filenames, query.time_start, query.time_end, query.channels)

#     pass


# def select_files_by_time(
#     filenames: list[Path], time_start: np.datetime64, time_end: np.datetime64
# ) -> list[Path]:
#     """Select files by time."""
#     if time_start > time_end:
#         raise ValueError("time_start must be less than time_end.")
#     if np.isnat(time_start) and np.isnat(time_end):
#         return filenames
#     if time_start is not None and np.isnat(time_end):
#         return [
#             f for f in filenames if get_timestamp(read_headers(f)[0][0]) >= time_start
#         ]
#     if np.isnat(time_start) and time_end is not None:
#         return [
#             f for f in filenames if get_timestamp(read_headers(f)[0][0]) <= time_end
#         ]
#     return [
#         f
#         for f in filenames
#         if time_start <= get_timestamp(read_headers(f)[0][0]) <= time_end
#     ]


# def read_catalogue(filepath: Path) -> Catalogue:
#     with open(filepath, "r") as f:
#         mdict = json.load(f)

#     file_format = validate_file_format(file_format=list(mdict.keys()).pop()).value

#     return Catalogue(
#         file_format=file_format,
#         filenames=[Path(f) for f in mdict[file_format]["filenames"]],
#         timestamps=[
#             [np.datetime64(t) for t in l] for l in mdict[file_format]["timestamps"]
#         ],
#         timestamps_orig=[
#             [np.datetime64(t) for t in l] for l in mdict[file_format]["timestamps_orig"]
#         ],
#         sampling_rate_orig=mdict[file_format]["sampling_rate_orig"],
#         sampling_rate=mdict[file_format]["sampling_rate"],
#         fixed_gain=mdict[file_format]["fixed_gain"],
#         hydrophone_sensitivity=mdict[file_format]["hydrophone_sensitivity"],
#         hydrophone_SN=mdict[file_format]["hydrophone_SN"],
#     )
