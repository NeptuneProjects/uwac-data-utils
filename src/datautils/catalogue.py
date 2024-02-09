# -*- coding: utf-8 -*-

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import scipy

from datautils.data import Header
from datautils.formats.formats import FileFormat, validate_file_format
from datautils.formats.shru import format_shru_headers, read_shru_headers
from datautils.formats.sio import format_sio_headers, read_sio_headers
from datautils.formats.wav import format_wav_headers, read_wav_headers
from datautils.query import FileInfoQuery
from datautils.time import convert_timestamp_to_yyd, get_timestamp, correct_clock_drift


@dataclass
class Catalogue:
    filenames: list[Path]
    timestamps: list[list[np.datetime64]]
    timestamps_orig: list[list[np.datetime64]]
    sampling_rate_orig: float
    sampling_rate: float
    fixed_gain: list[float]
    hydrophone_sensitivity: list[float]
    hydrophone_SN: list[str]

    def save_to_json(self, savepath: Path):

        mdict = {
            "SHRU": {
                "filenames": [str(f) for f in self.filenames],
                "timestamps": [
                    [np.datetime_as_string(t) for t in l] for l in self.timestamps
                ],
                "timestamps_orig": [
                    [np.datetime_as_string(t) for t in l] for l in self.timestamps_orig
                ],
                "sampling_rate_orig": self.sampling_rate_orig,
                "sampling_rate": self.sampling_rate,
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
                "timestamps": self._to_ydarray(self.timestamps),
                "timestamps_orig": self._to_ydarray(self.timestamps_orig),
                "rhfs_orig": self.sampling_rate_orig,
                "rhfs": self.sampling_rate,
                "fixed_gain": self.fixed_gain,
                "hydrophone_sensitivity": self.hydrophone_sensitivity,
                "hydrophone_SN": self.hydrophone_SN,
            }
        }
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(savepath, mdict)

    def _to_ydarray(self, list_of_datetimes: list[list[np.datetime64]]) -> np.ndarray:
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


def apply_header_formatting(
    file_format: FileFormat, *args, **kwargs
) -> tuple[list[Header], list[np.datetime64], list[np.datetime64]]:
    """Apply time corrections to a list of records."""
    if file_format == FileFormat.SHRU:
        return format_shru_headers(*args, **kwargs)
    if file_format == FileFormat.SIO:
        return format_sio_headers(*args, **kwargs)
    if file_format == FileFormat.WAV:
        return format_wav_headers(*args, **kwargs)


def build_catalogues(queries: list[FileInfoQuery]) -> None:
    for q in queries:
        catalogue = _build_catalogue(q)
        catalogue.save_to_mat(
            savepath=Path(q.data.destination) / f"{q.serial}_FileInfo.mat"
        )
        catalogue.save_to_json(
            savepath=Path(q.data.destination) / f"{q.serial}_FileInfo.json"
        )


def _build_catalogue(query: FileInfoQuery) -> Catalogue:
    files = sorted(Path(query.data.directory).glob(query.data.glob_pattern))

    if len(files) == 0:
        logging.error("No SHRU files found in directory.")
        raise FileNotFoundError("No SHRU files found in directory.")

    filenames = []
    timestamps = []
    timestamps_orig = []
    for i, f in enumerate(files):
        headers, file_format = read_headers(f, file_format=query.data.file_format)

        if len(headers) == 0:
            logging.warning(f"File {f} has no valid records.")
            continue
        logging.info(f"File {f} has {len(headers)} valid record(s).")

        filenames.append(f)
        file_timestamps_orig = []
        file_timestamps = []
        for record in headers:
            ts_orig = get_timestamp(record)
            ts = correct_clock_drift(ts_orig, query.clock)
            file_timestamps_orig.append(ts_orig)
            file_timestamps.append(ts)

        # Apply format-specific corrections
        headers, file_timestamps, file_timestamps_orig = apply_header_formatting(
            file_format=file_format,
            file_iter=i,
            headers=headers,
            file_timestamps=file_timestamps,
            file_timestamps_orig=file_timestamps_orig,
        )

        timestamps.append(file_timestamps)
        timestamps_orig.append(file_timestamps_orig)

        logging.debug(f"Header extracted from {f}.")

    # Extract sampling rate:
    sampling_rate_orig = get_sampling_rate(file_format, headers)
    sampling_rate = sampling_rate_orig / (1 + query.clock.drift_rate / 24 / 3600)

    return Catalogue(
        filenames=filenames,
        timestamps=timestamps,
        timestamps_orig=timestamps_orig,
        sampling_rate_orig=sampling_rate_orig,
        sampling_rate=sampling_rate,
        fixed_gain=[query.hydrophones.fixed_gain] * len(filenames),
        hydrophone_sensitivity=[query.hydrophones.sensitivity] * len(filenames),
        hydrophone_SN=[query.serial] * len(filenames),
    )


def get_headers_reader(
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


def get_sampling_rate(file_format: FileFormat, headers: list[Header]) -> float:
    """Get sampling rate from headers."""
    if file_format == FileFormat.SHRU:
        return headers[0].rhfs
    if file_format == FileFormat.SIO:
        return headers[0].rhfs
    if file_format == FileFormat.WAV:
        return headers[0].framerate


def read_headers(
    filename: Path, file_format: str = None
) -> tuple[list[Header], FileFormat]:
    reader, file_format = get_headers_reader(
        suffix=filename.suffix, file_format=file_format
    )
    return reader(filename), file_format
