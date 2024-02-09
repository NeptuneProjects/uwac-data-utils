# -*- coding: utf-8 -*-

from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from datautils.formats.shru import format_shru_headers, read_shru_headers
from datautils.formats.sio import read_sio_headers, format_sio_headers
from datautils.formats.wav import read_wav_headers, format_wav_headers


class FormatCheckerMixin:
    @classmethod
    def is_format(cls, extension: str) -> bool:
        normalized_extension = extension.lower()
        return any(normalized_extension == item.value.lower() for item in cls)


class SHRUFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "SHRU"
    D23 = ".D23"


class SIOFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "SIO"
    SIO = ".SIO"


class WAVFileFormat(FormatCheckerMixin, Enum):
    FORMAT = "WAV"
    WAV = ".WAV"


class FileFormat(Enum):
    SHRU = SHRUFileFormat.FORMAT.value
    SIO = SIOFileFormat.FORMAT.value
    WAV = WAVFileFormat.FORMAT.value


class Header(Protocol): ...


def get_file_format(desc: str) -> str:
    """Get file format from suffix."""
    if SHRUFileFormat.is_format(desc):
        return FileFormat.SHRU
    if SIOFileFormat.is_format(desc):
        return FileFormat.SIO
    if WAVFileFormat.is_format(desc):
        return FileFormat.WAV
    raise ValueError(f"File format cannot be inferred from file extension '{desc}'.")


def get_headers_reader(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> tuple[callable, FileFormat]:
    """Factory to get header reader for file format."""
    file_format = _validate_file_format(suffix, file_format)
    if file_format == FileFormat.SHRU:
        return read_shru_headers, file_format
    if file_format == FileFormat.SIO:
        return read_sio_headers, file_format
    if file_format == FileFormat.WAV:
        return read_wav_headers, file_format
    raise ValueError(f"File format {file_format} is not recognized.")


def read_headers(
    filename: Path, file_format: str = None
) -> tuple[list[Header], FileFormat]:
    reader, file_format = get_headers_reader(
        suffix=filename.suffix, file_format=file_format
    )
    return reader(filename), file_format


def _validate_file_format(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> FileFormat:
    if suffix is None and file_format is None:
        raise ValueError("An argument 'suffix' or 'file_format' must be provided.")
    if file_format is not None:
        file_format = FileFormat(file_format)
    if file_format is None:
        file_format = get_file_format(suffix)
    if get_file_format(suffix) != file_format:
        raise ValueError("The provided 'suffix' and 'file_format' are not consistent.")
    return file_format


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


def get_sampling_rate(file_format: FileFormat, headers: list[Header]) -> float:
    """Get sampling rate from headers."""
    if file_format == FileFormat.SHRU:
        return headers[0].rhfs
    if file_format == FileFormat.SIO:
        return headers[0].rhfs
    if file_format == FileFormat.WAV:
        return headers[0].framerate
