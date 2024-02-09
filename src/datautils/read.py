# -*- coding: utf-8 -*-

from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from datautils.data import Header
from datautils.formats.formats import FileFormat, validate_file_format
from datautils.formats.shru import format_shru_headers, read_shru_headers
from datautils.formats.sio import format_sio_headers, read_sio_headers
from datautils.formats.wav import format_wav_headers, read_wav_headers


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


def read_headers(
    filename: Path, file_format: str = None
) -> tuple[list[Header], FileFormat]:
    reader, file_format = get_headers_reader(
        suffix=filename.suffix, file_format=file_format
    )
    return reader(filename), file_format


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
