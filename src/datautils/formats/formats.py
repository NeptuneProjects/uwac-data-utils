# -*- coding: utf-8 -*-

from enum import Enum
from typing import Optional


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


def get_file_format(desc: str) -> str:
    """Get file format from suffix."""
    if SHRUFileFormat.is_format(desc):
        return FileFormat.SHRU
    if SIOFileFormat.is_format(desc):
        return FileFormat.SIO
    if WAVFileFormat.is_format(desc):
        return FileFormat.WAV
    raise ValueError(f"File format cannot be inferred from file extension '{desc}'.")


def validate_file_format(
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
