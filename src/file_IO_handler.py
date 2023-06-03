"""File IO Handler.

Functions to read and write content to local plaintext files.
"""

import os
import json
import gzip
from typing import Any
import pathlib


def get_plaintext_file_contents(path_to_file: pathlib.Path) -> str:
    """Return the contents of plaintext file.

    Args:
        path_to_file: path to file.

    Returns:
        Contents of file.
    """
    with open(path_to_file, "r") as f:
        return f.read()


def save_json(
    obj: Any, filename: pathlib.Path, make_dirs_if_necessary=False, indent=2, **kwargs
) -> None:
    """Wrapper on `json.dump(...)` to handle compression.

    Saves compressed file (`.json.gz`) if filename ends with `.gz`.

    Args:
        obj: python dict or object to save to json.
        filename: path to save `.json` or `.json.gz` file.
        make_dirs_if_necessary: flag
        indent: indent level when saving json
        **kwargs: other args for `json.dump(...)`

    Returns:
        None
    """
    if make_dirs_if_necessary:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.suffix == ".gz":
        with gzip.open(filename, "wt") as f:
            json.dump(obj, f, indent=indent, **kwargs)
    with open(filename, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=indent, **kwargs)

    return None


def load_json(filename: pathlib.Path) -> Any:
    """Wrapper on `json.load(...)` to handle compression.

    Loads compressed file if filename ends with `.gz`.

    Args:
        filename: path to `.json` or `.json.gz` file.

    Returns:
        Any.
    """
    if filename.suffix == ".gz":
        with gzip.open(filename, "rt") as f:
            return json.load(f)
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)
