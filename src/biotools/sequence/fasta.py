"""FASTA input and output helpers."""

from __future__ import annotations

import logging
from os import PathLike
from typing import Mapping

logger = logging.getLogger(__name__)


def dict_to_fasta(
    seq_dict: Mapping[str, str],
    save_file: str | PathLike[str],
) -> None:
    """Write a mapping of identifiers to sequences as a FASTA file.

    Args:
        seq_dict: Mapping from FASTA headers to sequence strings.
        save_file: Destination FASTA path.
    """
    with open(save_file, "w") as fp:
        for name, seq in seq_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{seq}\n")
    logger.info("Wrote %d sequences to %s", len(seq_dict), save_file)


def fasta_to_dict(load_file: str | PathLike[str]) -> dict[str, str]:
    """Parse a FASTA file into a dictionary keyed by record identifier.

    Args:
        load_file: FASTA file to read.

    Returns:
        Mapping from complete header text, without ``">"``, to sequence.
    """
    with open(load_file, "r") as fp:
        fasta_lines = fp.readlines()

    seq_dict: dict[str, str] = {}
    last_name: str | None = None
    current_seq = ""
    for line in fasta_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if last_name is not None:
                seq_dict[last_name] = current_seq
            last_name = line[1:]
            current_seq = ""
        else:
            current_seq += line

    if last_name is not None:
        seq_dict[last_name] = current_seq

    logger.debug("Loaded %d sequences from %s", len(seq_dict), load_file)
    return seq_dict
