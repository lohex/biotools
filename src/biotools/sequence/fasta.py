"""FASTA input and output helpers."""

import logging

logger = logging.getLogger(__name__)


def dict_to_fasta(seq_dict, save_file):
    """Write a mapping of identifiers to sequences as a FASTA file."""
    with open(save_file, "w") as fp:
        for name, seq in seq_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{seq}\n")
    logger.info("Wrote %d sequences to %s", len(seq_dict), save_file)


def fasta_to_dict(load_file):
    """Parse a FASTA file into a dictionary keyed by record identifier."""
    with open(load_file, "r") as fp:
        fasta_lines = fp.readlines()

    seq_dict = {}
    last_name = None
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
