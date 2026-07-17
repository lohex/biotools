"""Backward-compatible facade for :mod:`biotools.sequence`.

New code may import from the focused ``biotools.sequence`` modules directly.
Existing imports from ``biotools.seqtools`` remain supported.
"""

from .sequence import (
    BlastSearch,
    dict_to_fasta,
    fasta_to_dict,
    global_alignment_identity,
    global_alignment_score,
    global_alignment_seqs,
    global_alignment_similarity,
)
from .sequence.alignment import _global_alignment_obj

__all__ = [
    "BlastSearch",
    "dict_to_fasta",
    "fasta_to_dict",
    "global_alignment_identity",
    "global_alignment_score",
    "global_alignment_seqs",
    "global_alignment_similarity",
]
