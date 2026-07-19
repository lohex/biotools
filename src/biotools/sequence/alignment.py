"""Protein sequence alignment helpers."""

from __future__ import annotations

from typing import TypeAlias

import biotite.sequence as sequence
import biotite.sequence.align as align

GapPenalty: TypeAlias = int | tuple[int, int]


def _global_alignment_obj(
    seq_a: str,
    seq_b: str,
    gap_penalty: GapPenalty = (-10, -1),
) -> align.Alignment:
    """Return the optimal global protein alignment for two sequences.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.
        gap_penalty: Linear penalty or ``(gap_open, gap_extension)`` penalties.

    Returns:
        The first optimal global alignment reported by Biotite.
    """
    s1 = sequence.ProteinSequence(seq_a)
    s2 = sequence.ProteinSequence(seq_b)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    return align.align_optimal(
        s1,
        s2,
        matrix,
        gap_penalty=gap_penalty,
        local=False,
    )[0]


def global_alignment_score(
    seq_a: str,
    seq_b: str,
    gap_penalty: GapPenalty = (-10, -1),
) -> int:
    """Compute the score of the optimal global protein alignment.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.
        gap_penalty: Linear penalty or ``(gap_open, gap_extension)`` penalties.

    Returns:
        Score of the first optimal global alignment.
    """
    return _global_alignment_obj(seq_a, seq_b, gap_penalty).score


def global_alignment_seqs(
    seq_a: str,
    seq_b: str,
    gap_penalty: GapPenalty = (-10, -1),
) -> tuple[str, str]:
    """Return the gapped sequences from the optimal global alignment.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.
        gap_penalty: Linear penalty or ``(gap_open, gap_extension)`` penalties.

    Returns:
        The aligned sequences, including ``"-"`` gap characters.
    """
    alignment = _global_alignment_obj(seq_a, seq_b, gap_penalty)
    return alignment.get_gapped_sequences()


def global_alignment_identity(
    seq_a: str,
    seq_b: str,
    gap_penalty: GapPenalty = (-10, -1),
) -> float:
    """Calculate percent identity for the optimal global alignment.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.
        gap_penalty: Linear penalty or ``(gap_open, gap_extension)`` penalties.

    Returns:
        Identity percentage normalized by the ungapped length of ``seq_a``.
    """
    gapped_a, gapped_b = global_alignment_seqs(seq_a, seq_b, gap_penalty)
    matches = sum(1 for a, b in zip(gapped_a, gapped_b) if a == b)
    alignment_length = len(gapped_a.replace("-", ""))
    return (matches / alignment_length) * 100


def global_alignment_similarity(
    seq_a: str,
    seq_b: str,
    gap_penalty: GapPenalty = (-10, -1),
) -> float:
    """Calculate percent similarity for the optimal global alignment.

    Similarity is defined as a positive substitution score in Biotite's
    standard protein matrix.

    Args:
        seq_a: First amino-acid sequence.
        seq_b: Second amino-acid sequence.
        gap_penalty: Linear penalty or ``(gap_open, gap_extension)`` penalties.

    Returns:
        Similarity percentage across aligned non-gap residue pairs, or ``0.0``
        when the alignment contains no comparable pairs.
    """
    gapped_a, gapped_b = global_alignment_seqs(seq_a, seq_b, gap_penalty)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    comparable_pairs = [
        (a, b)
        for a, b in zip(gapped_a, gapped_b)
        if a != "-" and b != "-"
    ]
    if not comparable_pairs:
        return 0.0

    similar_positions = sum(
        1 for a, b in comparable_pairs if matrix[a, b] > 0
    )
    return (similar_positions / len(comparable_pairs)) * 100
