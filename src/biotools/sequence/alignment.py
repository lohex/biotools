"""Protein sequence alignment helpers."""

import biotite.sequence as sequence
import biotite.sequence.align as align


def _global_alignment_obj(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Return the optimal global protein alignment for two sequences."""
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


def global_alignment_score(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Compute the score of the optimal global protein alignment."""
    return _global_alignment_obj(seq_a, seq_b, gap_penalty).score


def global_alignment_seqs(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Return the gapped sequences from the optimal global alignment."""
    alignment = _global_alignment_obj(seq_a, seq_b, gap_penalty)
    return alignment.get_gapped_sequences()


def global_alignment_identity(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Calculate percent identity for the optimal global alignment."""
    gapped_a, gapped_b = global_alignment_seqs(seq_a, seq_b, gap_penalty)
    matches = sum(1 for a, b in zip(gapped_a, gapped_b) if a == b)
    alignment_length = len(gapped_a.replace("-", ""))
    return (matches / alignment_length) * 100


def global_alignment_similarity(seq_a, seq_b, gap_penalty=(-10, -1)):
    """Calculate percent similarity for the optimal global alignment."""
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
