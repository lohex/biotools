"""Tests for protein sequence alignment helpers."""

import unittest

from biotools.sequence import global_alignment_similarity


class GlobalAlignmentSimilarityTests(unittest.TestCase):
    def test_identical_similar_and_dissimilar_sequences(self) -> None:
        cases = (
            ("AAAA", "AAAA", 100.0),
            ("IIII", "LLLL", 100.0),
            ("AAAA", "WWWW", 0.0),
        )

        for seq_a, seq_b, expected_similarity in cases:
            with self.subTest(seq_a=seq_a, seq_b=seq_b):
                self.assertEqual(
                    global_alignment_similarity(seq_a, seq_b),
                    expected_similarity,
                )


if __name__ == "__main__":
    unittest.main()
