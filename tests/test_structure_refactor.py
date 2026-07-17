"""Regression tests for the split structure API and compatibility facades."""

import unittest

import numpy as np
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools import pdbtools
from biotools.structure import align_homologs, rename_chain, reset_index


def _residue(number, coordinate, name="ALA"):
    residue = Residue.Residue((" ", number, " "), name, " ")
    residue.add(
        Atom.Atom(
            "CA",
            np.asarray(coordinate, dtype=float),
            0.0,
            1.0,
            " ",
            " CA ",
            number,
            element="C",
        )
    )
    return residue


def _structure(chain_coordinates):
    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, coordinates in chain_coordinates.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for number, coordinate in enumerate(coordinates, start=1):
            chain.add(_residue(number, coordinate))
    return structure


class StructureRefactorTests(unittest.TestCase):
    def test_legacy_facade_reexports_new_implementation(self):
        self.assertIs(pdbtools.align_homologs, align_homologs)
        self.assertIs(pdbtools.rename_chain, rename_chain)

    def test_chain_swap_is_simultaneous(self):
        structure = _structure({"A": [(0, 0, 0)], "B": [(1, 0, 0)]})
        chain_a, chain_b = list(structure[0])

        rename_chain(structure, {"A": "B", "B": "A"})

        self.assertIs(structure[0]["B"], chain_a)
        self.assertIs(structure[0]["A"], chain_b)

    def test_reset_index_uses_collision_free_intermediate_ids(self):
        structure = _structure({"A": [(0, 0, 0), (1, 0, 0)]})
        chain = structure[0]["A"]
        residues = list(chain)
        residues[0].id = (" ", 0, " ")

        reset_index(structure)

        self.assertEqual([residue.id[1] for residue in chain], [1, 2])
        self.assertIs(chain[(" ", 1, " ")], residues[0])
        self.assertIs(chain[(" ", 2, " ")], residues[1])

    def test_homolog_alignment_returns_complete_target_structure(self):
        reference = _structure(
            {"A": [(0, 0, 0), (1, 0, 0), (0, 1, 0)]}
        )
        target = _structure(
            {
                "B": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                "C": [(20, 0, 0)],
            }
        )

        aligned = align_homologs(reference, target, "A", "B")

        self.assertIsNot(aligned, target)
        self.assertEqual({chain.id for chain in aligned[0]}, {"B", "C"})
        self.assertEqual({chain.id for chain in target[0]}, {"B", "C"})


if __name__ == "__main__":
    unittest.main()
