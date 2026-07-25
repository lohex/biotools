"""Regression tests for the split structure API and compatibility facades."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools import pdbtools
from biotools.structure import io as structure_io
from biotools.structure import (
    align_homologs,
    clip_chain,
    get_interaction_residues,
    get_interaction_residues_full,
    rename_chain,
    reset_index,
)


def _residue(
    number: int,
    coordinate: Sequence[float],
    name: str = "ALA",
) -> Residue.Residue:
    """Create a minimal amino-acid residue containing one C-alpha atom."""
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


def _structure(
    chain_coordinates: Mapping[str, Sequence[Sequence[float]]],
) -> Structure.Structure:
    """Create a minimal single-model structure from per-chain coordinates."""
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
    """Verify refactored structure helpers and their legacy facade."""

    def test_legacy_facade_reexports_new_implementation(self) -> None:
        """The legacy module should expose the new function objects."""
        self.assertIs(pdbtools.align_homologs, align_homologs)
        self.assertIs(pdbtools.rename_chain, rename_chain)

    def test_chain_swap_is_simultaneous(self) -> None:
        """Chain swaps should not collide in Biopython's child index."""
        structure = _structure({"A": [(0, 0, 0)], "B": [(1, 0, 0)]})
        chain_a, chain_b = list(structure[0])

        rename_chain(structure, {"A": "B", "B": "A"})

        self.assertIs(structure[0]["B"], chain_a)
        self.assertIs(structure[0]["A"], chain_b)

    def test_reset_index_uses_collision_free_intermediate_ids(self) -> None:
        """Residue renumbering should safely handle an existing target ID."""
        structure = _structure({"A": [(0, 0, 0), (1, 0, 0)]})
        chain = structure[0]["A"]
        residues = list(chain)
        residues[0].id = (" ", 0, " ")

        reset_index(structure)

        self.assertEqual([residue.id[1] for residue in chain], [1, 2])
        self.assertIs(chain[(" ", 1, " ")], residues[0])
        self.assertIs(chain[(" ", 2, " ")], residues[1])

    def test_clip_chain_accepts_amino_acids_without_ca_atoms(self) -> None:
        """Sequence-based clipping should not require C-alpha atoms."""
        structure = _structure({"A": [(0, 0, 0), (1, 0, 0)]})
        chain = structure[0]["A"]
        incomplete_residue = Residue.Residue((" ", 3, " "), "ALA", " ")
        chain.add(incomplete_residue)

        result = clip_chain(structure, {"A": "AAA"})

        self.assertIs(result, structure)
        self.assertIs(chain[incomplete_residue.id], incomplete_residue)
        self.assertEqual(len(list(chain)), 3)

    def test_homolog_alignment_returns_complete_target_structure(self) -> None:
        """Homolog alignment should retain every chain in the target."""
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

    def test_homolog_alignment_ignores_residues_without_ca_atoms(self) -> None:
        """Incomplete residues should not invalidate homolog superposition."""
        reference = _structure(
            {"A": [(0, 0, 0), (1, 0, 0), (0, 1, 0)]}
        )
        target = _structure(
            {"B": [(10, 0, 0), (11, 0, 0), (10, 1, 0)]}
        )
        incomplete_residue = Residue.Residue((" ", 4, " "), "ALA", " ")
        target[0]["B"].add(incomplete_residue)

        aligned = align_homologs(reference, target, "A", "B")

        self.assertEqual(
            aligned[0]["B"][incomplete_residue.id].get_resname(),
            incomplete_residue.get_resname(),
        )

    def test_neighbor_search_matches_full_interaction_search(self) -> None:
        """KD-tree and brute-force contact searches should agree."""
        structure = _structure(
            {
                "A": [(0, 0, 0), (10, 0, 0)],
                "B": [(3, 0, 0), (20, 0, 0)],
            }
        )

        neighbor_result = get_interaction_residues(
            structure,
            "A",
            "B",
            cutoff=5.0,
        )
        full_result = get_interaction_residues_full(
            structure,
            "A",
            "B",
            cutoff=5.0,
        )

        self.assertEqual(neighbor_result, full_result)

    @patch("biotools.structure.io.get_pdb_structure_as_mmcif")
    @patch("biotools.structure.io.get_pdb_structure_as_pdb")
    def test_pdb_loader_uses_mmcif_as_default_fallback(
        self,
        load_pdb: MagicMock,
        load_mmcif: MagicMock,
    ) -> None:
        """The default loader should fall back from PDB to mmCIF."""
        expected = object()
        load_pdb.side_effect = OSError("PDB unavailable")
        load_mmcif.return_value = expected

        result = structure_io.get_pdb_structure("1ABC", target_folder="data")

        self.assertIs(result, expected)
        load_pdb.assert_called_once_with("1abc", "data", verbose=True)
        load_mmcif.assert_called_once_with("1abc", "data", verbose=True)

    @patch("biotools.structure.io.MMCIFParser.get_structure")
    @patch("Bio.PDB.PDBList.PDBList.retrieve_pdb_file")
    def test_pdb_loader_falls_back_when_download_returns_none(
        self,
        retrieve_file: MagicMock,
        parse_mmcif: MagicMock,
    ) -> None:
        """A missing legacy download should trigger the mmCIF fallback."""
        expected = object()
        retrieve_file.side_effect = [None, "downloaded.cif"]
        parse_mmcif.return_value = expected

        result = structure_io.get_pdb_structure("9XYZ", target_folder="data")

        self.assertIs(result, expected)
        self.assertEqual(retrieve_file.call_count, 2)
        self.assertEqual(
            retrieve_file.call_args_list[0].kwargs["file_format"],
            "pdb",
        )
        self.assertEqual(
            retrieve_file.call_args_list[1].kwargs["file_format"],
            "mmCif",
        )
        parse_mmcif.assert_called_once_with("9xyz", "downloaded.cif")

    @patch("biotools.structure.io.get_pdb_structure_as_mmcif")
    @patch("biotools.structure.io.get_pdb_structure_as_pdb")
    def test_pdb_loader_can_prefer_mmcif(
        self,
        load_pdb: MagicMock,
        load_mmcif: MagicMock,
    ) -> None:
        """The optional flag should attempt mmCIF before PDB."""
        expected = object()
        load_mmcif.return_value = expected

        result = structure_io.get_pdb_structure("1ABC", prefer_mmcif=True)

        self.assertIs(result, expected)
        load_mmcif.assert_called_once_with("1abc", ".", verbose=True)
        load_pdb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
