"""Tests for FreeSASA and interaction-surface analysis."""

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from biotools import pdbtools
from biotools.structure import (
    analyze_interaction_surface,
    calculate_sasa,
    SASAResidue,
    SASAResult,
)


def _residue(
    number: int,
    name: str,
    coordinate: Sequence[float],
    *,
    hetero_flag: str = " ",
) -> Residue.Residue:
    residue = Residue.Residue((hetero_flag, number, " "), name, " ")
    atom_name = "O" if name == "HOH" else "CA"
    element = "O" if name == "HOH" else "C"
    residue.add(
        Atom.Atom(
            atom_name,
            np.asarray(coordinate, dtype=float),
            0.0,
            1.0,
            " ",
            f"{atom_name:>4}",
            number,
            element=element,
        )
    )
    return residue


def _structure() -> Structure.Structure:
    structure = Structure.Structure("surface")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues in {
        "A": [
            _residue(1, "ALA", (0.0, 0.0, 0.0)),
            _residue(2, "HOH", (1.0, 0.0, 0.0), hetero_flag="W"),
        ],
        "B": [_residue(1, "GLY", (3.0, 0.0, 0.0))],
        "C": [_residue(1, "SER", (20.0, 0.0, 0.0))],
        "W": [_residue(1, "HOH", (2.0, 0.0, 0.0), hetero_flag="W")],
    }.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue in residues:
            chain.add(residue)
    return structure


def _freesasa_json() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "structures": [
                        {
                            "area": {"total": 150.0},
                            "chains": [
                                {
                                    "label": "A",
                                    "area": {"total": 150.0},
                                    "residues": [
                                        {
                                            "name": "ALA",
                                            "number": "10",
                                            "area": {"total": 100.0},
                                            "relative-area": {"total": 104.05},
                                        },
                                        {
                                            "name": "GLY",
                                            "number": "11A",
                                            "area": {"total": 50.0},
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    )


def _completed(
    command: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_calculate_sasa_calls_freesasa_and_parses_residue_areas() -> None:
    structure = Structure.Structure("multi-character-chain")
    model = Model.Model(0)
    chain = Chain.Chain("peptide")
    chain.add(_residue(10, "ALA", (0.0, 0.0, 0.0)))
    model.add(chain)
    structure.add(model)
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        input_path = Path(command[-1])
        observed["input_exists"] = input_path.is_file()
        observed["pdb"] = input_path.read_text()
        return _completed(command, stdout=_freesasa_json())

    with patch(
        "biotools.structure.surface.subprocess.run",
        side_effect=run,
    ):
        result = calculate_sasa(structure, executable="custom-freesasa")

    assert observed["command"][:3] == [
        "custom-freesasa",
        "--format=json",
        "--output-depth=residue",
    ]
    assert observed["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }
    assert observed["input_exists"]
    assert " A  10" in observed["pdb"]
    assert chain.id == "peptide"
    assert result.total_absolute_sasa == 150.0
    assert result.chain_absolute_sasa == {"peptide": 150.0}
    assert result.residues[0] == SASAResidue(
        chain_id="peptide",
        residue_id=(" ", 10, " "),
        residue_name="ALA",
        absolute_sasa=100.0,
        relative_sasa=1.0405,
    )
    assert result.residues[1].residue_id == (" ", 11, "A")
    assert result.residues[1].relative_sasa is None


def test_calculate_sasa_reports_missing_executable() -> None:
    with patch(
        "biotools.structure.surface.subprocess.run",
        side_effect=FileNotFoundError("freesasa"),
    ), pytest.raises(FileNotFoundError, match="FreeSASA is not installed"):
        calculate_sasa(_structure())


def _sasa_result(
    chain_areas: dict[str, float],
    residue_areas: dict[str, float],
) -> SASAResult:
    residues = tuple(
        SASAResidue(
            chain_id=chain_id,
            residue_id=(" ", 1, " "),
            residue_name="ALA" if chain_id == "A" else "GLY",
            absolute_sasa=area,
            relative_sasa=None,
        )
        for chain_id, area in residue_areas.items()
    )
    return SASAResult(
        total_absolute_sasa=sum(chain_areas.values()),
        chain_absolute_sasa=chain_areas,
        residues=residues,
    )


def _mock_sasa(structures_seen):
    separated = {
        "A": _sasa_result({"A": 100.0}, {"A": 50.0}),
        "B": _sasa_result({"B": 100.0}, {"B": 60.0}),
    }
    complex_result = _sasa_result(
        {"A": 60.0, "B": 80.0},
        {"A": 30.0, "B": 50.0},
    )

    def calculate(structure, *, executable):
        chain_ids = tuple(chain.id for chain in structure[0])
        residue_names = tuple(
            residue.get_resname() for residue in structure.get_residues()
        )
        structures_seen.append((chain_ids, residue_names, executable))
        if len(chain_ids) == 1:
            return separated[chain_ids[0]]
        return complex_result

    return calculate


def test_interaction_surface_reports_residue_chain_and_total_scores() -> None:
    structure = _structure()
    structures_seen = []
    with patch(
        "biotools.structure.surface.calculate_sasa",
        side_effect=_mock_sasa(structures_seen),
    ):
        result = analyze_interaction_surface(
            structure,
            "A",
            "B",
            executable="custom-freesasa",
        )

    assert structures_seen == [
        (("A",), ("ALA",), "custom-freesasa"),
        (("B",), ("GLY",), "custom-freesasa"),
        (("A", "B"), ("ALA", "GLY"), "custom-freesasa"),
    ]
    assert {chain.id for chain in structure[0]} == {"A", "B", "C", "W"}
    assert any(
        residue.get_resname() == "HOH" for residue in structure.get_residues()
    )

    assert result.chain_ids == ("A", "B")
    assert result.chain_scores["A"].sasa_separated == 100.0
    assert result.chain_scores["A"].sasa_complex == 60.0
    assert result.chain_scores["A"].delta_sasa_absolute == 40.0
    assert result.chain_scores["A"].delta_sasa_relative == pytest.approx(0.4)
    assert result.chain_scores["B"].delta_sasa_relative == pytest.approx(0.2)
    assert result.total.sasa_separated == 200.0
    assert result.total.sasa_complex == 140.0
    assert result.total.delta_sasa_absolute == 60.0
    assert result.total.delta_sasa_relative == pytest.approx(0.3)
    assert result.per_residue_scores is not None
    assert result.per_residue_scores[0].chain_id == "A"
    assert result.per_residue_scores[0].delta_sasa_absolute == 20.0
    assert result.per_residue_scores[0].delta_sasa_relative == pytest.approx(0.4)
    assert result.per_residue_scores[1].chain_id == "B"
    assert result.per_residue_scores[1].delta_sasa_absolute == 10.0


def test_interaction_surface_flags_omit_disabled_outputs() -> None:
    structures_seen = []
    with patch(
        "biotools.structure.surface.calculate_sasa",
        side_effect=_mock_sasa(structures_seen),
    ):
        result = analyze_interaction_surface(
            _structure(),
            "A",
            "B",
            per_residue_scores=False,
            relative_sasa=False,
        )

    assert result.per_residue_scores is None
    assert result.total.delta_sasa_absolute == 60.0
    assert result.total.delta_sasa_relative is None

    structures_seen = []
    with patch(
        "biotools.structure.surface.calculate_sasa",
        side_effect=_mock_sasa(structures_seen),
    ):
        relative_only = analyze_interaction_surface(
            _structure(),
            "A",
            "B",
            absolute_sasa=False,
        )

    assert relative_only.total.sasa_separated is None
    assert relative_only.total.sasa_complex is None
    assert relative_only.total.delta_sasa_absolute is None
    assert relative_only.total.delta_sasa_relative == pytest.approx(0.3)


def test_interaction_surface_validates_chain_and_output_selection() -> None:
    structure = _structure()
    with pytest.raises(ValueError, match="different chains"):
        analyze_interaction_surface(structure, "A", "A")
    with pytest.raises(ValueError, match="At least one"):
        analyze_interaction_surface(
            structure,
            "A",
            "B",
            relative_sasa=False,
            absolute_sasa=False,
        )
    with pytest.raises(ValueError, match="not found"):
        analyze_interaction_surface(structure, "A", "missing")


def test_surface_api_is_available_from_legacy_facade() -> None:
    assert pdbtools.calculate_sasa is calculate_sasa
    assert pdbtools.analyze_interaction_surface is analyze_interaction_surface
