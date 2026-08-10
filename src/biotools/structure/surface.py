"""Solvent-accessible and interaction-surface analysis with FreeSASA."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from .chains import (
    _get_chain,
    extract_chain,
    remove_wather_molecules,
    rename_chain,
)
from .io import (
    _resolve_pdb_chain_map,
    save_structure_to_file,
)

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

ResidueID = tuple[str, int, str]


@dataclass(frozen=True)
class SASAResidue:
    """Absolute and relative solvent accessibility of one residue."""

    chain_id: str
    residue_id: ResidueID
    residue_name: str
    absolute_sasa: float
    relative_sasa: float | None


@dataclass(frozen=True)
class SASAResult:
    """FreeSASA areas for a structure, its chains, and its residues."""

    total_absolute_sasa: float
    chain_absolute_sasa: dict[str, float]
    residues: tuple[SASAResidue, ...]


@dataclass(frozen=True)
class InteractionSurfaceScore:
    """SASA change between a separated component and the complex."""

    sasa_separated: float | None
    sasa_complex: float | None
    delta_sasa_absolute: float | None
    delta_sasa_relative: float | None


@dataclass(frozen=True)
class ResidueInteractionSurfaceScore(InteractionSurfaceScore):
    """Interaction-surface score associated with one residue."""

    chain_id: str
    residue_id: ResidueID
    residue_name: str


@dataclass(frozen=True)
class InteractionSurfaceResult:
    """Per-residue, per-chain, and total buried surface of two chains."""

    chain_ids: tuple[str, str]
    per_residue_scores: tuple[ResidueInteractionSurfaceScore, ...] | None
    chain_scores: dict[str, InteractionSurfaceScore]
    total: InteractionSurfaceScore


def _parse_residue_id(value: str) -> ResidueID:
    match = re.fullmatch(r"\s*(-?\d+)\s*([A-Za-z]?)\s*", value)
    if match is None:
        raise ValueError(f"Unsupported FreeSASA residue number: {value!r}")
    insertion_code = match.group(2) or " "
    return (" ", int(match.group(1)), insertion_code)


def _parse_freesasa_output(
    output: str,
    reverse_chain_map: dict[str, str],
) -> SASAResult:
    try:
        document = json.loads(output)
        result_document = document["results"][0]
        structure_document = result_document["structures"][0]
        total_absolute_sasa = float(structure_document["area"]["total"])
        chain_documents = structure_document["chains"]
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Could not parse FreeSASA JSON output") from exc

    chain_absolute_sasa = {}
    residues = []
    try:
        for chain_document in chain_documents:
            pdb_chain_id = str(chain_document["label"])
            chain_id = reverse_chain_map.get(pdb_chain_id, pdb_chain_id)
            chain_absolute_sasa[chain_id] = float(
                chain_document["area"]["total"]
            )
            for residue_document in chain_document.get("residues", []):
                relative_document = residue_document.get("relative-area")
                relative_sasa = (
                    None
                    if relative_document is None
                    or relative_document.get("total") is None
                    else float(relative_document["total"]) / 100.0
                )
                residues.append(
                    SASAResidue(
                        chain_id=chain_id,
                        residue_id=_parse_residue_id(
                            str(residue_document["number"])
                        ),
                        residue_name=str(residue_document["name"]),
                        absolute_sasa=float(
                            residue_document["area"]["total"]
                        ),
                        relative_sasa=relative_sasa,
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Could not parse FreeSASA chain or residue output") from exc

    return SASAResult(
        total_absolute_sasa=total_absolute_sasa,
        chain_absolute_sasa=chain_absolute_sasa,
        residues=tuple(residues),
    )


def _run_freesasa(input_path: Path, executable: str) -> str:
    command = [
        executable,
        "--format=json",
        "--output-depth=residue",
        str(input_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "FreeSASA is not installed or its 'freesasa' executable is not "
            "available on PATH"
        ) from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = (completed.stderr or completed.stdout).strip()
        raise OSError(
            f"FreeSASA failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stdout


def calculate_sasa(
    structure: Structure,
    *,
    executable: str = "freesasa",
) -> SASAResult:
    """Calculate solvent-accessible surface area with FreeSASA.

    The function accepts a Biopython structure and calls the FreeSASA command
    line program directly. Multi-character chain IDs are mapped temporarily to
    PDB-compatible IDs and restored in the returned records. FreeSASA's default
    behavior is retained: only the first model is analyzed, and hydrogens and
    ``HETATM`` records are ignored.

    Relative residue SASA is returned as a fraction, where ``1.0`` corresponds
    to 100 percent of FreeSASA's reference accessibility. Values may exceed
    ``1.0`` for unusually exposed conformations.

    Args:
        structure: Biopython structure to analyze; it is not modified.
        executable: FreeSASA executable name or path.

    Returns:
        Total, per-chain, and per-residue SASA values.

    Raises:
        FileNotFoundError: If FreeSASA is unavailable.
        OSError: If FreeSASA rejects the generated PDB input.
        ValueError: If the executable name or FreeSASA output is invalid.
    """
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("executable must be a non-empty string")

    prepared_structure = structure.copy()
    chain_map = _resolve_pdb_chain_map(prepared_structure)
    rename_chain(prepared_structure, chain_map)
    reverse_chain_map = {
        pdb_chain_id: original_chain_id
        for original_chain_id, pdb_chain_id in chain_map.items()
    }

    with TemporaryDirectory(prefix="biotools-freesasa-") as temp_dir:
        input_path = Path(temp_dir) / "structure.pdb"
        save_structure_to_file(
            prepared_structure,
            str(input_path),
            verbose=False,
        )
        output = _run_freesasa(input_path, executable)
    return _parse_freesasa_output(output, reverse_chain_map)


def _surface_score(
    sasa_separated: float,
    sasa_complex: float,
    *,
    relative_sasa: bool,
    absolute_sasa: bool,
) -> InteractionSurfaceScore:
    delta_absolute = sasa_separated - sasa_complex
    if not relative_sasa:
        delta_relative = None
    elif sasa_separated == 0.0:
        delta_relative = 0.0 if delta_absolute == 0.0 else None
    else:
        delta_relative = delta_absolute / sasa_separated
    return InteractionSurfaceScore(
        sasa_separated=sasa_separated if absolute_sasa else None,
        sasa_complex=sasa_complex if absolute_sasa else None,
        delta_sasa_absolute=delta_absolute if absolute_sasa else None,
        delta_sasa_relative=delta_relative,
    )


def _residue_map(result: SASAResult) -> dict[tuple[str, ResidueID], SASAResidue]:
    return {
        (residue.chain_id, residue.residue_id): residue
        for residue in result.residues
    }


def analyze_interaction_surface(
    structure: Structure,
    chain_a: str,
    chain_b: str,
    *,
    per_residue_scores: bool = True,
    relative_sasa: bool = True,
    absolute_sasa: bool = True,
    executable: str = "freesasa",
) -> InteractionSurfaceResult:
    """Calculate the surface buried by association of two protein chains.

    SASA is calculated for each chain in isolation and for the two-chain
    complex. ``delta_sasa_absolute`` is ``separated - complex`` in square
    angstroms. ``delta_sasa_relative`` is that loss divided by separated SASA,
    so ``1.0`` denotes complete burial. The total absolute delta is the
    two-sided buried surface; the commonly reported interface area is half of
    this value.

    Only the selected chains are retained. Water and all other hetero residues
    are removed from the copied structures before calculation, leaving the
    input structure unchanged.

    Args:
        structure: Biopython structure containing both chains.
        chain_a: First interacting chain ID.
        chain_b: Second interacting chain ID.
        per_residue_scores: Include residue-level surface changes.
        relative_sasa: Include relative SASA changes.
        absolute_sasa: Include absolute SASA values and changes.
        executable: FreeSASA executable name or path.

    Returns:
        Per-residue (optionally), per-chain, and total surface changes.

    Raises:
        ValueError: If chain IDs or output selections are invalid, or if
            FreeSASA returns inconsistent residue sets.
        FileNotFoundError: If FreeSASA is unavailable.
        OSError: If FreeSASA cannot process a generated structure.
    """
    if chain_a == chain_b:
        raise ValueError("chain_a and chain_b must identify different chains")
    if not relative_sasa and not absolute_sasa:
        raise ValueError(
            "At least one of relative_sasa and absolute_sasa must be True"
        )
    _get_chain(structure, chain_a)
    _get_chain(structure, chain_b)

    complex_structure = extract_chain(
        structure,
        [chain_a, chain_b],
        verbose=False,
    )
    remove_wather_molecules(complex_structure)
    separated_structures = {
        chain_id: extract_chain(
            complex_structure,
            chain_id,
            verbose=False,
        )
        for chain_id in (chain_a, chain_b)
    }

    separated_results = {
        chain_id: calculate_sasa(chain_structure, executable=executable)
        for chain_id, chain_structure in separated_structures.items()
    }
    complex_result = calculate_sasa(
        complex_structure,
        executable=executable,
    )

    chain_scores = {}
    for chain_id in (chain_a, chain_b):
        try:
            separated_area = separated_results[
                chain_id
            ].chain_absolute_sasa[chain_id]
            complex_area = complex_result.chain_absolute_sasa[chain_id]
        except KeyError as exc:
            raise ValueError(
                f"FreeSASA output is missing chain {chain_id!r}"
            ) from exc
        chain_scores[chain_id] = _surface_score(
            separated_area,
            complex_area,
            relative_sasa=relative_sasa,
            absolute_sasa=absolute_sasa,
        )

    residue_scores = None
    if per_residue_scores:
        complex_residues = _residue_map(complex_result)
        scores = []
        for chain_id in (chain_a, chain_b):
            for separated_residue in separated_results[chain_id].residues:
                key = (chain_id, separated_residue.residue_id)
                try:
                    complex_residue = complex_residues.pop(key)
                except KeyError as exc:
                    raise ValueError(
                        "FreeSASA complex output is missing residue "
                        f"{chain_id}:{separated_residue.residue_id}"
                    ) from exc
                if separated_residue.residue_name != complex_residue.residue_name:
                    raise ValueError(
                        "FreeSASA residue identity differs between separated "
                        f"and complex calculations for {key!r}"
                    )
                score = _surface_score(
                    separated_residue.absolute_sasa,
                    complex_residue.absolute_sasa,
                    relative_sasa=relative_sasa,
                    absolute_sasa=absolute_sasa,
                )
                scores.append(
                    ResidueInteractionSurfaceScore(
                        chain_id=chain_id,
                        residue_id=separated_residue.residue_id,
                        residue_name=separated_residue.residue_name,
                        sasa_separated=score.sasa_separated,
                        sasa_complex=score.sasa_complex,
                        delta_sasa_absolute=score.delta_sasa_absolute,
                        delta_sasa_relative=score.delta_sasa_relative,
                    )
                )
        if complex_residues:
            extra_residues = ", ".join(
                f"{chain_id}:{residue_id}"
                for chain_id, residue_id in complex_residues
            )
            raise ValueError(
                "FreeSASA complex output contains unmatched residues: "
                f"{extra_residues}"
            )
        residue_scores = tuple(scores)

    total_separated = sum(
        result.total_absolute_sasa for result in separated_results.values()
    )
    total_score = _surface_score(
        total_separated,
        complex_result.total_absolute_sasa,
        relative_sasa=relative_sasa,
        absolute_sasa=absolute_sasa,
    )
    return InteractionSurfaceResult(
        chain_ids=(chain_a, chain_b),
        per_residue_scores=residue_scores,
        chain_scores=chain_scores,
        total=total_score,
    )
