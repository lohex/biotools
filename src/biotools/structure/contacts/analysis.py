"""Preparation and typed analysis facade."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .. import _contacts
from .config import ContactConfig, get_contact_config
from .models import (
    AtomReference,
    ContactAnalysisResult,
    ContactDiagnostic,
    ContactObservation,
    ContactSystem,
    CriterionResult,
    GeometryMeasurement,
)


def _structure_hash(atoms: list[Any], references: dict[Any, AtomReference]) -> str:
    digest = hashlib.sha256()
    for atom in atoms:
        digest.update(repr(references[atom]).encode("utf-8"))
        digest.update(atom.get_coord().astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def prepare_contact_system(
    structure: Any,
    partners: tuple[str, str],
    *,
    topology_backend: str = "templates",
) -> ContactSystem:
    """Validate two partners and prepare their shared chemistry/topology.

    The input structure is never modified.  Explicit alternate conformers are
    not combined: Biopython's currently selected atom for a disordered site is
    used and a coverage diagnostic records that choice.
    """
    chain_a, chain_b = partners
    model, object_a, object_b = _contacts._select_model_and_chains(
        structure, chain_a, chain_b
    )
    residues_a = tuple(
        residue for residue in object_a if _contacts._is_protein_residue(residue)
    )
    residues_b = tuple(
        residue for residue in object_b if _contacts._is_protein_residue(residue)
    )
    waters = tuple(
        residue
        for chain in model
        for residue in chain
        if _contacts._residue_name(residue) in _contacts.WATER_RESIDUE_NAMES
    )
    topology_residues = list(dict.fromkeys(residues_a + residues_b + waters))
    adjacency_lists = _contacts._build_bond_adjacency(
        model, topology_residues, topology_backend
    )
    adjacency = {atom: tuple(neighbors) for atom, neighbors in adjacency_lists.items()}
    atoms = [atom for residue in topology_residues for atom in residue.get_atoms()]
    structure_id = str(getattr(structure, "id", "") or "")
    references = {atom: AtomReference.from_atom(atom, structure_id) for atom in atoms}

    diagnostics: list[ContactDiagnostic] = []
    unsupported = [
        residue
        for chain in (object_a, object_b)
        for residue in chain
        if not _contacts._is_protein_residue(residue)
        and _contacts._residue_name(residue) not in _contacts.WATER_RESIDUE_NAMES
    ]
    if unsupported:
        diagnostics.append(
            ContactDiagnostic(
                "unsupported_residues",
                f"{len(unsupported)} nonstandard residues were excluded from chemical typing",
            )
        )
    disordered = [atom for atom in atoms if atom.is_disordered()]
    if disordered:
        diagnostics.append(
            ContactDiagnostic(
                "selected_altlocs",
                f"Biopython's selected conformer was used for {len(disordered)} disordered atoms",
            )
        )
    heavy_atoms = [atom for atom in atoms if _contacts._element(atom) != "H"]
    hydrogens = [atom for atom in atoms if _contacts._element(atom) == "H"]
    if not hydrogens:
        diagnostics.append(
            ContactDiagnostic(
                "missing_hydrogens",
                "No explicit hydrogen atoms are present; directional hydrogen "
                "bonds may be underdetected",
            )
        )
    coverage = {
        "protein_residues": len(residues_a) + len(residues_b),
        "water_sites": len(waters),
        "atoms": len(atoms),
        "heavy_atoms": len(heavy_atoms),
        "explicit_hydrogens": len(hydrogens),
        "unsupported_residues": len(unsupported),
    }
    return ContactSystem(
        structure=structure,
        model=model,
        chain_a=chain_a,
        chain_b=chain_b,
        residues_a=residues_a,
        residues_b=residues_b,
        water_residues=waters,
        adjacency=adjacency,
        atom_references=references,
        diagnostics=tuple(diagnostics),
        coverage=coverage,
        topology_backend=topology_backend,
        input_hash=_structure_hash(atoms, references),
    )


def _typed_observation(
    system: ContactSystem,
    raw: Any,
    config: ContactConfig,
) -> ContactObservation:
    def selected_atoms(residue: Any, label: str | None) -> tuple[Any, ...]:
        atoms = list(residue.get_atoms())
        if not label:
            return tuple(atoms)
        ring_table = (
            _contacts.AROMATIC_RINGS
            if config.refined_geometry
            else _contacts.LEGACY_AROMATIC_RINGS
        )
        for ring_id, names_in_ring in ring_table.get(
            _contacts._residue_name(residue), ()
        ):
            if label in {ring_id, "aromatic ring", "ring"}:
                names = set(names_in_ring)
                return tuple(
                    atom for atom in atoms if atom.get_name().strip() in names
                )
        names = set(label.split("/"))
        selected = tuple(atom for atom in atoms if atom.get_name().strip() in names)
        return selected or tuple(atoms)

    atoms_a = selected_atoms(raw.residue_a, raw.atom_a)
    atoms_b = selected_atoms(raw.residue_b, raw.atom_b)
    references_a = tuple(system.atom_references[atom] for atom in atoms_a)
    references_b = tuple(system.atom_references[atom] for atom in atoms_b)
    measurements = tuple(
        GeometryMeasurement(name, float(value), unit)
        for name, value, unit in raw.measurements
    )
    criteria = tuple(
        CriterionResult(name, bool(satisfied), limit)
        for name, satisfied, limit in raw.criteria
    )
    mediator = ()
    if raw.water is not None:
        oxygen = next(
            (atom for atom in raw.water.get_atoms() if _contacts._element(atom) == "O"),
            None,
        )
        if oxygen is not None:
            mediator = (system.atom_references[oxygen],)
    return ContactObservation(
        interaction_type=raw.interaction_type,
        partner_a=references_a,
        partner_b=references_b,
        role_a=raw.role_a,
        role_b=raw.role_b,
        geometry=measurements,
        criteria=criteria,
        rule_profile=config.name,
        evidence_level=raw.evidence_level,
        quality_flags=raw.quality_flags,
        mediator_waters=mediator,
    )


def analyze_contacts(
    prepared: ContactSystem,
    *,
    profile: str | ContactConfig = "refined",
    hydrogen_bond: bool = True,
    salt_bridge: bool = True,
    hydrophobic_contact: bool = True,
    van_der_waals_contact: bool = True,
    pi_stacking_parallel: bool = True,
    cation_pi_candidate: bool = True,
    water_bridge: bool = True,
    pi_stacking_t_shaped: bool = True,
) -> ContactAnalysisResult:
    """Run all selected detectors and retain complete typed observations."""
    config = get_contact_config(profile)
    raw = _contacts._detect_contact_observations(
        prepared.model,
        list(prepared.residues_a),
        list(prepared.residues_b),
        hydrogen_bond=hydrogen_bond,
        salt_bridge=salt_bridge,
        hydrophobic_contact=hydrophobic_contact,
        van_der_waals_contact=van_der_waals_contact,
        pi_stacking_parallel=pi_stacking_parallel,
        cation_pi_candidate=cation_pi_candidate,
        water_bridge=water_bridge,
        pi_stacking_t_shaped=pi_stacking_t_shaped,
        topology_backend=prepared.topology_backend,
        config=config,
        adjacency_override=prepared.adjacency,
    )
    try:
        biopython_version = version("biopython")
    except PackageNotFoundError:
        biopython_version = "unknown"
    observations = tuple(_typed_observation(prepared, item, config) for item in raw)
    coverage = dict(prepared.coverage)
    coverage["observations"] = len(observations)
    return ContactAnalysisResult(
        observations=observations,
        diagnostics=prepared.diagnostics,
        coverage=coverage,
        config_hash=config.config_hash,
        rule_profile=config.name,
        backend_versions={"biopython": biopython_version},
        input_hash=prepared.input_hash,
    )
