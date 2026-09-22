"""OpenMM force-field charges for molecular-surface Coulomb potentials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .contacts.models import AtomReference
from .molecular_surface import MolecularSurfaceResult, map_electrostatic_potential

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure


def map_electrostatic_potential_openmm(
    structure: Structure,
    surface_obj: MolecularSurfaceResult,
    *,
    forcefield_files: Sequence[str] = ("amber14-all.xml",),
    add_hydrogens: bool = True,
    ph: float = 7.0,
    dielectric: float = 1.0,
    cutoff: float | None = None,
    field_name: str = "electrostatic_potential",
    chunk_size: int = 10_000,
) -> MolecularSurfaceResult:
    """Parameterize the surface's residues with OpenMM, then map their potential.

    The force field supplies *per-atom partial charges*. Missing hydrogens are
    added to an in-memory OpenMM model by default and contribute to the
    potential even when absent from the MSMS mesh. No atoms or coordinates in
    the input Biopython structure are changed. The function rejects atom or
    coordinate mismatches and force-field parameterization failures.

    This maps direct Coulomb potential, not OpenMM interaction energy or a
    Poisson-Boltzmann/solvent-screened potential. ``dielectric`` is a simple
    constant dielectric factor, not an implicit-solvent model.
    """
    if not isinstance(surface_obj, MolecularSurfaceResult):
        raise TypeError("surface_obj must be a MolecularSurfaceResult")
    if isinstance(forcefield_files, str) or not forcefield_files or not all(
        isinstance(path, str) and path.strip() for path in forcefield_files
    ):
        raise ValueError("forcefield_files must contain one or more file names")
    if not np.isfinite(ph) or not 0.0 <= ph <= 14.0:
        raise ValueError("ph must be between 0 and 14")

    try:
        from openmm import NonbondedForce, Vec3, unit
        from openmm.app import ForceField, Modeller, NoCutoff, Topology, element
    except ModuleNotFoundError as exc:
        if exc.name == "openmm" or (exc.name or "").startswith("openmm."):
            raise ModuleNotFoundError(
                "OpenMM electrostatic mapping requires the optional contacts "
                "dependencies: pip install 'biotools[contacts]'"
            ) from exc
        raise

    matching_models = [
        model for model in structure if str(model.id) == surface_obj.parameters.model_id
    ]
    if len(matching_models) != 1:
        raise ValueError(
            "The structure must contain exactly one model with ID "
            f"{surface_obj.parameters.model_id!r}"
        )
    model = matching_models[0]
    surface_atoms = {atom.reference: atom for atom in surface_obj.atoms}
    if len(surface_atoms) != len(surface_obj.atoms):
        raise ValueError("The surface contains duplicate atom references")
    included_residues = {
        (
            reference.chain_id,
            reference.hetero_flag,
            reference.residue_number,
            reference.insertion_code,
            reference.residue_name,
        )
        for reference in surface_atoms
    }

    topology = Topology()
    positions = []
    matched: set[AtomReference] = set()
    for source_chain in model:
        openmm_chain = None
        for source_residue in source_chain:
            hetero_flag, residue_number, insertion_code = source_residue.id
            residue_key = (
                str(source_chain.id),
                str(hetero_flag).strip(),
                int(residue_number),
                str(insertion_code).strip(),
                str(source_residue.get_resname()).strip(),
            )
            if residue_key not in included_residues:
                continue
            if openmm_chain is None:
                openmm_chain = topology.addChain(str(source_chain.id))
            openmm_residue = topology.addResidue(
                residue_key[4],
                openmm_chain,
                id=str(residue_number),
                insertionCode=residue_key[3],
            )
            for source_atom in source_residue.get_atoms():
                reference = AtomReference.from_atom(source_atom, str(structure.id))
                coordinates = np.asarray(source_atom.coord, dtype=float)
                symbol = str(getattr(source_atom, "element", "") or "").strip()
                if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
                    raise ValueError(f"Nonfinite coordinates for {reference!r}")
                if reference in surface_atoms:
                    surface_atom = surface_atoms[reference]
                    if symbol.upper() != surface_atom.element:
                        raise ValueError(
                            f"Surface and structure elements differ for {reference!r}"
                        )
                    if not np.allclose(
                        coordinates, surface_atom.coordinates, rtol=0.0, atol=1e-3
                    ):
                        raise ValueError(
                            f"Surface and structure coordinates differ for {reference!r}"
                        )
                    matched.add(reference)
                try:
                    atom_element = element.get_by_symbol(symbol.capitalize())
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"OpenMM cannot identify element {symbol!r} for {reference!r}"
                    ) from exc
                topology.addAtom(
                    str(source_atom.get_name()).strip(),
                    atom_element,
                    openmm_residue,
                )
                positions.append(Vec3(*map(float, coordinates)))

    if matched != set(surface_atoms):
        missing = next(iter(set(surface_atoms) - matched))
        raise ValueError(
            f"Surface atom {missing!r} was not found in the supplied structure"
        )

    position_quantity = unit.Quantity(positions, unit.angstrom)
    topology.createStandardBonds()
    topology.createDisulfideBonds(position_quantity)

    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(topology, position_quantity)
    try:
        if add_hydrogens:
            modeller.addHydrogens(forcefield, pH=float(ph))
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=None,
            rigidWater=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            "OpenMM could not parameterize the selected structure. Ensure "
            "standard residue names, complete heavy atoms, compatible "
            "ligands/waters, and suitable forcefield_files."
        ) from exc

    nonbonded_forces = [
        force for force in system.getForces() if isinstance(force, NonbondedForce)
    ]
    if len(nonbonded_forces) != 1:
        raise ValueError("Expected exactly one OpenMM NonbondedForce")
    nonbonded = nonbonded_forces[0]
    if nonbonded.getNumParticles() != len(list(modeller.topology.atoms())):
        raise ValueError("OpenMM particle count does not match the topology")
    charges = np.asarray(
        [
            nonbonded.getParticleParameters(index)[0].value_in_unit(
                unit.elementary_charge
            )
            for index in range(nonbonded.getNumParticles())
        ],
        dtype=float,
    )
    charge_positions = np.asarray(
        modeller.positions.value_in_unit(unit.angstrom), dtype=float
    )
    if charge_positions.shape != (len(charges), 3):
        raise ValueError("OpenMM charge positions do not match particle charges")

    return map_electrostatic_potential(
        surface_obj,
        charges,
        charge_positions=charge_positions,
        dielectric=dielectric,
        cutoff=cutoff,
        field_name=field_name,
        chunk_size=chunk_size,
        charge_source=(
            f"openmm({','.join(forcefield_files)}, "
            f"add_hydrogens={add_hydrogens}, pH={float(ph):g})"
        ),
    )


__all__ = ["map_electrostatic_potential_openmm"]
