"""Chain selection, sequence extraction, and safe identifier manipulation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
import logging
from typing import TYPE_CHECKING

from Bio.PDB.Polypeptide import is_aa

from ..sequence.alignment import global_alignment_seqs
from .io import map_three_to_one

if TYPE_CHECKING:
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)


def _get_chain(structure: Structure, chain_id: str) -> Chain:
    """Return a chain object by ID.

    Args:
        structure: Structure to search across all models.
        chain_id: Identifier of the requested chain.

    Returns:
        First matching chain in structure iteration order.

    Raises:
        ValueError: If the structure contains no matching chain.
    """
    for chain in structure.get_chains():
        if chain.id == chain_id:
            return chain
    raise ValueError(f"Chain {chain_id!r} not found in structure")


def _get_protein_residues(chain: Chain) -> list[Residue]:
    """Return amino-acid residues from a chain.

    Args:
        chain: Biopython chain to inspect.

    Returns:
        Residues recognized as amino acids by Biopython.
    """
    return [residue for residue in chain.get_residues() if is_aa(residue)]


def _validate_chain_for_protein_alignment(chain: Chain) -> list[Residue]:
    """Validate a chain for protein index-based alignment logic.

    Args:
        chain: Chain expected to contain only protein residues with C-alpha
            atoms.

    Returns:
        Validated amino-acid residues in chain order.

    Raises:
        ValueError: If non-protein residues occur or the residue and C-alpha
            atom counts differ.
    """
    hetero_residues = [
        residue for residue in chain.get_residues() if not is_aa(residue)
    ]
    if hetero_residues:
        residue_labels = [
            f"{residue.get_resname()}:{residue.id[1]}"
            for residue in hetero_residues[:5]
        ]
        raise ValueError(
            f"Chain {chain.id!r} contains non-protein residues that would make "
            f"protein index mapping ambiguous: {', '.join(residue_labels)}"
        )

    protein_residues = _get_protein_residues(chain)
    ca_atoms = [atom for atom in chain.get_atoms() if atom.get_id() == "CA"]
    if len(protein_residues) != len(ca_atoms):
        raise ValueError(
            f"Chain {chain.id!r} has {len(protein_residues)} protein residues but "
            f"{len(ca_atoms)} CA atoms; residue-to-atom mapping is not clean."
        )
    return protein_residues


def extract_chain(
    structure: Structure,
    chain_id: str | Collection[str],
) -> Structure:
    """Return a copy of a structure containing only selected chains.

    Args:
        structure: Source structure, which remains unchanged.
        chain_id: One chain ID or a collection of IDs to retain.

    Returns:
        Structure copy containing only the requested chains.

    Raises:
        Exception: If none of the requested chains exist.
    """
    if isinstance(chain_id, str):
        chain_id = [chain_id]
    selected = structure.copy()
    chains_to_keep = []
    for model in selected:
        chains_to_remove = [chain for chain in model if chain.id not in chain_id]
        logger.debug(
            "Removing chains %s from model %s",
            [chain.id for chain in chains_to_remove],
            model.id,
        )
        for chain in chains_to_remove:
            model.detach_child(chain.id)
        chains_to_keep += [chain for chain in model if chain.id in chain_id]

    if not chains_to_keep:
        raise Exception(f"Chain {chain_id} not found in structure!")
    return selected


def remove_wather_molecules(structure: Structure) -> Structure:
    """Remove water molecules and all hetero residues in place.

    Args:
        structure: Structure to modify.

    Returns:
        The same modified structure object.
    """
    for model in structure:
        for chain in model:
            residues_to_remove = [
                residue
                for residue in chain
                if residue.get_resname() == "HOH"
                or residue.get_full_id()[-1][0] != " "
            ]
            for residue in residues_to_remove:
                chain.detach_child(residue.id)
    return structure


def get_aa_sequence(
    structure: Structure,
    show_gaps: bool = True,
) -> dict[str, str]:
    """Extract per-chain amino-acid sequences from a structure.

    Args:
        structure: Structure containing the chains to inspect.
        show_gaps: Insert ``"-"`` for missing integer residue numbers.

    Returns:
        Mapping from chain ID to one-letter amino-acid sequence.
    """
    chain_seqs: dict[str, str] = {}
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        last_res_id = None
        amino_acids = ""
        for residue in chain.get_residues():
            if not is_aa(residue):
                continue
            _, res_id, _ = residue.get_id()
            if show_gaps and last_res_id is not None and res_id > last_res_id + 1:
                amino_acids += "-" * (res_id - last_res_id - 1)
            amino_acids += map_three_to_one(residue.get_resname())
            last_res_id = res_id
        chain_seqs[chain_id] = amino_acids
    return chain_seqs


def clip_chain(
    structure: Structure,
    chain_seqs: Mapping[str, str],
    verbose: bool = False,
) -> Structure:
    """Trim residues that align to gaps in target chain sequences.

    The operation is purely sequence based. All residues recognized as amino
    acids participate in both the alignment and residue-index mapping,
    regardless of which atoms are present.

    Args:
        structure: Structure to modify in place.
        chain_seqs: Target amino-acid sequence for each chain to process.
        verbose: Log removed residue IDs at informational level.

    Returns:
        The same modified structure object.

    Raises:
        ValueError: If a selected chain has no amino-acid residues or cannot be
            mapped cleanly to its alignment.
    """
    for chain in structure.get_chains():
        if chain.id not in chain_seqs:
            continue

        protein_residues = [
            residue
            for residue in chain.get_residues()
            if is_aa(residue)
        ]
        if not protein_residues:
            raise ValueError(
                f"Chain {chain.id!r} has no amino-acid residues"
            )

        pdb_sequence = "".join(
            map_three_to_one(residue.get_resname())
            for residue in protein_residues
        )
        gapped_pdb_seq, gapped_seq = global_alignment_seqs(
            pdb_sequence,
            chain_seqs[chain.id],
            gap_penalty=(-10, -1),
        )
        if len(gapped_pdb_seq.replace("-", "")) != len(protein_residues):
            raise ValueError(
                f"Protein residue count mismatch while clipping chain {chain.id!r}: "
                f"{len(protein_residues)} residues vs alignment-derived "
                f"{len(gapped_pdb_seq.replace('-', ''))} positions."
            )

        remove = []
        residue_index = 0
        for pdb_residue, target_residue in zip(gapped_pdb_seq, gapped_seq):
            if pdb_residue == "-":
                continue
            residue = protein_residues[residue_index]
            residue_index += 1
            if target_residue == "-":
                remove.append(residue.id)

        if verbose:
            logger.info(
                "Removing %d residues from chain %s: %s",
                len(remove),
                chain.id,
                remove,
            )
        for residue_id in remove:
            chain.detach_child(residue_id)
    return structure


def reset_index(structure: Structure) -> Structure:
    """Safely renumber residues in every chain starting from one.

    Free negative IDs are used temporarily to avoid collisions in Biopython's
    child index. Hetero flags and insertion codes are preserved.

    Args:
        structure: Structure to renumber in place.

    Returns:
        The same renumbered structure object.
    """
    for model in structure:
        for chain in model:
            residues = list(chain)
            final_ids = [
                (hetero_flag, index, insertion_code)
                for index, (hetero_flag, _, insertion_code) in enumerate(
                    (residue.id for residue in residues),
                    start=1,
                )
            ]
            reserved_ids = {residue.id for residue in residues} | set(final_ids)
            temporary_number = -1
            for residue in residues:
                hetero_flag, _, insertion_code = residue.id
                temporary_id = (hetero_flag, temporary_number, insertion_code)
                while temporary_id in reserved_ids or chain.has_id(temporary_id):
                    temporary_number -= 1
                    temporary_id = (hetero_flag, temporary_number, insertion_code)
                reserved_ids.add(temporary_id)
                residue.id = temporary_id
                temporary_number -= 1

            for residue, final_id in zip(residues, final_ids):
                residue.id = final_id
    return structure


def rename_chain(
    structure: Structure,
    chain_ids: Mapping[str, str],
) -> Structure:
    """Rename chains simultaneously according to a mapping.

    Temporary IDs prevent collisions during swaps and chained renames.

    Args:
        structure: Structure whose chains are modified in place.
        chain_ids: Mapping from current chain IDs to desired IDs.

    Returns:
        The same structure with renamed chains.

    Raises:
        ValueError: If the mapping would create duplicate IDs within a model.
    """
    plans = []
    for model in structure:
        chains = list(model)
        final_ids = [chain_ids.get(chain.id, chain.id) for chain in chains]
        duplicate_ids = {
            chain_id for chain_id in final_ids if final_ids.count(chain_id) > 1
        }
        if duplicate_ids:
            raise ValueError(
                f"Chain rename would produce duplicate IDs in model {model.id!r}: "
                f"{sorted(duplicate_ids)!r}"
            )
        renames = [
            (chain, target_id)
            for chain, target_id in zip(chains, final_ids)
            if chain.id != target_id
        ]
        plans.append((model, renames, set(final_ids)))

    for model, renames, reserved_ids in plans:
        temporary_renames = []
        for index, (chain, target_id) in enumerate(renames):
            temporary_id = f"__biotools_tmp_{index}__"
            while temporary_id in reserved_ids or model.has_id(temporary_id):
                temporary_id += "_"
            reserved_ids.add(temporary_id)
            chain.id = temporary_id
            temporary_renames.append((chain, target_id))
        for chain, target_id in temporary_renames:
            chain.id = target_id
    return structure
