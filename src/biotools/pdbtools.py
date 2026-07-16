"""Utilities for loading, transforming, and comparing PDB structures."""

import logging
import re
import numpy as np
from collections import defaultdict

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from .seqtools import global_alignment_seqs

logger = logging.getLogger(__name__)

_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _resolve_pdb_chain_map(structure, chain_map=None):
    """Build a one-character chain ID mapping for PDB export."""
    chain_map = {} if chain_map is None else dict(chain_map)
    resolved_map = {}
    used_ids = set()

    for old_id, new_id in chain_map.items():
        if len(new_id) != 1 or new_id not in _PDB_CHAIN_IDS:
            raise ValueError(
                f"Invalid PDB chain ID {new_id!r} for mmCIF chain {old_id!r}"
            )
        if new_id in used_ids:
            raise ValueError(f"Duplicate target PDB chain ID {new_id!r}")
        used_ids.add(new_id)
        resolved_map[old_id] = new_id

    available_ids = [chain_id for chain_id in _PDB_CHAIN_IDS if chain_id not in used_ids]
    chain_iter = iter(available_ids)

    for chain in structure.get_chains():
        old_id = chain.id
        if old_id in resolved_map:
            continue
        if len(old_id) == 1 and old_id in _PDB_CHAIN_IDS and old_id not in used_ids:
            resolved_map[old_id] = old_id
            used_ids.add(old_id)
            continue
        try:
            resolved_map[old_id] = next(chain_iter)
        except StopIteration as exc:
            raise ValueError(
                "Too many chains for safe PDB export; provide a custom chain_map "
                "or keep the structure in mmCIF format."
            ) from exc

    return resolved_map


def _get_chain(structure, chain_id):
    """Return a chain object by ID or raise a descriptive error."""
    for chain in structure.get_chains():
        if chain.id == chain_id:
            return chain
    raise ValueError(f"Chain {chain_id!r} not found in structure")


def _get_protein_residues(chain):
    """Return amino acid residues from a chain."""
    return [residue for residue in chain.get_residues() if is_aa(residue)]


def _validate_chain_for_protein_alignment(chain):
    """Validate that a chain can be used safely for protein index-based logic."""
    hetero_residues = [
        residue for residue in chain.get_residues()
        if not is_aa(residue)
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
    ca_atoms = [atom for atom in chain.get_atoms() if atom.get_id() == 'CA']
    if len(protein_residues) != len(ca_atoms):
        raise ValueError(
            f"Chain {chain.id!r} has {len(protein_residues)} protein residues but "
            f"{len(ca_atoms)} CA atoms; residue-to-atom mapping is not clean."
        )

    return protein_residues


def get_pdb_structure(pdb_id, target_folder = "."):
    """Fetch a structure from the PDB and load it with Biopython.

    Args:
        pdb_id: Four-character PDB identifier.
        target_folder: Directory used to store downloaded structure files.

    Returns:
        Loaded Biopython structure object. Falls back to mmCIF parsing if no
        PDB file can be loaded.
    """
    from Bio.PDB import PDBList

    pdb_id = pdb_id.lower()
    pdbl = PDBList()
    
    pdb_error = None
    try:
        logger.debug("Retrieving PDB structure %s in PDB format", pdb_id)
        pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=target_folder, file_format='pdb')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        return structure
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        pdb_error = exc
        logger.warning(
            "Could not load structure %s in PDB format; trying mmCIF: %s",
            pdb_id,
            exc,
        )

    try:
        logger.debug("Retrieving PDB structure %s in mmCIF format", pdb_id)
        cif_file = pdbl.retrieve_pdb_file(pdb_id, pdir=target_folder, file_format="mmCif")
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, cif_file)
        return structure
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        if pdb_error is not None:
            raise RuntimeError(
                f"Failed to load structure {pdb_id!r} as PDB ({pdb_error}) "
                f"and mmCIF ({exc})"
            ) from exc
        raise RuntimeError(f"Failed to load structure {pdb_id!r} as mmCIF") from exc
    
def convert_cif_to_pdb(cif_file, pdb_file, chain_map=None, return_chain_map=False):
    """Convert an mmCIF structure file to PDB format.

    Args:
        cif_file: Path to the input mmCIF file.
        pdb_file: Path to the output PDB file.
        chain_map: Optional mapping from mmCIF chain IDs to one-character PDB
            chain IDs.
        return_chain_map: Return the resolved chain ID mapping together with the
            structure.

    Returns:
        Loaded Biopython structure object written to ``pdb_file``. If
        ``return_chain_map`` is ``True``, also returns the resolved chain ID
        mapping used for export.
    """
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', cif_file)
    resolved_map = _resolve_pdb_chain_map(structure, chain_map)

    for chain in structure.get_chains():
        original_id = chain.id
        chain.xtra["original_chain_id"] = original_id
        chain.id = resolved_map[original_id]

    structure.xtra["chain_id_map"] = resolved_map

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)
    logger.info("Converted mmCIF file %s to PDB file %s", cif_file, pdb_file)
    logger.debug("Chain ID mapping used for PDB export: %s", resolved_map)
    if return_chain_map:
        return structure, resolved_map
    return structure

def load_pdb_from_file(pdb_file):
    """Load a PDB structure from a local file.

    Args:
        pdb_file: Path to the local PDB file.

    Returns:
        Loaded Biopython structure object.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    return structure

def get_seqres_from_pdb(pdb_file):
    """Extract SEQRES sequences from a PDB text file.

    Args:
        pdb_file: Path to the PDB file.

    Returns:
        Dictionary mapping chain IDs to one-letter amino acid sequences.
    """
    with open(pdb_file, 'r') as fp:
        pdb_lines = fp.readlines()

    seqres = defaultdict(list)
    seqres_line = re.compile(r'^SEQRES\s+\d+\s+([A-Z])\s+\d+\s+(.*?)\s+$')
    for line in pdb_lines:
        if seqres_line.match(line):
            groups = seqres_line.search(line)
            seqres[groups.group(1)] += groups.group(2).split(' ')
    
    seqres = {
        chain: ''.join([map_three_to_one(three) for three in seq]) for chain, seq in seqres.items()
    }
    return seqres

def extract_chain(structure, chain_id):
    """Return a copy of the structure containing only selected chains.

    Args:
        structure: Input structure containing one or more chains.
        chain_id: Single chain ID or iterable of chain IDs to keep.

    Returns:
        Copy of the structure with all other chains removed.
    """
    if isinstance(chain_id, str):
        chain_id = [chain_id]
    struc = structure.copy()
    chains_to_keep = []
    for model in struc:
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
        raise Exception(f'Chain {chain_id} not found in structure!')
    return struc


def save_structure_to_file(structure, filename):
    """Write a structure to a PDB file.

    Args:
        structure: Structure to save.
        filename: Output path for the PDB file.
    """
    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(structure, )
    io.save(filename)
    logger.info("Saved structure to %s", filename)

def remove_wather_molecules(structure):
    """Remove water molecules and hetero residues from a structure.

    Args:
        structure: Structure to process in place.

    Returns:
        The modified structure without water molecules or hetero residues.
    """
    for model in structure:
        for chain in model:
            residues_to_remove = [res for res in chain if res.get_resname() == 'HOH' or res.get_full_id()[-1][0] != " "]
            for res in residues_to_remove:
                chain.detach_child(res.id)
    return structure

def map_three_to_one(res):
    """Map a three-letter residue code to a one-letter amino acid code.

    Unknown residue names are converted to ``"X"``.

    Args:
        res: Three-letter residue code.

    Returns:
        One-letter amino acid code.
    """
    from Bio.Data import IUPACData
    # mapping provided by Biopython (keys may have varied casing), use direct checks
    iupac_map = IUPACData.protein_letters_3to1
    # common non-standard residue fallbacks (PDB resnames are often uppercase)
    fallbacks = {'MSE': 'M', 'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z'}
    if not res:
        return 'X'
    # try different casings against the iupac_map
    for key in (res, res.upper(), res.title(), res.lower()):
        if key in iupac_map:
            return iupac_map[key]
    # fallback for common non-standard PDB residue names
    if res.upper() in fallbacks:
        return fallbacks[res.upper()]
    return 'X'

def get_aa_sequence(structure, show_gaps=True):
    """Extract per-chain amino acid sequences from a structure.

    Args:
        structure: Structure to extract sequences from.
        show_gaps: Insert ``"-"`` for missing residue index positions between
            amino acid residues.

    Returns:
        Dictionary mapping chain IDs to one-letter amino acid sequences. Only
        amino acid residues are included.
    """
    chain_seqs = {}
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        last_res_id = None
        sequence = ''
        for res in chain.get_residues():
            if not is_aa(res):
                continue
            _, res_id, _ = res.get_id()
            res_name = res.get_resname()
            if show_gaps and last_res_id is not None and res_id > last_res_id + 1:
                jump_size = res_id - last_res_id - 1
                sequence += '-'*jump_size
            sequence += map_three_to_one(res_name)
            last_res_id = res_id

        chain_seqs[chain_id] = sequence

    return chain_seqs

def get_rmsd(structure1, structure2, ca_only=True):
    """Compute RMSD between two structures using Biopython superposition.

    Args:
        structure1: Reference structure.
        structure2: Structure compared to the reference.
        ca_only: Use only C-alpha atoms when ``True``.

    Returns:
        RMSD value after optimal superposition.
    """
    from Bio.PDB import Superimposer

    if ca_only:
        atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
        atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']
    else:
        atoms1 = structure1.get_atoms() 
        atoms2 = structure2.get_atoms()
    
    si = Superimposer()
    si.set_atoms(atoms1, atoms2)
    return si.rms

def align_structure(structure1, structure2):
    """Align one structure onto another using C-alpha atoms.

    Args:
        structure1: Reference structure.
        structure2: Structure to align onto ``structure1``.

    Returns:
        Copy of ``structure2`` after superposition onto ``structure1``.
    """
    from Bio.PDB import Superimposer

    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']
    
    si = Superimposer()
    si.set_atoms(atoms1, atoms2)

    target = structure2.copy()
    si.apply(target.get_atoms())

    return target

def _identify_homo_aa(gapped_a, gapped_b):
    """Identify matching ungapped residue indices from two aligned sequences.

    Args:
        gapped_a: First aligned sequence containing gap characters.
        gapped_b: Second aligned sequence containing gap characters.

    Returns:
        Two index lists describing corresponding ungapped residue positions.
    """
    a_pos = 0
    b_pos = 0
    a_select = []
    b_select = []
    for a, b in zip(gapped_a, gapped_b):
        if a != "-" and b != "-":
            a_select.append(a_pos)
            b_select.append(b_pos)
            a_pos += 1
            b_pos += 1
        elif a == "-":
            b_pos += 1
        elif b == "-":
            a_pos += 1

    return a_select, b_select

def align_homologs(structure1, structure2, chain1, chain2):
    """Align homologous residues between two selected chains.

    The residue correspondence is derived from a global sequence alignment and
    the resulting superposition is applied to ``structure2``.

    Args:
        structure1: Reference structure.
        structure2: Structure to align onto ``structure1``.
        chain1: Chain ID in ``structure1``.
        chain2: Chain ID in ``structure2``.

    Returns:
        Copy of ``structure2`` after homolog-based superposition.
    """
    from Bio.PDB import Superimposer

    chain_obj_a = _get_chain(structure1, chain1)
    chain_obj_b = _get_chain(structure2, chain2)
    _validate_chain_for_protein_alignment(chain_obj_a)
    _validate_chain_for_protein_alignment(chain_obj_b)

    seqs_a = get_aa_sequence(structure1)
    seqs_b = get_aa_sequence(structure2)
    gapped_seq_a, gapped_seq_b = global_alignment_seqs(seqs_a[chain1], seqs_b[chain2])
    homo_selected_a,  homo_selected_b = _identify_homo_aa(gapped_seq_a, gapped_seq_b)

    structure1 = extract_chain(structure1, chain1)
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
    atoms1_aligned = [atom for k, atom in enumerate(atoms1) if k in homo_selected_a]

    structure2 = extract_chain(structure2, chain2)
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']
    atoms2_aligned = [atom for k, atom in enumerate(atoms2) if k in homo_selected_b]
    
    si = Superimposer()
    si.set_atoms(atoms1_aligned, atoms2_aligned)

    target = structure2.copy()
    si.apply(target.get_atoms())

    return target

def get_alignment(structure1, structure2, chain1=None, chain2=None):
    """Create a Biopython superimposer for two structures or chains.

    Args:
        structure1: Reference structure.
        structure2: Structure to align to the reference.
        chain1: Optional chain ID to extract from ``structure1``.
        chain2: Optional chain ID to extract from ``structure2``.

    Returns:
        Configured ``Bio.PDB.Superimposer`` instance.
    """
    from Bio.PDB import Superimposer
    
    structure1 = extract_chain(structure1, chain1) if chain1 is not None else structure1
    structure2 = extract_chain(structure2, chain2) if chain2 is not None else structure2
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']

    si = Superimposer()
    si.set_atoms(atoms1, atoms2)
    return si

def apply_transformation(superimposer, structure):
    """Apply a fitted Biopython superposition to a structure.

    Args:
        superimposer: Fitted ``Bio.PDB.Superimposer`` instance.
        structure: Structure transformed in place.

    Returns:
        The transformed structure.
    """
    superimposer.apply(structure.get_atoms())
    return structure

def get_residue_coords(structure, c_alpha: bool = False):
    """Collect residue-wise coordinate vectors from a structure.

    Args:
        structure: Structure providing residue coordinates.
        c_alpha: Use only C-alpha coordinates for each residue.

    Returns:
        List of ``(residue_id, residue_name, coordinates)`` tuples.
    """
    centers = []
    all_residues = structure.get_residues()
    for r, residue in enumerate(all_residues):
        if not is_aa(residue):
            continue
        
        all_atoms = [res for res in residue.get_atoms()]
        if c_alpha:
            all_atoms = list(filter(lambda res: res.id == "CA", all_atoms))

        coord = [res.get_vector() for res in all_atoms]
        res_id = residue.id[1]
        res_name = residue.get_resname()
        centers.append((res_id, res_name, coord))
    return centers

def get_min_dist(atom_list_a, atom_list_b, cutoff=30.0):
    """Compute the minimal Euclidean distance between two atom lists.

    Args:
        atom_list_a: Iterable of coordinate vectors for the first residue set.
        atom_list_b: Iterable of coordinate vectors for the second residue set.
        cutoff: Retained for API compatibility and currently not used to prune
            the distance search.

    Returns:
        Minimal pairwise distance between both atom lists.
    """
    min_dist = np.inf
    for a in atom_list_a:
        for b in atom_list_b:
            dist = np.linalg.norm(a - b)
            if dist < min_dist:
                min_dist = dist
                if min_dist == 0.0:
                    return 0.0

    return float(min_dist)

def get_interaction_residues(struc, chain_a, chain_b, cutoff=5.0):
    """Find residue pairs across two chains within a distance cutoff.

    Args:
        struc: Structure containing both chains.
        chain_a: First chain ID.
        chain_b: Second chain ID.
        cutoff: Maximum atom-to-atom distance for an interaction.

    Returns:
        List of interacting residue pairs with residue IDs, names, and distance.
    """
    chain_a_struc = extract_chain(struc, chain_a)
    all_atoms_a = get_residue_coords(chain_a_struc)

    chain_b_struc = extract_chain(struc, chain_b)
    all_atoms_b = get_residue_coords(chain_b_struc)

    interaction = []
    for a, a_type, vec_a in all_atoms_a:
        for b, b_type, vec_b  in all_atoms_b:
            dist = get_min_dist(vec_a, vec_b)
            if dist <= cutoff:
                interaction.append([a, a_type, b, b_type, dist])
    return interaction

def clip_chain(structure, chain_seqs: dict, verbose: bool = False):
    """Trim residues from chains that align to gaps in target sequences.

    Args:
        structure: Structure modified in place.
        chain_seqs: Dictionary mapping chain IDs to expected amino acid
            sequences.
        verbose: Log the number and IDs of removed residues for each processed
            chain.

    Returns:
        The modified structure with clipped chains.
    """
    seqs = get_aa_sequence(structure, show_gaps=False)
    
    for chain in structure.get_chains():
        if chain.id in chain_seqs.keys():
            protein_residues = _validate_chain_for_protein_alignment(chain)
            gapped_pdb_seq, gapped_seq = global_alignment_seqs(
                seqs[chain.id],
                chain_seqs[chain.id],
                gap_penalty=(-10, -1)
            )           
            remove = []
            if len(gapped_pdb_seq.replace('-', '')) != len(protein_residues):
                raise ValueError(
                    f"Protein residue count mismatch while clipping chain {chain.id!r}: "
                    f"{len(protein_residues)} residues vs alignment-derived "
                    f"{len(gapped_pdb_seq.replace('-', ''))} positions."
                )
            residue_index = 0
            for pdb_residue, target_residue in zip(gapped_pdb_seq, gapped_seq):
                if pdb_residue == '-':
                    continue
                residue = protein_residues[residue_index]
                residue_index += 1
                if target_residue == '-':
                    remove.append(residue.id)

            if verbose:
                logger.info(
                    "Removing %d residues from chain %s: %s",
                    len(remove),
                    chain.id,
                    remove,
                )
            for rm_id in remove:
                chain.detach_child(rm_id)
                    
    return structure

def move_to_center(structure):
    """Translate a structure copy so its center of mass is at the origin.

    Args:
        structure: Structure to translate.

    Returns:
        Translated copy of the structure.
    """
    m = structure.center_of_mass()
    rot_structure = structure.copy()
    for residue in rot_structure.get_residues():
        residue.transform(np.eye(3), -m)
    return rot_structure

def superimpose_PCA(structure, apply_rot=True, apply_shift=True):
    """Reorient a structure along its principal component axes.

    Args:
        structure: Structure to transform.
        apply_rot: Apply the PCA-derived rotation.
        apply_shift: Center coordinates before rotation.

    Returns:
        Tuple containing the transformed structure copy, translation vector,
        and rotation matrix.
    """
    from Bio.PDB.Superimposer import Superimposer

    coords = get_residue_coords(structure, c_alpha=True)
    X = np.array([v.get_array() for _, _, (v, ) in coords])
    shift = -X.mean(0)
    X_centered = X + shift
    _, EV = np.linalg.eig(np.cov(X_centered.T))
    rot = np.linalg.inv(EV)

    rot_structure = structure.copy()
    for residue in rot_structure.get_residues():
        if apply_shift:
            residue.transform(np.eye(3), shift)
        if apply_rot:
            residue.transform(rot,  np.zeros(3))

    return rot_structure, shift, rot

def reset_index(structure):
    """Renumber residues in each chain starting from 1.

    Args:
        structure: Structure modified in place.

    Returns:
        The modified structure with reset residue indices.
    """
    for model in structure:
        for c, chain in enumerate(model):
            #chain.id = chr(ord('A') + c) 
            for i, residue in enumerate(chain):
                a, _, b = residue.id
                residue.id = (a, i+1, b)
    
    return structure

def rename_chain(structure, chain_ids):
    """Rename chains in a structure according to a mapping.

    Args:
        structure: Structure modified in place.
        chain_ids: Mapping from old chain IDs to new chain IDs.

    Returns:
        The modified structure with renamed chains.
    """
    for model in structure:
        for c, chain in enumerate(model):
            if chain.id in chain_ids.keys():
                chain.id = chain_ids[chain.id]

    return structure

def plot_structure(structure):
    """Create a ``py3Dmol`` view for a structure.

    Args:
        structure: Structure to visualize.

    Returns:
        Configured ``py3Dmol.view`` object.
    """
    import py3Dmol
    from Bio.PDB import PDBIO
    from io import StringIO

    buf = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(buf)
    pdb_str = buf.getvalue()

    view = py3Dmol.view(width=800, height=400)
    view.addModel(pdb_str, 'pdb')
    view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    view.zoomTo()
    return view
