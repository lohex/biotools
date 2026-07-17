"""Loading, saving, and format conversion for protein structures."""

from collections import defaultdict
import logging
import re

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

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

    available_ids = [
        chain_id for chain_id in _PDB_CHAIN_IDS if chain_id not in used_ids
    ]
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


def get_pdb_structure(pdb_id, target_folder="."):
    """Fetch a structure from the PDB and load it with Biopython."""
    from Bio.PDB import PDBList

    pdb_id = pdb_id.lower()
    pdbl = PDBList()
    pdb_error = None
    try:
        logger.debug("Retrieving PDB structure %s in PDB format", pdb_id)
        pdb_file = pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=target_folder,
            file_format="pdb",
        )
        return PDBParser(QUIET=True).get_structure(pdb_id, pdb_file)
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        pdb_error = exc
        logger.warning(
            "Could not load structure %s in PDB format; trying mmCIF: %s",
            pdb_id,
            exc,
        )

    try:
        logger.debug("Retrieving PDB structure %s in mmCIF format", pdb_id)
        cif_file = pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=target_folder,
            file_format="mmCif",
        )
        return MMCIFParser(QUIET=True).get_structure(pdb_id, cif_file)
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        if pdb_error is not None:
            raise RuntimeError(
                f"Failed to load structure {pdb_id!r} as PDB ({pdb_error}) "
                f"and mmCIF ({exc})"
            ) from exc
        raise RuntimeError(f"Failed to load structure {pdb_id!r} as mmCIF") from exc


def convert_cif_to_pdb(cif_file, pdb_file, chain_map=None, return_chain_map=False):
    """Convert an mmCIF structure file to PDB format."""
    from Bio.PDB import PDBIO

    structure = MMCIFParser(QUIET=True).get_structure("struct", cif_file)
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
    """Load a PDB structure from a local file."""
    return PDBParser(QUIET=True).get_structure("struct", pdb_file)


def save_structure_to_file(structure, filename):
    """Write a structure to a PDB file."""
    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)
    logger.info("Saved structure to %s", filename)


def map_three_to_one(res):
    """Map a three-letter residue code to a one-letter amino acid code."""
    from Bio.Data import IUPACData

    iupac_map = IUPACData.protein_letters_3to1
    fallbacks = {"MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z"}
    if not res:
        return "X"
    for key in (res, res.upper(), res.title(), res.lower()):
        if key in iupac_map:
            return iupac_map[key]
    return fallbacks.get(res.upper(), "X")


def get_seqres_from_pdb(pdb_file):
    """Extract SEQRES sequences from a PDB text file."""
    with open(pdb_file, "r") as fp:
        pdb_lines = fp.readlines()

    seqres = defaultdict(list)
    seqres_line = re.compile(r"^SEQRES\s+\d+\s+([A-Z])\s+\d+\s+(.*?)\s+$")
    for line in pdb_lines:
        match = seqres_line.match(line)
        if match:
            seqres[match.group(1)] += match.group(2).split(" ")

    return {
        chain: "".join(map_three_to_one(three) for three in residues)
        for chain, residues in seqres.items()
    }
