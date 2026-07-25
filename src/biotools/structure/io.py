"""Loading, saving, and format conversion for protein structures."""

from __future__ import annotations

from collections import defaultdict
import logging
from os import PathLike
import re
from typing import Mapping, TYPE_CHECKING

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)

_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _resolve_pdb_chain_map(
    structure: Structure,
    chain_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a one-character chain ID mapping for PDB export.

    Args:
        structure: Structure whose chains need PDB-compatible identifiers.
        chain_map: Optional explicit mapping from source to target chain IDs.

    Returns:
        Complete mapping for every chain in ``structure``.

    Raises:
        ValueError: If an explicit ID is invalid or duplicated, or if the
            structure contains more chains than PDB format can represent.
    """
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


def get_pdb_structure_as_pdb(
    pdb_id: str,
    target_folder: str | PathLike[str] = ".",
    *,
    verbose: bool = True,
) -> Structure:
    """Fetch and parse a structure in legacy PDB/``.ent`` format.

    Args:
        pdb_id: Four-character Protein Data Bank identifier.
        target_folder: Directory in which the downloaded file is stored.
        verbose: Log download progress.

    Returns:
        Parsed Biopython structure.

    Raises:
        FileNotFoundError: If the PDB archive provides no legacy PDB file.
        OSError: If the file cannot be downloaded or read.
        ValueError: If the identifier or downloaded structure is invalid.
        PDBConstructionException: If Biopython cannot construct the structure.
    """
    from Bio.PDB import PDBList

    pdb_id = pdb_id.lower()
    if verbose:
        logger.info("Retrieving PDB structure %s in PDB format", pdb_id)
    pdb_file = PDBList().retrieve_pdb_file(
        pdb_id,
        pdir=target_folder,
        file_format="pdb",
    )
    if not pdb_file:
        raise FileNotFoundError(
            f"PDB entry {pdb_id!r} is not available in legacy PDB format"
        )
    return PDBParser(QUIET=True).get_structure(pdb_id, pdb_file)


def get_pdb_structure_as_mmcif(
    pdb_id: str,
    target_folder: str | PathLike[str] = ".",
    *,
    verbose: bool = True,
) -> Structure:
    """Fetch and parse a structure in mmCIF format.

    Args:
        pdb_id: Four-character Protein Data Bank identifier.
        target_folder: Directory in which the downloaded file is stored.
        verbose: Log download progress.

    Returns:
        Parsed Biopython structure.

    Raises:
        FileNotFoundError: If the PDB archive provides no mmCIF file.
        OSError: If the file cannot be downloaded or read.
        ValueError: If the identifier or downloaded structure is invalid.
        PDBConstructionException: If Biopython cannot construct the structure.
    """
    from Bio.PDB import PDBList

    pdb_id = pdb_id.lower()
    if verbose:
        logger.info("Retrieving PDB structure %s in mmCIF format", pdb_id)
    cif_file = PDBList().retrieve_pdb_file(
        pdb_id,
        pdir=target_folder,
        file_format="mmCif",
    )
    if not cif_file:
        raise FileNotFoundError(
            f"PDB entry {pdb_id!r} is not available in mmCIF format"
        )
    return MMCIFParser(QUIET=True).get_structure(pdb_id, cif_file)


def get_pdb_structure(
    pdb_id: str,
    target_folder: str | PathLike[str] = ".",
    prefer_mmcif: bool = False,
    *,
    verbose: bool = True,
) -> Structure:
    """Fetch a structure with an automatic format fallback.

    PDB/``.ent`` is attempted first by default. Set ``prefer_mmcif=True`` to
    attempt mmCIF first instead. In either case, the other format is used as a
    fallback if downloading or parsing the preferred format fails.

    Args:
        pdb_id: Four-character PDB identifier.
        target_folder: Directory used to store downloaded structure files.
        prefer_mmcif: Attempt mmCIF before legacy PDB/``.ent`` format.
        verbose: Log download attempts.

    Returns:
        Loaded Biopython structure object.

    Raises:
        RuntimeError: If neither format can be downloaded and parsed.
    """
    pdb_id = pdb_id.lower()

    if prefer_mmcif:
        first_format, first_loader = ("mmCIF", get_pdb_structure_as_mmcif)
        fallback_format, fallback_loader = ("PDB", get_pdb_structure_as_pdb)
    else:
        first_format, first_loader = ("PDB", get_pdb_structure_as_pdb)
        fallback_format, fallback_loader = ("mmCIF", get_pdb_structure_as_mmcif)

    try:
        if verbose:
            return first_loader(pdb_id, target_folder, verbose=True)
        return first_loader(pdb_id, target_folder)
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        first_error = exc
        if verbose:
            logger.warning(
                "Could not load structure %s in %s format; trying %s: %s",
                pdb_id,
                first_format,
                fallback_format,
                exc,
            )

    try:
        if verbose:
            return fallback_loader(pdb_id, target_folder, verbose=True)
        return fallback_loader(pdb_id, target_folder)
    except (OSError, FileNotFoundError, ValueError, PDBConstructionException) as exc:
        raise RuntimeError(
            f"Failed to load structure {pdb_id!r} as {first_format} "
            f"({first_error}) and {fallback_format} ({exc})"
        ) from exc


def convert_cif_to_pdb(
    cif_file: str | PathLike[str],
    pdb_file: str | PathLike[str],
    chain_map: Mapping[str, str] | None = None,
    return_chain_map: bool = False,
    *,
    verbose: bool = True,
) -> Structure | tuple[Structure, dict[str, str]]:
    """Convert an mmCIF structure file to PDB format.

    Args:
        cif_file: Source mmCIF file.
        pdb_file: Destination PDB file.
        chain_map: Optional explicit one-character chain ID mapping.
        return_chain_map: Return the resolved mapping along with the structure.
        verbose: Log the converted paths and resolved chain-ID mapping.

    Returns:
        Converted structure, optionally paired with the resolved chain map.

    Raises:
        ValueError: If chain identifiers cannot be represented safely in PDB.
        OSError: If an input or output file cannot be accessed.
    """
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
    if verbose:
        logger.info("Converted mmCIF file %s to PDB file %s", cif_file, pdb_file)
        logger.info("Chain ID mapping used for PDB export: %s", resolved_map)
    if return_chain_map:
        return structure, resolved_map
    return structure


def load_pdb_from_file(pdb_file: str | PathLike[str]) -> Structure:
    """Load a PDB structure from a local file.

    Args:
        pdb_file: Local PDB file to parse.

    Returns:
        Parsed Biopython structure.
    """
    return PDBParser(QUIET=True).get_structure("struct", pdb_file)


def save_structure_to_file(
    structure: Structure,
    filename: str | PathLike[str],
    *,
    verbose: bool = True,
) -> None:
    """Write a structure to a PDB file.

    Args:
        structure: Biopython structure to serialize.
        filename: Destination PDB path.
        verbose: Log the output path.
    """
    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)
    if verbose:
        logger.info("Saved structure to %s", filename)


def map_three_to_one(res: str | None) -> str:
    """Map a three-letter residue code to a one-letter amino-acid code.

    Args:
        res: Three-letter residue name in any letter case.

    Returns:
        One-letter code, including supported ambiguous and nonstandard codes;
        unknown or empty residue names yield ``"X"``.
    """
    from Bio.Data import IUPACData

    iupac_map = IUPACData.protein_letters_3to1
    fallbacks = {"MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z"}
    if not res:
        return "X"
    for key in (res, res.upper(), res.title(), res.lower()):
        if key in iupac_map:
            return iupac_map[key]
    return fallbacks.get(res.upper(), "X")


def get_seqres_from_pdb(
    pdb_file: str | PathLike[str],
) -> dict[str, str]:
    """Extract SEQRES sequences from a PDB text file.

    Args:
        pdb_file: PDB text file containing SEQRES records.

    Returns:
        Mapping from chain ID to one-letter amino-acid sequence.
    """
    with open(pdb_file, "r") as fp:
        pdb_lines = fp.readlines()

    seqres: defaultdict[str, list[str]] = defaultdict(list)
    seqres_line = re.compile(r"^SEQRES\s+\d+\s+([A-Z])\s+\d+\s+(.*?)\s+$")
    for line in pdb_lines:
        match = seqres_line.match(line)
        if match:
            seqres[match.group(1)] += match.group(2).split(" ")

    return {
        chain: "".join(map_three_to_one(three) for three in residues)
        for chain, residues in seqres.items()
    }
