"""Metadata retrieval from the official RCSB PDB Data API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_RCSB_CORE_API = "https://data.rcsb.org/rest/v1/core"
_PDB_ID_PATTERN = re.compile(r"[A-Za-z0-9]{4}")


class RCSBMetadataError(RuntimeError):
    """Raised when RCSB metadata cannot be downloaded or interpreted."""


@dataclass(frozen=True)
class PDBMetadata:
    """Selected entry metadata from RCSB PDB.

    Attributes:
        pdb_id: Normalized uppercase four-character PDB identifier.
        organisms: Unique scientific source-organism names across all polymer
            entities, in their first-seen RCSB order.
        deposited: Initial deposition date.
        released: Initial public release date, or ``None`` when RCSB provides
            no release date.
    """

    pdb_id: str
    organisms: tuple[str, ...]
    deposited: date
    released: date | None


def _fetch_rcsb_json(url: str, *, timeout: float) -> dict[str, Any]:
    """Fetch one JSON object with consistent RCSB error handling."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "biotools/0.1 RCSB-metadata-client",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(f"RCSB resource not found: {url}") from error
        raise RCSBMetadataError(
            f"RCSB request failed with HTTP status {error.code}: {url}"
        ) from error
    except (URLError, OSError) as error:
        raise RCSBMetadataError(f"Could not reach RCSB Data API: {url}") from error

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RCSBMetadataError(f"RCSB returned invalid JSON: {url}") from error
    if not isinstance(decoded, dict):
        raise RCSBMetadataError(f"RCSB returned an unexpected JSON value: {url}")
    return decoded


def _parse_rcsb_date(value: Any, *, field_name: str) -> date:
    """Parse the date component of an RCSB ISO-8601 timestamp."""
    if not isinstance(value, str):
        raise RCSBMetadataError(f"RCSB response is missing {field_name}")
    try:
        return date.fromisoformat(value[:10])
    except ValueError as error:
        raise RCSBMetadataError(
            f"RCSB returned an invalid {field_name}: {value!r}"
        ) from error


def get_pdb_metadata(
    pdb_id: str,
    *,
    timeout: float = 15.0,
) -> PDBMetadata:
    """Fetch source organisms and accession dates for a PDB entry from RCSB.

    Args:
        pdb_id: Four-character Protein Data Bank identifier.
        timeout: Per-request network timeout in seconds.

    Returns:
        Frozen :class:`PDBMetadata` with scientific organism names and dates.

    Raises:
        ValueError: If the PDB identifier or timeout is invalid.
        FileNotFoundError: If RCSB has no matching entry or polymer entity.
        RCSBMetadataError: If RCSB is unavailable or returns malformed data.
    """
    if not isinstance(pdb_id, str) or _PDB_ID_PATTERN.fullmatch(pdb_id) is None:
        raise ValueError("pdb_id must be a four-character alphanumeric identifier")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    normalized_id = pdb_id.upper()
    entry_url = f"{_RCSB_CORE_API}/entry/{quote(normalized_id, safe='')}"
    entry = _fetch_rcsb_json(entry_url, timeout=timeout)

    accession_info = entry.get("rcsb_accession_info")
    identifiers = entry.get("rcsb_entry_container_identifiers")
    if not isinstance(accession_info, dict) or not isinstance(identifiers, dict):
        raise RCSBMetadataError("RCSB entry response is missing metadata containers")
    deposited = _parse_rcsb_date(
        accession_info.get("deposit_date"),
        field_name="deposit_date",
    )
    release_value = accession_info.get("initial_release_date")
    released = (
        _parse_rcsb_date(release_value, field_name="initial_release_date")
        if release_value is not None
        else None
    )

    entity_ids = identifiers.get("polymer_entity_ids", [])
    if not isinstance(entity_ids, list):
        raise RCSBMetadataError("RCSB entry has invalid polymer_entity_ids")
    organisms: list[str] = []
    seen_organisms: set[str] = set()
    for entity_id in entity_ids:
        encoded_entity_id = quote(str(entity_id), safe="")
        entity_url = (
            f"{_RCSB_CORE_API}/polymer_entity/{normalized_id}/"
            f"{encoded_entity_id}"
        )
        entity = _fetch_rcsb_json(entity_url, timeout=timeout)
        sources = entity.get("rcsb_entity_source_organism", [])
        if sources is None:
            sources = []
        if not isinstance(sources, list):
            raise RCSBMetadataError(
                f"RCSB polymer entity {entity_id!r} has invalid organism data"
            )
        for source in sources:
            if not isinstance(source, dict):
                continue
            scientific_name = source.get("ncbi_scientific_name")
            if not isinstance(scientific_name, str) or not scientific_name.strip():
                scientific_name = source.get("scientific_name")
            if not isinstance(scientific_name, str):
                continue
            scientific_name = scientific_name.strip()
            if scientific_name and scientific_name not in seen_organisms:
                seen_organisms.add(scientific_name)
                organisms.append(scientific_name)

    return PDBMetadata(
        pdb_id=normalized_id,
        organisms=tuple(organisms),
        deposited=deposited,
        released=released,
    )
