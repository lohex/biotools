"""Versioned aggregation of typed contact and water-site features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .models import AtomReference, ContactAnalysisResult


@dataclass(frozen=True)
class ContactFeatureSet:
    schema: str
    features: Mapping[str, float]
    member_ids: Mapping[str, tuple[str, ...]]
    contact_config_hash: str


def _atom_id(reference: AtomReference) -> str:
    residue_position = f"{reference.residue_number}{reference.insertion_code}"
    return (
        f"{reference.model_id}:{reference.chain_id}:{reference.hetero_flag}:"
        f"{residue_position}:{reference.residue_name}:{reference.atom_name}:"
        f"{reference.altloc}"
    )


def aggregate_contact_features(
    contacts: ContactAnalysisResult,
    waters: Any | None = None,
    *,
    schema: str = "contact_features_v1",
) -> ContactFeatureSet:
    """Aggregate counts and distance summaries without discarding members."""
    if schema != "contact_features_v1":
        raise ValueError("schema must be 'contact_features_v1'")
    counts: dict[str, int] = defaultdict(int)
    distances: dict[str, list[float]] = defaultdict(list)
    members: dict[str, list[str]] = defaultdict(list)
    for observation in contacts.observations:
        interaction = observation.interaction_type
        counts[interaction] += 1
        member_atoms = [
            *(_atom_id(item) for item in observation.partner_a),
            "--",
            *(_atom_id(item) for item in observation.partner_b),
        ]
        member = "|".join(member_atoms)
        members[interaction].append(member)
        distance_measurements = [
            item.value for item in observation.geometry if "distance" in item.name
        ]
        if distance_measurements:
            distances[interaction].append(min(distance_measurements))

    features: dict[str, float] = {}
    for interaction in sorted(counts):
        values = distances[interaction]
        features[f"{interaction}.observation_count"] = float(counts[interaction])
        if values:
            features[f"{interaction}.distance_min"] = float(min(values))
            features[f"{interaction}.distance_median"] = float(np.median(values))
            features[f"{interaction}.distance_max"] = float(max(values))

    if waters is not None:
        sites = tuple(getattr(waters, "sites", ()))
        accepted = [site for site in sites if getattr(site, "status", "") == "accepted"]
        features["water_site.count"] = float(len(sites))
        features["water_site.accepted_count"] = float(len(accepted))
        members["water_site"] = [str(site.site_id) for site in sites]

    return ContactFeatureSet(
        schema=schema,
        features=MappingProxyType(features),
        member_ids=MappingProxyType(
            {key: tuple(values) for key, values in sorted(members.items())}
        ),
        contact_config_hash=contacts.config_hash,
    )
