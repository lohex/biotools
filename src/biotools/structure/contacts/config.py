"""Named and serializable contact criterion profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class ContactConfig:
    """Numerical criteria used by contact and water-bridge detectors.

    Distances are in Angstrom and angles in degrees.  The values in
    ``refined`` are deliberately exposed as starting criteria rather than
    presented as universal physical constants.
    """

    name: str = "refined"
    hydrogen_bond_distance: float = 3.5
    hydrogen_bond_angle: float = 150.0
    salt_bridge_distance: float = 5.5
    hydrophobic_distance: float = 4.0
    vdw_tolerance: float = 0.5
    steric_clash_overlap: float = 0.4
    pi_distance: float = 5.5
    pi_parallel_angle: float = 30.0
    pi_t_shaped_angle: float = 60.0
    pi_parallel_offset: float = 2.0
    pi_projection_tolerance: float = 0.75
    ring_planarity_tolerance: float = 0.15
    pi_min_nearest_atom_distance: float = 1.5
    cation_pi_distance: float = 6.0
    cation_pi_min_height: float = 1.5
    water_anchor_min_distance: float = 2.6
    water_anchor_max_distance: float = 3.5
    exclude_one_four: bool = False
    refined_geometry: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ContactConfig":
        return cls(**dict(values))

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


LEGACY = ContactConfig(
    name="legacy",
    hydrogen_bond_distance=3.0,
    cation_pi_min_height=0.0,
    ring_planarity_tolerance=1_000_000.0,
    pi_min_nearest_atom_distance=0.0,
    refined_geometry=False,
)

REFINED = ContactConfig()


def get_contact_config(
    profile: str | ContactConfig = "refined",
) -> ContactConfig:
    """Resolve a built-in profile name or return a supplied configuration."""
    if isinstance(profile, ContactConfig):
        return profile
    profiles = {LEGACY.name: LEGACY, REFINED.name: REFINED}
    try:
        return profiles[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown contact profile {profile!r}; choose one of "
            f"{', '.join(sorted(profiles))}"
        ) from exc
