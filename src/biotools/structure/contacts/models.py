"""Data contracts for prepared contact systems and analysis results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, order=True)
class AtomReference:
    """Stable semantic atom identity, independent of coordinate-array order."""

    structure_id: str
    model_id: str
    chain_id: str
    hetero_flag: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atom_name: str
    altloc: str

    @classmethod
    def from_atom(cls, atom: Any, structure_id: str = "") -> "AtomReference":
        residue = atom.get_parent()
        chain = residue.get_parent()
        model = chain.get_parent()
        hetero, number, insertion = residue.id
        return cls(
            structure_id=str(structure_id),
            model_id=str(model.id),
            chain_id=str(chain.id),
            hetero_flag=str(hetero).strip(),
            residue_number=int(number),
            insertion_code=str(insertion).strip(),
            residue_name=str(residue.get_resname()).strip(),
            atom_name=str(atom.get_name()).strip(),
            altloc=str(atom.get_altloc()).strip(),
        )


@dataclass(frozen=True)
class GeometryMeasurement:
    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class CriterionResult:
    name: str
    satisfied: bool
    limit: str


@dataclass(frozen=True)
class ContactDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    atom: AtomReference | None = None


@dataclass(frozen=True)
class ContactObservation:
    """One atomic or chemical-group contact observation."""

    interaction_type: str
    partner_a: tuple[AtomReference, ...]
    partner_b: tuple[AtomReference, ...]
    role_a: str
    role_b: str
    geometry: tuple[GeometryMeasurement, ...]
    criteria: tuple[CriterionResult, ...]
    rule_profile: str
    evidence_level: str = "explicit_geometry"
    quality_flags: tuple[str, ...] = ()
    mediator_waters: tuple[AtomReference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ContactObservation":
        data = dict(values)
        data["partner_a"] = tuple(AtomReference(**value) for value in data["partner_a"])
        data["partner_b"] = tuple(AtomReference(**value) for value in data["partner_b"])
        data["geometry"] = tuple(GeometryMeasurement(**value) for value in data["geometry"])
        data["criteria"] = tuple(CriterionResult(**value) for value in data["criteria"])
        data["mediator_waters"] = tuple(
            AtomReference(**value) for value in data.get("mediator_waters", ())
        )
        data["quality_flags"] = tuple(data.get("quality_flags", ()))
        return cls(**data)


@dataclass
class ContactSystem:
    """Validated selection and shared topology used by every detector."""

    structure: Any = field(repr=False)
    model: Any = field(repr=False)
    chain_a: str
    chain_b: str
    residues_a: tuple[Any, ...] = field(repr=False)
    residues_b: tuple[Any, ...] = field(repr=False)
    water_residues: tuple[Any, ...] = field(repr=False)
    adjacency: Mapping[Any, tuple[Any, ...]] = field(repr=False)
    atom_references: Mapping[Any, AtomReference] = field(repr=False)
    diagnostics: tuple[ContactDiagnostic, ...]
    coverage: Mapping[str, int | float]
    topology_backend: str
    input_hash: str


@dataclass(frozen=True)
class ContactAnalysisResult:
    observations: tuple[ContactObservation, ...]
    diagnostics: tuple[ContactDiagnostic, ...]
    coverage: Mapping[str, int | float]
    config_hash: str
    rule_profile: str
    backend_versions: Mapping[str, str]
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "coverage": dict(self.coverage),
            "config_hash": self.config_hash,
            "rule_profile": self.rule_profile,
            "backend_versions": dict(self.backend_versions),
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ContactAnalysisResult":
        data = dict(values)
        data["observations"] = tuple(
            ContactObservation.from_dict(value) for value in data["observations"]
        )
        diagnostics = []
        for value in data["diagnostics"]:
            diagnostic = dict(value)
            if diagnostic.get("atom") is not None:
                diagnostic["atom"] = AtomReference(**diagnostic["atom"])
            diagnostics.append(ContactDiagnostic(**diagnostic))
        data["diagnostics"] = tuple(diagnostics)
        return cls(**data)
