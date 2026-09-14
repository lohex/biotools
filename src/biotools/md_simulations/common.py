"""Shared OpenMM simulation helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from openmm import Platform, XmlSerializer

Ensemble: TypeAlias = Literal["NVT", "NPT"]
ResumeMode: TypeAlias = Literal[
    "auto", "state", "checkpoint", "coordinates"
]
ResolvedResumeMode: TypeAlias = Literal[
    "state", "checkpoint", "coordinates"
]


@runtime_checkable
class MDStageResult(Protocol):
    """Structural result that can be passed into the next MD pipeline stage."""

    output_path: Path


MDInput: TypeAlias = str | PathLike[str] | MDStageResult


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration identity used to validate exact checkpoint restarts."""

    ensemble: Ensemble
    temperature_k: float
    pressure_bar: float | None
    timestep_fs: float
    friction_per_ps: float
    forcefield_files: tuple[str, ...]
    nonbonded_cutoff_nm: float
    barostat_interval_steps: int | None
    system_hash: str
    integrator_hash: str
    platform_name: str
    platform_properties: tuple[tuple[str, str], ...] = ()

    def checkpoint_compatible(self, other: SimulationConfig) -> bool:
        """Return whether an exact OpenMM checkpoint can be loaded safely."""
        return (
            self.ensemble == other.ensemble
            and self.system_hash == other.system_hash
            and self.integrator_hash == other.integrator_hash
            and self.platform_name == other.platform_name
            and self.platform_properties == other.platform_properties
        )


def _serialized_hash(value: object) -> str:
    serialized = XmlSerializer.serialize(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def simulation_config(
    simulation,
    system,
    integrator,
    *,
    ensemble: Ensemble,
    temperature_k: float,
    pressure_bar: float | None,
    timestep_fs: float,
    friction_per_ps: float,
    forcefield_files: tuple[str, ...],
    nonbonded_cutoff_nm: float,
    barostat_interval_steps: int | None,
) -> SimulationConfig:
    """Capture user-facing settings and exact OpenMM compatibility hashes."""
    platform = simulation.context.getPlatform()
    properties = tuple(
        sorted(
            (
                name,
                platform.getPropertyValue(simulation.context, name),
            )
            for name in platform.getPropertyNames()
        )
    )
    return SimulationConfig(
        ensemble=ensemble,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        timestep_fs=timestep_fs,
        friction_per_ps=friction_per_ps,
        forcefield_files=forcefield_files,
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
        barostat_interval_steps=barostat_interval_steps,
        system_hash=_serialized_hash(system),
        integrator_hash=_serialized_hash(integrator),
        platform_name=platform.getName(),
        platform_properties=properties,
    )


def structure_path(input_file: MDInput) -> Path:
    """Resolve a path or the final structure from an MD result object."""
    if isinstance(input_file, (str, PathLike)):
        return Path(input_file)
    output_path = getattr(input_file, "output_path", None)
    if output_path is None:
        raise TypeError(
            "input_file must be path-like or expose an output_path attribute"
        )
    return Path(output_path)


def _existing_restart_path(name: str, value: object) -> Path:
    path = Path(value)  # type: ignore[arg-type]
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def resolve_resume_input(
    input_file: MDInput,
    *,
    state_input_file: str | PathLike[str] | None,
    checkpoint_input_file: str | PathLike[str] | None,
    resume_from: ResumeMode,
    target_config: SimulationConfig,
) -> tuple[Path | None, Path | None, ResolvedResumeMode]:
    """Resolve explicit or result-derived continuation files."""
    if resume_from not in {"auto", "state", "checkpoint", "coordinates"}:
        raise ValueError(
            "resume_from must be 'auto', 'state', 'checkpoint', or "
            "'coordinates'"
        )
    if state_input_file is not None and checkpoint_input_file is not None:
        raise ValueError(
            "state_input_file and checkpoint_input_file are mutually exclusive"
        )
    if state_input_file is not None:
        if resume_from not in {"auto", "state"}:
            raise ValueError(
                "resume_from conflicts with the explicit state_input_file"
            )
        return (
            _existing_restart_path("state_input_file", state_input_file),
            None,
            "state",
        )
    if checkpoint_input_file is not None:
        if resume_from not in {"auto", "checkpoint"}:
            raise ValueError(
                "resume_from conflicts with the explicit checkpoint_input_file"
            )
        return (
            None,
            _existing_restart_path(
                "checkpoint_input_file", checkpoint_input_file
            ),
            "checkpoint",
        )

    is_result = not isinstance(input_file, (str, PathLike))
    if not is_result:
        if resume_from in {"state", "checkpoint"}:
            raise ValueError(
                f"resume_from='{resume_from}' requires an MD result object or "
                f"an explicit {resume_from}_input_file"
            )
        return None, None, "coordinates"
    if resume_from == "coordinates":
        return None, None, "coordinates"

    state_value = getattr(input_file, "state_path", None)
    checkpoint_value = getattr(input_file, "checkpoint_path", None)
    source_config = getattr(input_file, "simulation_config", None)
    checkpoint_is_compatible = bool(
        checkpoint_value is not None
        and isinstance(source_config, SimulationConfig)
        and source_config.checkpoint_compatible(target_config)
    )
    if resume_from == "checkpoint":
        if checkpoint_value is None:
            raise ValueError("Input result does not provide a checkpoint_path")
        if not checkpoint_is_compatible:
            raise ValueError(
                "Input checkpoint is not compatible with the requested "
                "simulation configuration"
            )
        return (
            None,
            _existing_restart_path("checkpoint_path", checkpoint_value),
            "checkpoint",
        )
    if resume_from == "state":
        if state_value is None:
            raise ValueError("Input result does not provide a state_path")
        return (
            _existing_restart_path("state_path", state_value),
            None,
            "state",
        )

    if checkpoint_is_compatible:
        checkpoint_path = Path(checkpoint_value)  # type: ignore[arg-type]
        if checkpoint_path.is_file():
            return None, checkpoint_path, "checkpoint"
    if state_value is not None:
        return (
            _existing_restart_path("state_path", state_value),
            None,
            "state",
        )
    if checkpoint_value is not None:
        if checkpoint_is_compatible:
            raise FileNotFoundError(
                f"checkpoint_path not found: {Path(checkpoint_value)}"
            )
        raise ValueError(
            "Input result only provides an incompatible checkpoint; write an "
            "XML State in the preceding stage or use "
            "resume_from='coordinates' explicitly"
        )
    return None, None, "coordinates"


def validate_io_paths(
    input_file: MDInput,
    output_file: str | PathLike[str],
) -> tuple[Path, Path]:
    """Return validated input and output paths without overwriting the input."""
    input_path = structure_path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")
    return input_path, output_path


def simulation_platform_options(
    platform_name: str | None,
    platform_properties: Mapping[str, str] | None,
) -> dict[str, object]:
    """Build validated keyword arguments for ``openmm.app.Simulation``."""
    if platform_name is not None and not platform_name.strip():
        raise ValueError("platform_name must not be empty")
    if platform_properties and platform_name is None:
        raise ValueError(
            "platform_properties requires an explicit platform_name"
        )
    if platform_name is None:
        return {}

    options: dict[str, object] = {
        "platform": Platform.getPlatformByName(platform_name)
    }
    if platform_properties:
        options["platformProperties"] = dict(platform_properties)
    return options
