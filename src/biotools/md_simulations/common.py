"""Shared OpenMM simulation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from openmm import Platform


def validate_io_paths(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
) -> tuple[Path, Path]:
    """Return validated input and output paths without overwriting the input."""
    input_path = Path(input_file)
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
