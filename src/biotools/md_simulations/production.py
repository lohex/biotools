"""Fixed-length NVT and NPT molecular-dynamics production runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    ForceField,
    HBonds,
    NoCutoff,
    PDBFile,
    PME,
    Simulation,
    StateDataReporter,
    XTCReporter,
)
from openmm.unit import (
    bar,
    femtoseconds,
    kelvin,
    nanometer,
    picosecond,
)

from .common import (
    Ensemble,
    MDInput,
    ResolvedResumeMode,
    ResumeMode,
    SimulationConfig,
    resolve_resume_input,
    simulation_config,
    simulation_platform_options,
    validate_io_paths,
)
from .equilibration import _write_equilibration_outputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionResult:
    """Files and timing metadata from a completed production run."""

    output_path: Path
    trajectory_path: Path
    ensemble: Ensemble
    steps: int
    initial_step: int
    final_step: int
    elapsed_time_ps: float
    simulation_time_ps: float
    target_temperature_k: float
    target_pressure_bar: float | None
    timestep_fs: float
    trajectory_interval_steps: int
    log_path: Path | None = None
    state_path: Path | None = None
    checkpoint_path: Path | None = None
    log_interval_steps: int | None = None
    checkpoint_interval_steps: int | None = None
    input_state_path: Path | None = None
    input_checkpoint_path: Path | None = None
    appended: bool = False
    simulation_config: SimulationConfig | None = None
    resume_mode: ResolvedResumeMode = "coordinates"


def _validate_output_paths(
    input_path: Path,
    output_path: Path,
    trajectory_file: str | PathLike[str],
    log_file: str | PathLike[str] | None,
    state_output_file: str | PathLike[str] | None,
    checkpoint_output_file: str | PathLike[str] | None,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Resolve production outputs and reject ambiguous destinations."""
    trajectory_path = Path(trajectory_file)
    log_path = Path(log_file) if log_file is not None else None
    state_path = (
        Path(state_output_file) if state_output_file is not None else None
    )
    checkpoint_path = (
        Path(checkpoint_output_file)
        if checkpoint_output_file is not None
        else None
    )
    if trajectory_path.suffix.lower() not in {".dcd", ".xtc"}:
        raise ValueError("trajectory_file must have a .dcd or .xtc suffix")

    destinations = [output_path]
    for name, path in (
        ("trajectory_file", trajectory_path),
        ("log_file", log_path),
        ("state_output_file", state_path),
        ("checkpoint_output_file", checkpoint_path),
    ):
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path == input_path.resolve():
            raise ValueError(f"{name} must not overwrite input_file")
        if any(
            resolved_path == destination.resolve()
            for destination in destinations
        ):
            raise ValueError("Production output files must all be different")
        destinations.append(path)
    return trajectory_path, log_path, state_path, checkpoint_path


def _add_reporters(
    simulation: Simulation,
    *,
    trajectory_path: Path,
    trajectory_interval_steps: int,
    log_path: Path | None,
    log_interval_steps: int,
    checkpoint_path: Path | None,
    checkpoint_interval_steps: int,
    periodic: bool,
    append: bool,
    total_steps: int,
) -> None:
    """Attach trajectory, state-data, and checkpoint reporters."""
    trajectory_reporter_type = (
        DCDReporter if trajectory_path.suffix.lower() == ".dcd" else XTCReporter
    )
    simulation.reporters.append(
        trajectory_reporter_type(
            str(trajectory_path),
            trajectory_interval_steps,
            append=append,
            enforcePeriodicBox=periodic,
        )
    )
    if log_path is not None:
        simulation.reporters.append(
            StateDataReporter(
                str(log_path),
                log_interval_steps,
                step=True,
                time=True,
                progress=True,
                remainingTime=True,
                speed=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=periodic,
                density=periodic,
                separator=",",
                totalSteps=total_steps,
                append=append,
            )
        )
    if checkpoint_path is not None:
        simulation.reporters.append(
            CheckpointReporter(
                str(checkpoint_path), checkpoint_interval_steps
            )
        )


def run_production(
    input_file: MDInput,
    output_file: str | PathLike[str],
    *,
    trajectory_file: str | PathLike[str],
    steps: int,
    ensemble: str = "NPT",
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    timestep_fs: float = 2.0,
    friction_per_ps: float = 1.0,
    trajectory_interval_steps: int = 5_000,
    log_file: str | PathLike[str] | None = None,
    log_interval_steps: int = 5_000,
    checkpoint_interval_steps: int = 50_000,
    barostat_interval_steps: int = 25,
    forcefield_files: Sequence[str] = (
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ),
    nonbonded_cutoff_nm: float = 1.0,
    platform_name: str | None = None,
    platform_properties: Mapping[str, str] | None = None,
    random_seed: int | None = None,
    state_input_file: str | PathLike[str] | None = None,
    checkpoint_input_file: str | PathLike[str] | None = None,
    resume_from: ResumeMode = "auto",
    state_output_file: str | PathLike[str] | None = None,
    checkpoint_output_file: str | PathLike[str] | None = None,
    append: bool = False,
    keep_ids: bool = False,
    verbose: bool = True,
) -> ProductionResult:
    """Run fixed production MD from a PDB or preceding MD result.

    ``trajectory_file`` selects DCD or XTC through its suffix.  A supplied
    ``checkpoint_output_file`` is updated periodically and again after the
    final step.  ``steps`` is always the additional budget for this call.
    Existing trajectory and log files are extended only with ``append=True``.
    Result inputs automatically supply a compatible checkpoint or XML State.
    """
    input_path, output_path = validate_io_paths(input_file, output_file)
    trajectory_path, log_path, state_path, checkpoint_path = (
        _validate_output_paths(
            input_path,
            output_path,
            trajectory_file,
            log_file,
            state_output_file,
            checkpoint_output_file,
        )
    )

    normalized_ensemble = ensemble.upper()
    if normalized_ensemble not in {"NVT", "NPT"}:
        raise ValueError("ensemble must be 'NVT' or 'NPT'")
    selected_ensemble: Ensemble = normalized_ensemble  # type: ignore[assignment]
    for name, value in (
        ("temperature_k", temperature_k),
        ("timestep_fs", timestep_fs),
        ("friction_per_ps", friction_per_ps),
        ("nonbonded_cutoff_nm", nonbonded_cutoff_nm),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    for name, value in (
        ("steps", steps),
        ("trajectory_interval_steps", trajectory_interval_steps),
        ("log_interval_steps", log_interval_steps),
        ("checkpoint_interval_steps", checkpoint_interval_steps),
        ("barostat_interval_steps", barostat_interval_steps),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if selected_ensemble == "NPT" and pressure_bar <= 0:
        raise ValueError("pressure_bar must be greater than zero")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed must not be negative")
    if verbose:
        logger.info(
            "Running %d production steps from %s in the %s ensemble",
            steps,
            input_path,
            selected_ensemble,
        )
    pdb = PDBFile(str(input_path))
    periodic = pdb.topology.getPeriodicBoxVectors() is not None
    if selected_ensemble == "NPT" and not periodic:
        raise ValueError("NPT production requires periodic box vectors")

    forcefield = ForceField(*forcefield_files)
    system_options = {"constraints": HBonds}
    if periodic:
        system_options.update(
            {
                "nonbondedMethod": PME,
                "nonbondedCutoff": nonbonded_cutoff_nm * nanometer,
            }
        )
    else:
        system_options["nonbondedMethod"] = NoCutoff
    system = forcefield.createSystem(pdb.topology, **system_options)
    barostat = None
    if selected_ensemble == "NPT":
        barostat = MonteCarloBarostat(
            pressure_bar * bar,
            temperature_k * kelvin,
            barostat_interval_steps,
        )
        if random_seed is not None:
            barostat.setRandomNumberSeed(random_seed)
        system.addForce(barostat)

    integrator = LangevinMiddleIntegrator(
        temperature_k * kelvin,
        friction_per_ps / picosecond,
        timestep_fs * femtoseconds,
    )
    if random_seed is not None:
        integrator.setRandomNumberSeed(random_seed)
    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        **simulation_platform_options(platform_name, platform_properties),
    )
    config = simulation_config(
        simulation,
        system,
        integrator,
        ensemble=selected_ensemble,
        temperature_k=temperature_k,
        pressure_bar=(
            pressure_bar if selected_ensemble == "NPT" else None
        ),
        timestep_fs=timestep_fs,
        friction_per_ps=friction_per_ps,
        forcefield_files=tuple(forcefield_files),
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
        barostat_interval_steps=(
            barostat_interval_steps if selected_ensemble == "NPT" else None
        ),
    )
    state_input_path, checkpoint_input_path, resume_mode = (
        resolve_resume_input(
            input_file,
            state_input_file=state_input_file,
            checkpoint_input_file=checkpoint_input_file,
            resume_from=resume_from,
            target_config=config,
        )
    )
    if append:
        if resume_mode == "coordinates":
            raise ValueError(
                "append=True requires a State or checkpoint continuation"
            )
        for name, path in (
            ("trajectory_file", trajectory_path),
            ("log_file", log_path),
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(
                    f"{name} must exist when append=True: {path}"
                )
    if state_input_path is not None:
        try:
            simulation.loadState(str(state_input_path))
        except Exception as error:
            raise ValueError(
                f"Could not load OpenMM State: {state_input_path}"
            ) from error
        integrator.setTemperature(temperature_k * kelvin)
        integrator.setStepSize(timestep_fs * femtoseconds)
        if barostat is not None:
            simulation.context.setParameter(
                MonteCarloBarostat.Pressure(), pressure_bar
            )
            simulation.context.setParameter(
                MonteCarloBarostat.Temperature(), temperature_k
            )
    elif checkpoint_input_path is not None:
        try:
            simulation.loadCheckpoint(str(checkpoint_input_path))
        except Exception as error:
            raise ValueError(
                "Could not load checkpoint into the configured OpenMM "
                "System; checkpoints require the same compatible System "
                "and Integrator"
            ) from error
    else:
        simulation.context.setPositions(pdb.positions)
        if random_seed is None:
            simulation.context.setVelocitiesToTemperature(
                temperature_k * kelvin
            )
        else:
            simulation.context.setVelocitiesToTemperature(
                temperature_k * kelvin, random_seed
            )

    initial_step = simulation.currentStep
    initial_time_ps = simulation.context.getState().getTime().value_in_unit(
        picosecond
    )
    _add_reporters(
        simulation,
        trajectory_path=trajectory_path,
        trajectory_interval_steps=trajectory_interval_steps,
        log_path=log_path,
        log_interval_steps=log_interval_steps,
        checkpoint_path=checkpoint_path,
        checkpoint_interval_steps=checkpoint_interval_steps,
        periodic=periodic,
        append=append,
        total_steps=initial_step + steps,
    )
    if verbose:
        logger.info(
            "Using OpenMM platform %s",
            simulation.context.getPlatform().getName(),
        )
    simulation.step(steps)

    final_time_ps = simulation.context.getState().getTime().value_in_unit(
        picosecond
    )
    _write_equilibration_outputs(
        simulation,
        output_path=output_path,
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        periodic=periodic,
        keep_ids=keep_ids,
    )
    if verbose:
        logger.info(
            "Saved production trajectory %s and final PDB %s after %d steps",
            trajectory_path,
            output_path,
            steps,
        )
    return ProductionResult(
        output_path=output_path,
        trajectory_path=trajectory_path,
        ensemble=selected_ensemble,
        steps=steps,
        initial_step=initial_step,
        final_step=simulation.currentStep,
        elapsed_time_ps=float(final_time_ps - initial_time_ps),
        simulation_time_ps=float(final_time_ps),
        target_temperature_k=temperature_k,
        target_pressure_bar=(
            pressure_bar if selected_ensemble == "NPT" else None
        ),
        timestep_fs=timestep_fs,
        trajectory_interval_steps=trajectory_interval_steps,
        log_path=log_path,
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        log_interval_steps=(log_interval_steps if log_path else None),
        checkpoint_interval_steps=(
            checkpoint_interval_steps if checkpoint_path else None
        ),
        input_state_path=state_input_path,
        input_checkpoint_path=checkpoint_input_path,
        appended=append,
        simulation_config=config,
        resume_mode=resume_mode,
    )
