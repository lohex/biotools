"""Gentle NVT heating followed by adaptive equilibration."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from os import PathLike

from openmm import LangevinMiddleIntegrator
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, PME, Simulation
from openmm.unit import femtoseconds, kelvin, nanometer, picosecond

from .common import simulation_platform_options, validate_io_paths
from .equilibration import (
    EquilibrationAssessment,
    EquilibrationCriteria,
    EquilibrationProgress,
    EquilibrationResult,
    EquilibrationSample,
    MonitorCallback,
    StabilityMonitor,
    _degrees_of_freedom,
    _normalize_assessment,
    _sample_state,
    _total_mass,
    _validate_auxiliary_output_paths,
    _write_equilibration_outputs,
)

logger = logging.getLogger(__name__)


def _stage_step_counts(total_steps: int, stages: int) -> tuple[int, ...]:
    """Distribute a step budget exactly across heating stages."""
    base_steps, remainder = divmod(total_steps, stages)
    return tuple(
        base_steps + (1 if index < remainder else 0)
        for index in range(stages)
    )


def soft_equilibrate_nvt(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    initial_temperature_k: float = 50.0,
    temperature_k: float = 300.0,
    initial_timestep_fs: float = 0.5,
    timestep_fs: float = 2.0,
    heating_steps: int = 50_000,
    heating_stages: int = 10,
    max_steps: int = 1_000_000,
    check_interval_steps: int = 5_000,
    friction_per_ps: float = 1.0,
    criteria: EquilibrationCriteria | None = None,
    monitor: MonitorCallback | None = None,
    forcefield_files: Sequence[str] = (
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ),
    nonbonded_cutoff_nm: float = 1.0,
    platform_name: str | None = None,
    platform_properties: Mapping[str, str] | None = None,
    random_seed: int | None = None,
    state_output_file: str | PathLike[str] | None = None,
    checkpoint_output_file: str | PathLike[str] | None = None,
    keep_ids: bool = False,
    verbose: bool = True,
) -> EquilibrationResult:
    """Gently heat a minimized structure, then adaptively equilibrate in NVT.

    Temperature and timestep are linearly increased over ``heating_stages`` in
    one OpenMM Context.  The first stage uses the initial values and the last
    stage uses the target values.  After heating, the usual stability monitor
    controls termination, with ``max_steps`` acting as the total hard limit.
    """
    input_path, output_path = validate_io_paths(input_file, output_file)
    state_path, checkpoint_path = _validate_auxiliary_output_paths(
        input_path,
        output_path,
        state_output_file,
        checkpoint_output_file,
    )
    for name, value in (
        ("initial_temperature_k", initial_temperature_k),
        ("temperature_k", temperature_k),
        ("initial_timestep_fs", initial_timestep_fs),
        ("timestep_fs", timestep_fs),
        ("friction_per_ps", friction_per_ps),
        ("nonbonded_cutoff_nm", nonbonded_cutoff_nm),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if initial_temperature_k >= temperature_k:
        raise ValueError(
            "initial_temperature_k must be less than temperature_k"
        )
    if initial_timestep_fs >= timestep_fs:
        raise ValueError("initial_timestep_fs must be less than timestep_fs")
    if heating_steps <= 0:
        raise ValueError("heating_steps must be greater than zero")
    if heating_stages < 2:
        raise ValueError("heating_stages must be at least 2")
    if heating_stages > heating_steps:
        raise ValueError("heating_stages must not exceed heating_steps")
    if max_steps <= heating_steps:
        raise ValueError("max_steps must be greater than heating_steps")
    if check_interval_steps <= 0:
        raise ValueError("check_interval_steps must be greater than zero")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed must not be negative")
    if monitor is not None and criteria is not None:
        raise ValueError("criteria cannot be combined with a custom monitor")
    simulation_options = simulation_platform_options(
        platform_name, platform_properties
    )

    if verbose:
        logger.info(
            "Soft-equilibrating PDB file %s from %.1f K to %.1f K",
            input_path,
            initial_temperature_k,
            temperature_k,
        )
    pdb = PDBFile(str(input_path))
    periodic = pdb.topology.getPeriodicBoxVectors() is not None
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
    integrator = LangevinMiddleIntegrator(
        initial_temperature_k * kelvin,
        friction_per_ps / picosecond,
        initial_timestep_fs * femtoseconds,
    )
    if random_seed is not None:
        integrator.setRandomNumberSeed(random_seed)
    simulation = Simulation(
        pdb.topology, system, integrator, **simulation_options
    )
    simulation.context.setPositions(pdb.positions)
    if random_seed is None:
        simulation.context.setVelocitiesToTemperature(
            initial_temperature_k * kelvin
        )
    else:
        simulation.context.setVelocitiesToTemperature(
            initial_temperature_k * kelvin, random_seed
        )
    if verbose:
        logger.info(
            "Using OpenMM platform %s",
            simulation.context.getPlatform().getName(),
        )

    num_particles = system.getNumParticles()
    degrees_of_freedom = _degrees_of_freedom(system)
    total_mass = _total_mass(system)
    samples: list[EquilibrationSample] = []
    executed_steps = 0
    stage_steps = _stage_step_counts(heating_steps, heating_stages)
    for stage_index, steps in enumerate(stage_steps):
        fraction = stage_index / (heating_stages - 1)
        stage_temperature = initial_temperature_k + fraction * (
            temperature_k - initial_temperature_k
        )
        stage_timestep = initial_timestep_fs + fraction * (
            timestep_fs - initial_timestep_fs
        )
        integrator.setTemperature(stage_temperature * kelvin)
        integrator.setStepSize(stage_timestep * femtoseconds)
        simulation.step(steps)
        executed_steps += steps
        samples.append(
            _sample_state(
                simulation,
                step=executed_steps,
                degrees_of_freedom=degrees_of_freedom,
                total_mass=total_mass,
                periodic=periodic,
                barostat=None,
                phase="heating",
                target_temperature_k=stage_temperature,
                timestep_fs=stage_timestep,
            )
        )

    callback: MonitorCallback
    if monitor is None:
        callback = StabilityMonitor(criteria)
    else:
        callback = monitor
        reset = getattr(callback, "reset", None)
        if callable(reset):
            reset()

    equilibration_samples: list[EquilibrationSample] = []
    assessment = EquilibrationAssessment(
        stop=False,
        successful=False,
        reason="heating_complete",
    )
    while executed_steps < max_steps:
        block_steps = min(check_interval_steps, max_steps - executed_steps)
        simulation.step(block_steps)
        executed_steps += block_steps
        sample = _sample_state(
            simulation,
            step=executed_steps,
            degrees_of_freedom=degrees_of_freedom,
            total_mass=total_mass,
            periodic=periodic,
            barostat=None,
            phase="equilibration",
            target_temperature_k=temperature_k,
            timestep_fs=timestep_fs,
        )
        samples.append(sample)
        equilibration_samples.append(sample)
        progress = EquilibrationProgress(
            ensemble="NVT",
            target_temperature_k=temperature_k,
            target_pressure_bar=None,
            num_particles=num_particles,
            max_steps=max_steps,
            samples=tuple(equilibration_samples),
        )
        assessment = _normalize_assessment(callback(progress))
        if assessment.stop:
            break

    successful = assessment.stop and assessment.successful
    termination_reason = (
        assessment.reason if assessment.stop else "max_steps"
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
            "Saved softly equilibrated PDB file to %s after %d steps (%s)",
            output_path,
            executed_steps,
            termination_reason,
        )
    return EquilibrationResult(
        output_path=output_path,
        ensemble="NVT",
        successful=successful,
        termination_reason=termination_reason,
        steps=executed_steps,
        max_steps=max_steps,
        elapsed_time_ps=samples[-1].time_ps,
        target_temperature_k=temperature_k,
        target_pressure_bar=None,
        assessment=assessment,
        samples=tuple(samples),
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        warmup_steps=heating_steps,
        initial_temperature_k=initial_temperature_k,
        initial_timestep_fs=initial_timestep_fs,
        target_timestep_fs=timestep_fs,
        initial_step=0,
        final_step=simulation.currentStep,
    )
