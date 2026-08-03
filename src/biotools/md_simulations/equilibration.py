"""Adaptive NVT and NPT simulation with inspectable diagnostics."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from openmm import CMMotionRemover, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, PME, Simulation
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    bar,
    dalton,
    femtoseconds,
    gram,
    item,
    kelvin,
    kilojoule_per_mole,
    milliliter,
    nanometer,
    picosecond,
)

from .common import simulation_platform_options, validate_io_paths

logger = logging.getLogger(__name__)

Ensemble: TypeAlias = Literal["NVT", "NPT"]


@dataclass(frozen=True)
class EquilibrationSample:
    """One sampled thermodynamic state during equilibration."""

    step: int
    time_ps: float
    temperature_k: float
    potential_energy_kj_mol: float
    kinetic_energy_kj_mol: float
    total_energy_kj_mol: float
    volume_nm3: float | None
    density_g_ml: float | None
    pressure_bar: float | None
    phase: str = "equilibration"
    target_temperature_k: float | None = None
    timestep_fs: float | None = None


@dataclass(frozen=True)
class EquilibrationCriteria:
    """Tolerances used by :class:`StabilityMonitor`.

    Drift limits are slopes over the most recent sample window.  Energy drift
    is normalized by the number of particles, while volume drift is normalized
    by the mean volume.
    """

    window_samples: int = 10
    required_stable_windows: int = 3
    temperature_tolerance_k: float = 10.0
    temperature_drift_tolerance_k_per_ps: float = 0.1
    energy_drift_tolerance_kj_mol_particle_ps: float = 0.01
    volume_drift_tolerance_fraction_per_ps: float = 5e-4
    volume_coefficient_of_variation_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least 2")
        if self.required_stable_windows < 1:
            raise ValueError("required_stable_windows must be at least 1")
        for name, value in (
            ("temperature_tolerance_k", self.temperature_tolerance_k),
            (
                "temperature_drift_tolerance_k_per_ps",
                self.temperature_drift_tolerance_k_per_ps,
            ),
            (
                "energy_drift_tolerance_kj_mol_particle_ps",
                self.energy_drift_tolerance_kj_mol_particle_ps,
            ),
            (
                "volume_drift_tolerance_fraction_per_ps",
                self.volume_drift_tolerance_fraction_per_ps,
            ),
            (
                "volume_coefficient_of_variation_tolerance",
                self.volume_coefficient_of_variation_tolerance,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class EquilibrationProgress:
    """Read-only data passed to an equilibration monitor callback."""

    ensemble: Ensemble
    target_temperature_k: float
    target_pressure_bar: float | None
    num_particles: int
    max_steps: int
    samples: tuple[EquilibrationSample, ...]
    initial_step: int = 0

    @property
    def current_step(self) -> int:
        """Step represented by the most recent sample."""
        return self.samples[-1].step if self.samples else 0

    @property
    def steps_this_run(self) -> int:
        """Number of steps completed since loading the optional input state."""
        return self.current_step - self.initial_step


@dataclass(frozen=True)
class EquilibrationAssessment:
    """A monitor decision plus the criteria and metrics behind it."""

    stop: bool
    successful: bool
    reason: str
    criteria: Mapping[str, bool] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)


class EquilibrationMonitor(Protocol):
    """Protocol for callbacks that decide when equilibration should stop."""

    def __call__(
        self, progress: EquilibrationProgress
    ) -> EquilibrationAssessment | bool | None:
        """Inspect progress and optionally request termination."""


MonitorCallback: TypeAlias = Callable[
    [EquilibrationProgress], EquilibrationAssessment | bool | None
]


@dataclass(frozen=True)
class EquilibrationResult:
    """Outcome and complete sampled history of an equilibration run."""

    output_path: Path
    ensemble: Ensemble
    successful: bool
    termination_reason: str
    steps: int
    max_steps: int
    elapsed_time_ps: float
    target_temperature_k: float
    target_pressure_bar: float | None
    assessment: EquilibrationAssessment
    samples: tuple[EquilibrationSample, ...]
    state_path: Path | None = None
    checkpoint_path: Path | None = None
    warmup_steps: int = 0
    initial_temperature_k: float | None = None
    initial_timestep_fs: float | None = None
    target_timestep_fs: float | None = None
    input_state_path: Path | None = None
    input_checkpoint_path: Path | None = None
    initial_step: int = 0
    final_step: int = 0

    @property
    def converged(self) -> bool:
        """Alias for ``successful``."""
        return self.successful

    @property
    def final_sample(self) -> EquilibrationSample:
        """Return the final sampled thermodynamic state."""
        return self.samples[-1]


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    """Return the least-squares slope, guarding against identical times."""
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator == 0.0:
        return math.inf
    centered_values = values - float(np.mean(values))
    return float(np.dot(centered_times, centered_values) / denominator)


class StabilityMonitor:
    """Stop after thermodynamic observables remain stable for several windows.

    NVT checks mean temperature, temperature drift, and potential-energy drift.
    NPT additionally checks relative volume drift and volume fluctuations.  It
    intentionally does not require instantaneous pressure to be close to the
    target because pressure has very large equilibrium fluctuations.
    """

    def __init__(self, criteria: EquilibrationCriteria | None = None) -> None:
        self.criteria = criteria or EquilibrationCriteria()
        self._stable_windows = 0

    def reset(self) -> None:
        """Reset consecutive-window state before a new simulation."""
        self._stable_windows = 0

    def __call__(
        self, progress: EquilibrationProgress
    ) -> EquilibrationAssessment:
        criteria = self.criteria
        if len(progress.samples) < criteria.window_samples:
            self._stable_windows = 0
            return EquilibrationAssessment(
                stop=False,
                successful=False,
                reason="collecting_samples",
                criteria={"enough_samples": False},
                metrics={"sample_count": float(len(progress.samples))},
            )

        window = progress.samples[-criteria.window_samples :]
        times = np.asarray([sample.time_ps for sample in window], dtype=float)
        temperatures = np.asarray(
            [sample.temperature_k for sample in window], dtype=float
        )
        energies = np.asarray(
            [sample.potential_energy_kj_mol for sample in window], dtype=float
        )
        mean_temperature = float(np.mean(temperatures))
        temperature_drift = _linear_slope(times, temperatures)
        energy_drift_per_particle = (
            _linear_slope(times, energies) / progress.num_particles
        )
        checks: dict[str, bool] = {
            "enough_samples": True,
            "temperature_mean": abs(
                mean_temperature - progress.target_temperature_k
            )
            <= criteria.temperature_tolerance_k,
            "temperature_drift": abs(temperature_drift)
            <= criteria.temperature_drift_tolerance_k_per_ps,
            "potential_energy_drift": abs(energy_drift_per_particle)
            <= criteria.energy_drift_tolerance_kj_mol_particle_ps,
        }
        metrics = {
            "mean_temperature_k": mean_temperature,
            "temperature_drift_k_per_ps": temperature_drift,
            "potential_energy_drift_kj_mol_particle_ps": (
                energy_drift_per_particle
            ),
        }

        if progress.ensemble == "NPT":
            volumes = np.asarray(
                [sample.volume_nm3 for sample in window], dtype=float
            )
            mean_volume = float(np.mean(volumes))
            relative_volume_drift = (
                _linear_slope(times, volumes) / mean_volume
            )
            volume_cv = float(np.std(volumes, ddof=1) / mean_volume)
            checks.update(
                {
                    "volume_drift": abs(relative_volume_drift)
                    <= criteria.volume_drift_tolerance_fraction_per_ps,
                    "volume_fluctuation": volume_cv
                    <= criteria.volume_coefficient_of_variation_tolerance,
                }
            )
            densities = np.asarray(
                [sample.density_g_ml for sample in window], dtype=float
            )
            metrics.update(
                {
                    "mean_volume_nm3": mean_volume,
                    "relative_volume_drift_per_ps": relative_volume_drift,
                    "volume_coefficient_of_variation": volume_cv,
                    "mean_density_g_ml": float(np.mean(densities)),
                }
            )
            pressures = np.asarray(
                [sample.pressure_bar for sample in window], dtype=float
            )
            metrics.update(
                {
                    "mean_pressure_bar": float(np.mean(pressures)),
                    "pressure_standard_deviation_bar": float(
                        np.std(pressures, ddof=1)
                    ),
                }
            )

        stable = all(checks.values())
        self._stable_windows = self._stable_windows + 1 if stable else 0
        checks["stable_windows"] = (
            self._stable_windows >= criteria.required_stable_windows
        )
        metrics["stable_window_count"] = float(self._stable_windows)
        successful = stable and checks["stable_windows"]
        return EquilibrationAssessment(
            stop=successful,
            successful=successful,
            reason="equilibrated" if successful else "criteria_not_met",
            criteria=checks,
            metrics=metrics,
        )


def _degrees_of_freedom(system) -> int:
    """Calculate temperature degrees of freedom as OpenMM's reporter does."""
    dof = 0
    for index in range(system.getNumParticles()):
        if system.getParticleMass(index) > 0 * dalton:
            dof += 3
    for index in range(system.getNumConstraints()):
        particle1, particle2, _ = system.getConstraintParameters(index)
        if (
            system.getParticleMass(particle1) > 0 * dalton
            or system.getParticleMass(particle2) > 0 * dalton
        ):
            dof -= 1
    if any(
        type(system.getForce(index)) is CMMotionRemover
        for index in range(system.getNumForces())
    ):
        dof -= 3
    if dof <= 0:
        raise RuntimeError("Cannot calculate temperature with zero degrees of freedom")
    return dof


def _total_mass(system):
    """Return the total system mass as an OpenMM quantity."""
    return sum(
        (system.getParticleMass(index) for index in range(system.getNumParticles())),
        0 * dalton,
    )


def _sample_state(
    simulation: Simulation,
    *,
    step: int,
    degrees_of_freedom: int,
    total_mass,
    periodic: bool,
    barostat: MonteCarloBarostat | None,
    phase: str = "equilibration",
    target_temperature_k: float | None = None,
    timestep_fs: float | None = None,
) -> EquilibrationSample:
    state = simulation.context.getState(getEnergy=True)
    potential_energy = float(
        state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    )
    kinetic_energy = float(
        state.getKineticEnergy().value_in_unit(kilojoule_per_mole)
    )
    temperature = float(
        (
            2 * state.getKineticEnergy()
            / (degrees_of_freedom * MOLAR_GAS_CONSTANT_R)
        ).value_in_unit(kelvin)
    )
    volume = None
    density = None
    pressure = None
    if periodic:
        volume_quantity = state.getPeriodicBoxVolume()
        volume = float(volume_quantity.value_in_unit(nanometer**3))
        density = float(
            (total_mass / volume_quantity).value_in_unit(
                gram / item / milliliter
            )
        )
    if barostat is not None:
        pressure = float(
            barostat.computeCurrentPressure(
                simulation.context
            ).value_in_unit(bar)
        )
    sample = EquilibrationSample(
        step=step,
        time_ps=float(state.getTime().value_in_unit(picosecond)),
        temperature_k=temperature,
        potential_energy_kj_mol=potential_energy,
        kinetic_energy_kj_mol=kinetic_energy,
        total_energy_kj_mol=potential_energy + kinetic_energy,
        volume_nm3=volume,
        density_g_ml=density,
        pressure_bar=pressure,
        phase=phase,
        target_temperature_k=target_temperature_k,
        timestep_fs=timestep_fs,
    )
    numeric_values = (
        sample.time_ps,
        sample.temperature_k,
        sample.potential_energy_kj_mol,
        sample.kinetic_energy_kj_mol,
        sample.total_energy_kj_mol,
        sample.volume_nm3,
        sample.density_g_ml,
        sample.pressure_bar,
    )
    if not all(
        value is None or math.isfinite(value) for value in numeric_values
    ):
        raise RuntimeError("Equilibration produced a non-finite observable")
    return sample


def _normalize_assessment(
    decision: EquilibrationAssessment | bool | None,
) -> EquilibrationAssessment:
    if isinstance(decision, EquilibrationAssessment):
        if decision.successful and not decision.stop:
            raise ValueError("A successful monitor assessment must request stop")
        return decision
    if decision is True:
        return EquilibrationAssessment(
            stop=True,
            successful=True,
            reason="callback_requested_stop",
        )
    return EquilibrationAssessment(
        stop=False,
        successful=False,
        reason="callback_continues",
    )


def _validate_auxiliary_output_paths(
    input_path: Path,
    output_path: Path,
    state_output_file: str | PathLike[str] | None,
    checkpoint_output_file: str | PathLike[str] | None,
) -> tuple[Path | None, Path | None]:
    """Validate optional state and checkpoint destinations."""
    state_path = Path(state_output_file) if state_output_file is not None else None
    checkpoint_path = (
        Path(checkpoint_output_file)
        if checkpoint_output_file is not None
        else None
    )
    destinations = [output_path]
    for name, path in (
        ("state_output_file", state_path),
        ("checkpoint_output_file", checkpoint_path),
    ):
        if path is None:
            continue
        if path.resolve() == input_path.resolve():
            raise ValueError(f"{name} must not overwrite input_file")
        if any(path.resolve() == destination.resolve() for destination in destinations):
            raise ValueError("PDB, state, and checkpoint outputs must be different")
        destinations.append(path)
    return state_path, checkpoint_path


def _validate_resume_input_paths(
    state_input_file: str | PathLike[str] | None,
    checkpoint_input_file: str | PathLike[str] | None,
) -> tuple[Path | None, Path | None]:
    """Validate mutually exclusive XML State and checkpoint inputs."""
    if state_input_file is not None and checkpoint_input_file is not None:
        raise ValueError(
            "state_input_file and checkpoint_input_file are mutually exclusive"
        )
    state_input_path = (
        Path(state_input_file) if state_input_file is not None else None
    )
    checkpoint_input_path = (
        Path(checkpoint_input_file)
        if checkpoint_input_file is not None
        else None
    )
    for name, path in (
        ("state_input_file", state_input_path),
        ("checkpoint_input_file", checkpoint_input_path),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    return state_input_path, checkpoint_input_path


def _write_equilibration_outputs(
    simulation: Simulation,
    *,
    output_path: Path,
    state_path: Path | None,
    checkpoint_path: Path | None,
    periodic: bool,
    keep_ids: bool,
) -> None:
    """Write a final PDB and optional portable state and exact checkpoint."""
    final_state = simulation.context.getState(getPositions=True)
    if periodic:
        simulation.topology.setPeriodicBoxVectors(
            final_state.getPeriodicBoxVectors()
        )
    with output_path.open("w") as output:
        PDBFile.writeFile(
            simulation.topology,
            final_state.getPositions(),
            output,
            keepIds=keep_ids,
        )
    if state_path is not None:
        simulation.saveState(str(state_path))
    if checkpoint_path is not None:
        simulation.saveCheckpoint(str(checkpoint_path))


def equilibrate(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    ensemble: str = "NPT",
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    timestep_fs: float = 2.0,
    friction_per_ps: float = 1.0,
    max_steps: int = 1_000_000,
    check_interval_steps: int = 5_000,
    barostat_interval_steps: int = 25,
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
    state_input_file: str | PathLike[str] | None = None,
    checkpoint_input_file: str | PathLike[str] | None = None,
    state_output_file: str | PathLike[str] | None = None,
    checkpoint_output_file: str | PathLike[str] | None = None,
    keep_ids: bool = False,
    verbose: bool = True,
) -> EquilibrationResult:
    """Equilibrate a minimized PDB adaptively in the NVT or NPT ensemble.

    Dynamics run in blocks.  After each block, ``monitor`` receives all sampled
    thermodynamic states and may stop the run.  Without a custom monitor,
    :class:`StabilityMonitor` checks temperature and energy stability, plus
    volume stability for NPT.  The run always ends no later than ``max_steps``.
    """
    input_path, output_path = validate_io_paths(input_file, output_file)
    state_input_path, checkpoint_input_path = _validate_resume_input_paths(
        state_input_file,
        checkpoint_input_file,
    )
    state_path, checkpoint_path = _validate_auxiliary_output_paths(
        input_path,
        output_path,
        state_output_file,
        checkpoint_output_file,
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
    if selected_ensemble == "NPT" and pressure_bar <= 0:
        raise ValueError("pressure_bar must be greater than zero")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")
    if check_interval_steps <= 0:
        raise ValueError("check_interval_steps must be greater than zero")
    if barostat_interval_steps <= 0:
        raise ValueError("barostat_interval_steps must be greater than zero")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed must not be negative")
    simulation_options = simulation_platform_options(
        platform_name, platform_properties
    )
    if monitor is not None and criteria is not None:
        raise ValueError("criteria cannot be combined with a custom monitor")

    if verbose:
        logger.info(
            "Equilibrating PDB file %s in the %s ensemble",
            input_path,
            selected_ensemble,
        )
    pdb = PDBFile(str(input_path))
    periodic = pdb.topology.getPeriodicBoxVectors() is not None
    if selected_ensemble == "NPT" and not periodic:
        raise ValueError("NPT equilibration requires periodic box vectors")

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
        pdb.topology, system, integrator, **simulation_options
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
                "Could not load checkpoint into the configured OpenMM System; "
                "checkpoints require the same compatible System and Integrator"
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
    if verbose:
        logger.info(
            "Using OpenMM platform %s",
            simulation.context.getPlatform().getName(),
        )

    callback: MonitorCallback
    if monitor is None:
        callback = StabilityMonitor(criteria)
    else:
        callback = monitor
        reset = getattr(callback, "reset", None)
        if callable(reset):
            reset()

    num_particles = system.getNumParticles()
    degrees_of_freedom = _degrees_of_freedom(system)
    total_mass = _total_mass(system)
    samples: list[EquilibrationSample] = []
    initial_step = simulation.currentStep
    executed_steps = 0
    assessment = EquilibrationAssessment(
        stop=False,
        successful=False,
        reason="not_started",
    )
    while executed_steps < max_steps:
        block_steps = min(check_interval_steps, max_steps - executed_steps)
        simulation.step(block_steps)
        executed_steps += block_steps
        samples.append(
            _sample_state(
                simulation,
                step=simulation.currentStep,
                degrees_of_freedom=degrees_of_freedom,
                total_mass=total_mass,
                periodic=periodic,
                barostat=barostat,
                target_temperature_k=temperature_k,
                timestep_fs=timestep_fs,
            )
        )
        progress = EquilibrationProgress(
            ensemble=selected_ensemble,
            target_temperature_k=temperature_k,
            target_pressure_bar=(
                pressure_bar if selected_ensemble == "NPT" else None
            ),
            num_particles=num_particles,
            max_steps=max_steps,
            samples=tuple(samples),
            initial_step=initial_step,
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
            "Saved equilibrated PDB file to %s after %d steps (%s)",
            output_path,
            executed_steps,
            termination_reason,
        )
    return EquilibrationResult(
        output_path=output_path,
        ensemble=selected_ensemble,
        successful=successful,
        termination_reason=termination_reason,
        steps=executed_steps,
        max_steps=max_steps,
        elapsed_time_ps=samples[-1].time_ps,
        target_temperature_k=temperature_k,
        target_pressure_bar=(
            pressure_bar if selected_ensemble == "NPT" else None
        ),
        assessment=assessment,
        samples=tuple(samples),
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        initial_temperature_k=temperature_k,
        initial_timestep_fs=timestep_fs,
        target_timestep_fs=timestep_fs,
        input_state_path=state_input_path,
        input_checkpoint_path=checkpoint_input_path,
        initial_step=initial_step,
        final_step=simulation.currentStep,
    )
