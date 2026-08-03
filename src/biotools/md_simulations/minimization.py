"""OpenMM simulation minimization and convergence diagnostics."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from openmm import MinimizationReporter, VerletIntegrator
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, PME, Simulation
from openmm.unit import kilojoule_per_mole, nanometer, picoseconds

from .common import simulation_platform_options, validate_io_paths

logger = logging.getLogger(__name__)

FORCE_UNIT = kilojoule_per_mole / nanometer
_MINIMIZER_CONSTRAINT_TOLERANCE_FLOOR = 1e-4


@dataclass(frozen=True)
class MinimizationResult:
    """Summary of an OpenMM energy minimization."""

    output_path: Path
    initial_energy_kj_mol: float
    final_energy_kj_mol: float
    delta_energy_kj_mol: float
    initial_raw_rms_force_kj_mol_nm: float
    final_raw_rms_force_kj_mol_nm: float
    final_raw_max_force_kj_mol_nm: float
    final_objective_rms_gradient_kj_mol_nm: float | None
    final_max_constraint_error: float | None
    iterations: int
    optimizer_restarts: int
    tolerance_kj_mol_nm: float
    constraint_tolerance: float
    max_iterations: int
    converged: bool
    termination_reason: str

    @property
    def initial_rms_force_kj_mol_nm(self) -> float:
        """Alias for the pre-minimization raw RMS force."""
        return self.initial_raw_rms_force_kj_mol_nm

    @property
    def final_rms_force_kj_mol_nm(self) -> float:
        """Alias for the post-minimization raw RMS force."""
        return self.final_raw_rms_force_kj_mol_nm

    @property
    def final_max_force_kj_mol_nm(self) -> float:
        """Alias for the post-minimization raw maximum force."""
        return self.final_raw_max_force_kj_mol_nm


class _IterationReporter(MinimizationReporter):
    """Collect one entry for every OpenMM minimizer callback."""

    def __init__(
        self,
        num_particles: int,
        *,
        tolerance_kj_mol_nm: float | None = None,
        constraint_tolerance: float | None = None,
    ) -> None:
        super().__init__()
        if num_particles <= 0:
            raise ValueError("num_particles must be greater than zero")
        self.num_particles = num_particles
        self.tolerance_kj_mol_nm = tolerance_kj_mol_nm
        self.constraint_tolerance = constraint_tolerance
        self.convergence_requested = False
        self.history: list[dict[str, float | int]] = []

    def report(self, iteration, positions, gradient, args) -> bool:
        gradient_array = np.asarray(gradient, dtype=float)
        objective_rms_gradient = float(
            np.sqrt(np.sum(gradient_array**2) / self.num_particles)
        )
        self.history.append(
            {
                "iteration": int(iteration),
                "objective_rms_gradient_kj_mol_nm": objective_rms_gradient,
                "energy_kj_mol": float(args["system energy"]),
                "restraint_energy_kj_mol": float(args["restraint energy"]),
                "restraint_strength_kj_mol_nm2": float(
                    args["restraint strength"]
                ),
                "max_constraint_error": float(args["max constraint error"]),
            }
        )
        self.convergence_requested = bool(
            self.tolerance_kj_mol_nm is not None
            and self.constraint_tolerance is not None
            and objective_rms_gradient <= self.tolerance_kj_mol_nm
            and args["max constraint error"] <= self.constraint_tolerance
        )
        return self.convergence_requested


def _classify_minimization_termination(
    reporter: _IterationReporter,
    *,
    tolerance_kj_mol_nm: float,
    constraint_tolerance: float,
    max_iterations: int,
) -> tuple[bool, str]:
    """Infer convergence and termination reason from reporter callbacks."""
    if not reporter.history:
        return True, "already_converged"

    final = reporter.history[-1]
    gradient_converged = (
        final["objective_rms_gradient_kj_mol_nm"] <= tolerance_kj_mol_nm
    )
    constraints_satisfied = (
        final["max_constraint_error"] <= constraint_tolerance
    )
    if reporter.convergence_requested or (
        gradient_converged and constraints_satisfied
    ):
        return True, "converged"
    if max_iterations > 0 and final["iteration"] + 1 >= max_iterations:
        return False, "max_iterations"
    if not constraints_satisfied:
        return False, "constraint_error"
    return False, "optimizer_stopped"


def _should_restart_optimizer(
    reporter: _IterationReporter,
    *,
    termination_reason: str,
    tolerance_kj_mol_nm: float,
    constraint_tolerance: float,
    optimizer_restarts: int,
    max_optimizer_restarts: int,
) -> bool:
    """Return whether another L-BFGS attempt should use the current Context."""
    if (
        termination_reason != "optimizer_stopped"
        or optimizer_restarts >= max_optimizer_restarts
        or not reporter.history
    ):
        return False
    final_report = reporter.history[-1]
    return bool(
        final_report["max_constraint_error"] <= constraint_tolerance
        and final_report["objective_rms_gradient_kj_mol_nm"]
        > tolerance_kj_mol_nm
    )


def _get_raw_state_diagnostics(context) -> tuple[float, float, float]:
    """Return energy and unprojected Context force statistics."""
    state = context.getState(getEnergy=True, getForces=True)
    energy_kj_mol = state.getPotentialEnergy().value_in_unit(
        kilojoule_per_mole
    )
    forces = state.getForces(asNumpy=True).value_in_unit(FORCE_UNIT)
    rms_force = float(np.sqrt(np.mean(forces**2)))
    max_force = float(np.linalg.norm(forces, axis=1).max())
    return float(energy_kj_mol), rms_force, max_force


def minimize(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *,
    forcefield_files: Sequence[str] = (
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ),
    tolerance_kj_mol_nm: float = 10.0,
    max_iterations: int = 1000,
    max_optimizer_restarts: int = 0,
    nonbonded_cutoff_nm: float = 1.0,
    platform_name: str | None = None,
    platform_properties: Mapping[str, str] | None = None,
    keep_ids: bool = False,
    verbose: bool = True,
    return_diagnostics: bool = False,
) -> Path | MinimizationResult:
    """Energy-minimize a PDB structure with an OpenMM force field."""
    input_path, output_path = validate_io_paths(input_file, output_file)
    if tolerance_kj_mol_nm <= 0:
        raise ValueError("tolerance_kj_mol_nm must be greater than zero")
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")
    if max_optimizer_restarts < 0:
        raise ValueError("max_optimizer_restarts must not be negative")
    if nonbonded_cutoff_nm <= 0:
        raise ValueError("nonbonded_cutoff_nm must be greater than zero")
    simulation_options = simulation_platform_options(
        platform_name, platform_properties
    )

    if verbose:
        logger.info("Energy-minimizing PDB file %s", input_path)
    pdb = PDBFile(str(input_path))
    forcefield = ForceField(*forcefield_files)
    system_options = {"constraints": HBonds}
    if pdb.topology.getPeriodicBoxVectors() is None:
        system_options["nonbondedMethod"] = NoCutoff
    else:
        system_options["nonbondedMethod"] = PME
        system_options["nonbondedCutoff"] = nonbonded_cutoff_nm * nanometer

    system = forcefield.createSystem(pdb.topology, **system_options)
    integrator = VerletIntegrator(0.001 * picoseconds)
    simulation = Simulation(
        pdb.topology, system, integrator, **simulation_options
    )
    if verbose:
        logger.info(
            "Using OpenMM platform %s",
            simulation.context.getPlatform().getName(),
        )
    simulation.context.setPositions(pdb.positions)
    initial_energy, initial_raw_rms_force, _ = _get_raw_state_diagnostics(
        simulation.context
    )

    constraint_tolerance = max(
        _MINIMIZER_CONSTRAINT_TOLERANCE_FLOOR,
        float(integrator.getConstraintTolerance()),
    )
    total_iterations = 0
    optimizer_restarts = 0
    while True:
        reporter = _IterationReporter(
            system.getNumParticles(),
            tolerance_kj_mol_nm=tolerance_kj_mol_nm,
            constraint_tolerance=constraint_tolerance,
        )
        simulation.minimizeEnergy(
            tolerance=tolerance_kj_mol_nm * kilojoule_per_mole / nanometer,
            maxIterations=max_iterations,
            reporter=reporter,
        )
        total_iterations += len(reporter.history)
        converged, termination_reason = _classify_minimization_termination(
            reporter,
            tolerance_kj_mol_nm=tolerance_kj_mol_nm,
            constraint_tolerance=constraint_tolerance,
            max_iterations=max_iterations,
        )
        final_report = reporter.history[-1] if reporter.history else None
        if not _should_restart_optimizer(
            reporter,
            termination_reason=termination_reason,
            tolerance_kj_mol_nm=tolerance_kj_mol_nm,
            constraint_tolerance=constraint_tolerance,
            optimizer_restarts=optimizer_restarts,
            max_optimizer_restarts=max_optimizer_restarts,
        ):
            break
        optimizer_restarts += 1
        if verbose:
            logger.info(
                "Restarting OpenMM L-BFGS optimizer (%d/%d)",
                optimizer_restarts,
                max_optimizer_restarts,
            )

    final_energy, final_raw_rms_force, final_raw_max_force = (
        _get_raw_state_diagnostics(simulation.context)
    )
    final_objective_rms_gradient = (
        float(final_report["objective_rms_gradient_kj_mol_nm"])
        if final_report is not None
        else None
    )
    final_max_constraint_error = (
        float(final_report["max_constraint_error"])
        if final_report is not None
        else None
    )
    diagnostic_values = (
        initial_energy,
        final_energy,
        initial_raw_rms_force,
        final_raw_rms_force,
        final_raw_max_force,
    )
    reporter_values = (
        final_objective_rms_gradient,
        final_max_constraint_error,
    )
    if not all(math.isfinite(value) for value in diagnostic_values):
        raise RuntimeError("Minimization produced a non-finite energy or force")
    if not all(
        value is None or math.isfinite(value) for value in reporter_values
    ):
        raise RuntimeError(
            "Minimization produced a non-finite gradient or constraint error"
        )

    positions = simulation.context.getState(positions=True).getPositions()
    with output_path.open("w") as output:
        PDBFile.writeFile(
            simulation.topology, positions, output, keepIds=keep_ids
        )

    if verbose:
        logger.info("Saved minimized PDB file to %s", output_path)
    if not return_diagnostics:
        return output_path
    return MinimizationResult(
        output_path=output_path,
        initial_energy_kj_mol=initial_energy,
        final_energy_kj_mol=final_energy,
        delta_energy_kj_mol=final_energy - initial_energy,
        initial_raw_rms_force_kj_mol_nm=initial_raw_rms_force,
        final_raw_rms_force_kj_mol_nm=final_raw_rms_force,
        final_raw_max_force_kj_mol_nm=final_raw_max_force,
        final_objective_rms_gradient_kj_mol_nm=final_objective_rms_gradient,
        final_max_constraint_error=final_max_constraint_error,
        iterations=total_iterations,
        optimizer_restarts=optimizer_restarts,
        tolerance_kj_mol_nm=tolerance_kj_mol_nm,
        constraint_tolerance=constraint_tolerance,
        max_iterations=max_iterations,
        converged=converged,
        termination_reason=termination_reason,
    )
