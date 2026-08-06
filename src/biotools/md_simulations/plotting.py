"""Plot diagnostics returned by molecular-dynamics workflows."""

from __future__ import annotations

from typing import Any

from .equilibration import EquilibrationResult
from .minimization import MinimizationResult


def _plot_minimization_result(
    result: MinimizationResult,
    plt: Any,
    figsize: tuple[float, float] | None,
) -> tuple[Any, tuple[Any, ...]]:
    figure, axes_array = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize or (9.0, 7.0),
    )
    energy_axis, gradient_axis = axes_array

    iterations = [0, *(sample.iteration for sample in result.samples)]
    energies = [
        result.initial_energy_kj_mol,
        *(sample.energy_kj_mol for sample in result.samples),
    ]
    energy_axis.plot(iterations, energies, label="Potential energy")
    energy_axis.scatter(
        [result.iterations],
        [result.final_energy_kj_mol],
        marker="x",
        color="black",
        label="Final energy",
        zorder=3,
    )
    energy_axis.set_ylabel("Energy [kJ/mol]")
    energy_axis.legend()
    energy_axis.grid(alpha=0.3)

    if result.samples:
        gradient_axis.plot(
            [sample.iteration for sample in result.samples],
            [
                sample.objective_rms_gradient_kj_mol_nm
                for sample in result.samples
            ],
            label="Objective RMS gradient",
        )
    else:
        gradient_axis.text(
            0.5,
            0.5,
            "No optimizer iterations recorded",
            ha="center",
            va="center",
            transform=gradient_axis.transAxes,
        )
    gradient_axis.axhline(
        result.tolerance_kj_mol_nm,
        color="black",
        linestyle="--",
        label="Tolerance",
    )
    gradient_axis.set_xlabel("Minimizer iteration")
    gradient_axis.set_ylabel("RMS gradient [kJ/(mol nm)]")
    gradient_axis.legend()
    gradient_axis.grid(alpha=0.3)

    figure.suptitle("Energy minimization")
    figure.tight_layout()
    return figure, (energy_axis, gradient_axis)


def _plot_equilibration_result(
    result: EquilibrationResult,
    plt: Any,
    figsize: tuple[float, float] | None,
) -> tuple[Any, tuple[Any, ...]]:
    if not result.samples:
        raise ValueError("Cannot plot an equilibration result without samples")

    pressure_samples = tuple(
        sample for sample in result.samples if sample.pressure_bar is not None
    )
    panel_count = 3 if pressure_samples else 2
    figure, axes_array = plt.subplots(
        panel_count,
        1,
        sharex=True,
        figsize=figsize or (9.0, 3.0 * panel_count),
    )
    axes = tuple(axes_array)
    temperature_axis, energy_axis = axes[:2]
    times = [sample.time_ps for sample in result.samples]

    temperature_axis.plot(
        times,
        [sample.temperature_k for sample in result.samples],
        label="Temperature",
    )
    temperature_axis.plot(
        times,
        [
            sample.target_temperature_k
            if sample.target_temperature_k is not None
            else result.target_temperature_k
            for sample in result.samples
        ],
        color="black",
        linestyle="--",
        label="Target temperature",
    )
    temperature_axis.set_ylabel("Temperature [K]")
    temperature_axis.legend()
    temperature_axis.grid(alpha=0.3)

    energy_axis.plot(
        times,
        [sample.potential_energy_kj_mol for sample in result.samples],
        label="Potential",
    )
    energy_axis.plot(
        times,
        [sample.kinetic_energy_kj_mol for sample in result.samples],
        label="Kinetic",
    )
    energy_axis.plot(
        times,
        [sample.total_energy_kj_mol for sample in result.samples],
        label="Total",
    )
    energy_axis.set_ylabel("Energy [kJ/mol]")
    energy_axis.legend()
    energy_axis.grid(alpha=0.3)

    if pressure_samples:
        pressure_axis = axes[2]
        pressure_axis.plot(
            [sample.time_ps for sample in pressure_samples],
            [sample.pressure_bar for sample in pressure_samples],
            label="Pressure",
        )
        if result.target_pressure_bar is not None:
            pressure_axis.axhline(
                result.target_pressure_bar,
                color="black",
                linestyle="--",
                label="Target pressure",
            )
        pressure_axis.set_ylabel("Pressure [bar]")
        pressure_axis.legend()
        pressure_axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Time [ps]")
    figure.suptitle(f"{result.ensemble} equilibration")
    figure.tight_layout()
    return figure, axes


def plot_md_result(
    result: MinimizationResult | EquilibrationResult,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    """Plot diagnostics from a minimization or equilibration result.

    The function does not call :func:`matplotlib.pyplot.show`; callers can
    display or save the returned figure as appropriate for their environment.

    Args:
        result: Result returned by ``minimize(..., return_diagnostics=True)``,
            :func:`equilibrate`, or :func:`soft_equilibrate_nvt`.
        figsize: Optional Matplotlib figure size in inches.

    Returns:
        A ``(figure, axes)`` tuple. Minimizations have energy and objective-RMS
        gradient axes. Equilibrations have temperature and energy axes, plus a
        pressure axis when pressure samples are available.

    Raises:
        TypeError: If ``result`` is not a supported result object.
        ImportError: If Matplotlib is not installed.
        ValueError: If an equilibration result contains no samples.
    """
    if not isinstance(result, (MinimizationResult, EquilibrationResult)):
        raise TypeError(
            "result must be a MinimizationResult or EquilibrationResult"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_md_result() requires Matplotlib; install the 'md' extra"
        ) from exc

    if isinstance(result, MinimizationResult):
        return _plot_minimization_result(result, plt, figsize)
    return _plot_equilibration_result(result, plt, figsize)
