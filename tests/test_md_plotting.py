"""Tests for molecular-dynamics result plotting."""

import os
from pathlib import Path
from tempfile import gettempdir
import unittest

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(gettempdir()) / "biotools-matplotlib-tests"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biotools.mdtools import (
    EquilibrationAssessment,
    EquilibrationResult,
    EquilibrationSample,
    MinimizationResult,
    MinimizationSample,
    plot_md_result,
)


class PlotMDResultTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_plots_minimization_energy_and_gradient_history(self) -> None:
        result = MinimizationResult(
            output_path=Path("minimized.pdb"),
            initial_energy_kj_mol=100.0,
            final_energy_kj_mol=25.0,
            delta_energy_kj_mol=-75.0,
            initial_raw_rms_force_kj_mol_nm=30.0,
            final_raw_rms_force_kj_mol_nm=5.0,
            final_raw_max_force_kj_mol_nm=8.0,
            final_objective_rms_gradient_kj_mol_nm=4.0,
            final_max_constraint_error=5e-5,
            iterations=2,
            optimizer_restarts=0,
            tolerance_kj_mol_nm=10.0,
            constraint_tolerance=1e-4,
            max_iterations=1000,
            converged=True,
            termination_reason="converged",
            samples=(
                MinimizationSample(
                    iteration=1,
                    optimizer_attempt=0,
                    energy_kj_mol=50.0,
                    objective_rms_gradient_kj_mol_nm=20.0,
                    restraint_energy_kj_mol=0.0,
                    restraint_strength_kj_mol_nm2=0.0,
                    max_constraint_error=5e-5,
                ),
                MinimizationSample(
                    iteration=2,
                    optimizer_attempt=0,
                    energy_kj_mol=25.0,
                    objective_rms_gradient_kj_mol_nm=4.0,
                    restraint_energy_kj_mol=0.0,
                    restraint_strength_kj_mol_nm2=0.0,
                    max_constraint_error=5e-5,
                ),
            ),
        )

        figure, axes = plot_md_result(result)

        self.assertEqual(len(axes), 2)
        self.assertEqual(list(axes[0].lines[0].get_xdata()), [0, 1, 2])
        self.assertEqual(
            list(axes[0].lines[0].get_ydata()),
            [100.0, 50.0, 25.0],
        )
        self.assertEqual(list(axes[1].lines[0].get_xdata()), [1, 2])
        self.assertEqual(
            list(axes[1].lines[0].get_ydata()),
            [20.0, 4.0],
        )
        self.assertEqual(figure._suptitle.get_text(), "Energy minimization")

    def test_plots_equilibration_temperature_energy_and_pressure(self) -> None:
        samples = (
            EquilibrationSample(
                step=100,
                time_ps=0.2,
                temperature_k=280.0,
                potential_energy_kj_mol=-100.0,
                kinetic_energy_kj_mol=40.0,
                total_energy_kj_mol=-60.0,
                volume_nm3=10.0,
                density_g_ml=1.0,
                pressure_bar=1.5,
                target_temperature_k=290.0,
            ),
            EquilibrationSample(
                step=200,
                time_ps=0.4,
                temperature_k=300.0,
                potential_energy_kj_mol=-110.0,
                kinetic_energy_kj_mol=45.0,
                total_energy_kj_mol=-65.0,
                volume_nm3=10.1,
                density_g_ml=0.99,
                pressure_bar=0.8,
                target_temperature_k=300.0,
            ),
        )
        result = EquilibrationResult(
            output_path=Path("equilibrated.pdb"),
            ensemble="NPT",
            successful=True,
            termination_reason="stable",
            steps=200,
            max_steps=1000,
            elapsed_time_ps=0.4,
            target_temperature_k=300.0,
            target_pressure_bar=1.0,
            assessment=EquilibrationAssessment(
                stop=True,
                successful=True,
                reason="stable",
            ),
            samples=samples,
        )

        figure, axes = plot_md_result(result)

        self.assertEqual(len(axes), 3)
        self.assertEqual(
            list(axes[0].lines[0].get_ydata()),
            [280.0, 300.0],
        )
        self.assertEqual(
            list(axes[0].lines[1].get_ydata()),
            [290.0, 300.0],
        )
        self.assertEqual(len(axes[1].lines), 3)
        self.assertEqual(
            list(axes[2].lines[0].get_ydata()),
            [1.5, 0.8],
        )
        self.assertEqual(figure._suptitle.get_text(), "NPT equilibration")


if __name__ == "__main__":
    unittest.main()
