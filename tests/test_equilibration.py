"""Tests for adaptive molecular-dynamics equilibration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from biotools.mdtools import (
    EquilibrationAssessment,
    EquilibrationCriteria,
    EquilibrationProgress,
    EquilibrationSample,
    StabilityMonitor,
    equilibrate,
)


def _sample(
    index: int,
    *,
    temperature: float = 300.0,
    potential_energy: float = -1000.0,
    volume: float | None = None,
    density: float | None = None,
) -> EquilibrationSample:
    return EquilibrationSample(
        step=index * 5000,
        time_ps=index * 10.0,
        temperature_k=temperature,
        potential_energy_kj_mol=potential_energy,
        kinetic_energy_kj_mol=500.0,
        total_energy_kj_mol=potential_energy + 500.0,
        volume_nm3=volume,
        density_g_ml=density,
        pressure_bar=1.0 if volume is not None else None,
    )


class StabilityMonitorTests(unittest.TestCase):
    def test_nvt_requires_consecutive_stable_windows(self) -> None:
        monitor = StabilityMonitor(
            EquilibrationCriteria(
                window_samples=3,
                required_stable_windows=2,
            )
        )
        samples = tuple(_sample(index) for index in range(4))
        first = monitor(
            EquilibrationProgress(
                ensemble="NVT",
                target_temperature_k=300.0,
                target_pressure_bar=None,
                num_particles=100,
                max_steps=100_000,
                samples=samples[:3],
            )
        )
        second = monitor(
            EquilibrationProgress(
                ensemble="NVT",
                target_temperature_k=300.0,
                target_pressure_bar=None,
                num_particles=100,
                max_steps=100_000,
                samples=samples,
            )
        )

        self.assertFalse(first.stop)
        self.assertTrue(second.stop)
        self.assertTrue(second.successful)
        self.assertTrue(all(second.criteria.values()))
        self.assertEqual(second.metrics["stable_window_count"], 2.0)

    def test_npt_also_checks_volume_and_reports_density(self) -> None:
        monitor = StabilityMonitor(
            EquilibrationCriteria(
                window_samples=3,
                required_stable_windows=1,
            )
        )
        samples = tuple(
            _sample(index, volume=100.0, density=1.0)
            for index in range(3)
        )
        assessment = monitor(
            EquilibrationProgress(
                ensemble="NPT",
                target_temperature_k=300.0,
                target_pressure_bar=1.0,
                num_particles=100,
                max_steps=100_000,
                samples=samples,
            )
        )

        self.assertTrue(assessment.successful)
        self.assertTrue(assessment.criteria["volume_drift"])
        self.assertTrue(assessment.criteria["volume_fluctuation"])
        self.assertEqual(assessment.metrics["mean_density_g_ml"], 1.0)
        self.assertEqual(assessment.metrics["mean_pressure_bar"], 1.0)


class EquilibrateTests(unittest.TestCase):
    def test_custom_callback_stops_nvt_and_returns_history(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        seen_steps: list[int] = []

        def stop_after_first_block(progress):
            seen_steps.append(progress.current_step)
            return EquilibrationAssessment(
                stop=True,
                successful=True,
                reason="test_callback",
                criteria={"custom": True},
            )

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "equilibrated.pdb"
            result = equilibrate(
                input_path,
                output_path,
                ensemble="NVT",
                max_steps=10,
                check_interval_steps=2,
                timestep_fs=1.0,
                monitor=stop_after_first_block,
                forcefield_files=("amber14/tip3pfb.xml",),
                random_seed=7,
                verbose=False,
            )

            self.assertTrue(output_path.is_file())
            self.assertIn("END", output_path.read_text())

        self.assertEqual(seen_steps, [2])
        self.assertTrue(result.successful)
        self.assertEqual(result.termination_reason, "test_callback")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.samples), 1)
        self.assertGreater(result.final_sample.temperature_k, 0.0)

    def test_default_monitor_stops_at_maximum(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            result = equilibrate(
                input_path,
                Path(temp_dir) / "equilibrated.pdb",
                ensemble="NVT",
                max_steps=2,
                check_interval_steps=1,
                timestep_fs=1.0,
                forcefield_files=("amber14/tip3pfb.xml",),
                random_seed=7,
                verbose=False,
            )

        self.assertFalse(result.successful)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.samples), 2)

    def test_npt_rejects_structure_without_periodic_box(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "periodic box"):
                equilibrate(
                    input_path,
                    Path(temp_dir) / "equilibrated.pdb",
                    ensemble="NPT",
                    max_steps=1,
                    verbose=False,
                )

    def test_npt_runs_with_barostat_and_records_density(self) -> None:
        from openmm import Vec3
        from openmm.app import ForceField, Modeller, PDBFile, Topology
        from openmm.unit import nanometer

        forcefield = ForceField("amber14/tip3pfb.xml")
        modeller = Modeller(Topology(), [])
        modeller.addSolvent(
            forcefield,
            boxSize=Vec3(1.5, 1.5, 1.5) * nanometer,
        )

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "periodic.pdb"
            output_path = Path(temp_dir) / "equilibrated.pdb"
            with input_path.open("w") as output:
                PDBFile.writeFile(
                    modeller.topology,
                    modeller.positions,
                    output,
                )
            result = equilibrate(
                input_path,
                output_path,
                ensemble="NPT",
                max_steps=2,
                check_interval_steps=2,
                barostat_interval_steps=1,
                timestep_fs=1.0,
                forcefield_files=("amber14/tip3pfb.xml",),
                nonbonded_cutoff_nm=0.5,
                random_seed=11,
                monitor=lambda progress: True,
                verbose=False,
            )

        self.assertTrue(result.successful)
        self.assertEqual(result.ensemble, "NPT")
        self.assertIsNotNone(result.final_sample.volume_nm3)
        self.assertIsNotNone(result.final_sample.density_g_ml)
        self.assertIsNotNone(result.final_sample.pressure_bar)
        self.assertGreater(result.final_sample.density_g_ml, 0.0)


if __name__ == "__main__":
    unittest.main()
