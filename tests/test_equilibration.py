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
    soft_equilibrate_nvt,
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
            state_path = Path(temp_dir) / "equilibrated-state.xml"
            checkpoint_path = Path(temp_dir) / "equilibrated.chk"
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
                state_output_file=state_path,
                checkpoint_output_file=checkpoint_path,
                verbose=False,
            )

            self.assertTrue(output_path.is_file())
            self.assertIn("END", output_path.read_text())
            serialized_state = state_path.read_text()
            self.assertIn("<State", serialized_state)
            self.assertIn("<Velocities", serialized_state)
            self.assertGreater(checkpoint_path.stat().st_size, 0)

            state_continuation = equilibrate(
                input_path,
                Path(temp_dir) / "continued-from-state.pdb",
                ensemble="NVT",
                max_steps=1,
                check_interval_steps=1,
                timestep_fs=1.0,
                monitor=lambda progress: True,
                forcefield_files=("amber14/tip3pfb.xml",),
                state_input_file=state_path,
                verbose=False,
            )
            checkpoint_continuation = equilibrate(
                input_path,
                Path(temp_dir) / "continued-from-checkpoint.pdb",
                ensemble="NVT",
                max_steps=1,
                check_interval_steps=1,
                timestep_fs=1.0,
                monitor=lambda progress: True,
                forcefield_files=("amber14/tip3pfb.xml",),
                checkpoint_input_file=checkpoint_path,
                verbose=False,
            )

        self.assertEqual(seen_steps, [2])
        self.assertTrue(result.successful)
        self.assertEqual(result.termination_reason, "test_callback")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.samples), 1)
        self.assertGreater(result.final_sample.temperature_k, 0.0)
        self.assertEqual(result.state_path, state_path)
        self.assertEqual(result.checkpoint_path, checkpoint_path)
        self.assertEqual(state_continuation.input_state_path, state_path)
        self.assertEqual(state_continuation.initial_step, 2)
        self.assertEqual(state_continuation.final_step, 3)
        self.assertEqual(state_continuation.steps, 1)
        self.assertEqual(state_continuation.final_sample.step, 3)
        self.assertEqual(
            checkpoint_continuation.input_checkpoint_path,
            checkpoint_path,
        )
        self.assertEqual(checkpoint_continuation.initial_step, 2)
        self.assertEqual(checkpoint_continuation.final_step, 3)

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
            nvt_path = Path(temp_dir) / "nvt.pdb"
            nvt_state_path = Path(temp_dir) / "nvt-state.xml"
            output_path = Path(temp_dir) / "equilibrated.pdb"
            with input_path.open("w") as output:
                PDBFile.writeFile(
                    modeller.topology,
                    modeller.positions,
                    output,
                )
            equilibrate(
                input_path,
                nvt_path,
                ensemble="NVT",
                max_steps=2,
                check_interval_steps=2,
                timestep_fs=1.0,
                forcefield_files=("amber14/tip3pfb.xml",),
                nonbonded_cutoff_nm=0.5,
                random_seed=11,
                monitor=lambda progress: True,
                state_output_file=nvt_state_path,
                verbose=False,
            )
            result = equilibrate(
                nvt_path,
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
                state_input_file=nvt_state_path,
                verbose=False,
            )

        self.assertTrue(result.successful)
        self.assertEqual(result.ensemble, "NPT")
        self.assertIsNotNone(result.final_sample.volume_nm3)
        self.assertIsNotNone(result.final_sample.density_g_ml)
        self.assertIsNotNone(result.final_sample.pressure_bar)
        self.assertGreater(result.final_sample.density_g_ml, 0.0)
        self.assertEqual(result.initial_step, 2)
        self.assertEqual(result.final_step, 4)
        self.assertEqual(result.final_sample.step, 4)

    def test_resume_inputs_are_mutually_exclusive(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "resume"
            resume_path.write_text("placeholder")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                equilibrate(
                    input_path,
                    Path(temp_dir) / "equilibrated.pdb",
                    ensemble="NVT",
                    state_input_file=resume_path,
                    checkpoint_input_file=resume_path,
                    max_steps=1,
                    verbose=False,
                )


class SoftEquilibrateNVTTests(unittest.TestCase):
    def test_soft_nvt_ramps_temperature_and_timestep_in_one_run(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "soft-equilibrated.pdb"
            state_path = Path(temp_dir) / "soft-state.xml"
            checkpoint_path = Path(temp_dir) / "soft.chk"
            result = soft_equilibrate_nvt(
                input_path,
                output_path,
                initial_temperature_k=50.0,
                temperature_k=300.0,
                initial_timestep_fs=0.25,
                timestep_fs=1.0,
                heating_steps=4,
                heating_stages=2,
                max_steps=6,
                check_interval_steps=2,
                monitor=lambda progress: True,
                forcefield_files=("amber14/tip3pfb.xml",),
                random_seed=13,
                state_output_file=state_path,
                checkpoint_output_file=checkpoint_path,
                verbose=False,
            )

            self.assertTrue(output_path.is_file())
            serialized_state = state_path.read_text()
            self.assertIn("<State", serialized_state)
            self.assertIn("<Velocities", serialized_state)
            self.assertGreater(checkpoint_path.stat().st_size, 0)

        self.assertTrue(result.successful)
        self.assertEqual(result.steps, 6)
        self.assertEqual(result.warmup_steps, 4)
        self.assertEqual(result.initial_temperature_k, 50.0)
        self.assertEqual(result.initial_timestep_fs, 0.25)
        self.assertEqual(result.target_timestep_fs, 1.0)
        self.assertEqual(result.initial_step, 0)
        self.assertEqual(result.final_step, 6)
        self.assertEqual(
            [sample.phase for sample in result.samples],
            ["heating", "heating", "equilibration"],
        )
        self.assertEqual(result.samples[0].target_temperature_k, 50.0)
        self.assertEqual(result.samples[0].timestep_fs, 0.25)
        self.assertEqual(result.samples[1].target_temperature_k, 300.0)
        self.assertEqual(result.samples[1].timestep_fs, 1.0)

    def test_soft_nvt_validates_heating_schedule(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "equilibrated.pdb"
            with self.assertRaisesRegex(ValueError, "initial_temperature"):
                soft_equilibrate_nvt(
                    input_path,
                    output_path,
                    initial_temperature_k=300.0,
                    temperature_k=300.0,
                    verbose=False,
                )
            with self.assertRaisesRegex(ValueError, "greater than heating_steps"):
                soft_equilibrate_nvt(
                    input_path,
                    output_path,
                    heating_steps=10,
                    max_steps=10,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()
