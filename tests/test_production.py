"""Tests for fixed-length molecular-dynamics production runs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from biotools.mdtools import ProductionResult, run_production


class ProductionRunTests(unittest.TestCase):
    def test_nvt_writes_trajectory_log_and_restart_files(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "production-final.pdb"
            trajectory_path = temp_path / "production.xtc"
            log_path = temp_path / "production.csv"
            state_path = temp_path / "production-state.xml"
            checkpoint_path = temp_path / "production.chk"

            result = run_production(
                input_path,
                output_path,
                trajectory_file=trajectory_path,
                steps=2,
                ensemble="NVT",
                timestep_fs=1.0,
                trajectory_interval_steps=1,
                log_file=log_path,
                log_interval_steps=1,
                checkpoint_interval_steps=1,
                forcefield_files=("amber14/tip3pfb.xml",),
                random_seed=17,
                state_output_file=state_path,
                checkpoint_output_file=checkpoint_path,
                verbose=False,
            )

            self.assertIsInstance(result, ProductionResult)
            self.assertTrue(output_path.is_file())
            self.assertIn("END", output_path.read_text())
            self.assertGreater(trajectory_path.stat().st_size, 0)
            log_lines = log_path.read_text().splitlines()
            self.assertEqual(len(log_lines), 3)
            self.assertIn("Potential Energy", log_lines[0])
            self.assertIn("<State", state_path.read_text())
            self.assertGreater(checkpoint_path.stat().st_size, 0)

        self.assertEqual(result.ensemble, "NVT")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.initial_step, 0)
        self.assertEqual(result.final_step, 2)
        self.assertAlmostEqual(result.elapsed_time_ps, 0.002)
        self.assertAlmostEqual(result.simulation_time_ps, 0.002)
        self.assertIsNone(result.target_pressure_bar)
        self.assertEqual(result.trajectory_path, trajectory_path)
        self.assertEqual(result.log_path, log_path)
        self.assertEqual(result.state_path, state_path)
        self.assertEqual(result.checkpoint_path, checkpoint_path)

    def test_checkpoint_resume_appends_trajectory_and_log(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory_path = temp_path / "production.dcd"
            log_path = temp_path / "production.csv"
            checkpoint_path = temp_path / "production.chk"
            common_options = {
                "trajectory_file": trajectory_path,
                "steps": 2,
                "ensemble": "NVT",
                "timestep_fs": 1.0,
                "trajectory_interval_steps": 1,
                "log_file": log_path,
                "log_interval_steps": 1,
                "checkpoint_interval_steps": 1,
                "forcefield_files": ("amber14/tip3pfb.xml",),
                "random_seed": 23,
                "checkpoint_output_file": checkpoint_path,
                "verbose": False,
            }
            first = run_production(
                input_path,
                temp_path / "segment-1.pdb",
                **common_options,
            )
            initial_trajectory_size = trajectory_path.stat().st_size
            second = run_production(
                first,
                temp_path / "segment-2.pdb",
                append=True,
                **common_options,
            )

            self.assertGreater(
                trajectory_path.stat().st_size, initial_trajectory_size
            )
            self.assertEqual(len(log_path.read_text().splitlines()), 5)
            with self.assertRaisesRegex(
                ValueError, "incompatible checkpoint"
            ):
                run_production(
                    first,
                    temp_path / "incompatible.pdb",
                    trajectory_file=temp_path / "incompatible.dcd",
                    steps=1,
                    ensemble="NVT",
                    timestep_fs=2.0,
                    forcefield_files=("amber14/tip3pfb.xml",),
                    verbose=False,
                )

        self.assertEqual(first.initial_step, 0)
        self.assertEqual(first.final_step, 2)
        self.assertEqual(second.initial_step, 2)
        self.assertEqual(second.final_step, 4)
        self.assertEqual(second.steps, 2)
        self.assertTrue(second.appended)
        self.assertEqual(second.input_checkpoint_path, checkpoint_path)
        self.assertEqual(second.resume_mode, "checkpoint")
        self.assertTrue(
            first.simulation_config.checkpoint_compatible(
                second.simulation_config
            )
        )

    def test_validates_npt_trajectory_and_append_requirements(self) -> None:
        input_path = Path(__file__).with_name("data") / "water.pdb"
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "periodic box"):
                run_production(
                    input_path,
                    temp_path / "final.pdb",
                    trajectory_file=temp_path / "trajectory.dcd",
                    steps=1,
                    ensemble="NPT",
                    forcefield_files=("amber14/tip3pfb.xml",),
                    verbose=False,
                )
            with self.assertRaisesRegex(ValueError, r"\.dcd or \.xtc"):
                run_production(
                    input_path,
                    temp_path / "final.pdb",
                    trajectory_file=temp_path / "trajectory.pdb",
                    steps=1,
                    ensemble="NVT",
                    verbose=False,
                )
            with self.assertRaisesRegex(ValueError, "append=True requires"):
                run_production(
                    input_path,
                    temp_path / "final.pdb",
                    trajectory_file=temp_path / "trajectory.dcd",
                    steps=1,
                    ensemble="NVT",
                    append=True,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()
