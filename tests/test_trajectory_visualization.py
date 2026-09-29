"""Regression checks for DCD/XTC trajectory visualization."""

from __future__ import annotations

import numpy as np
import pytest
from biotite.structure.io.dcd import DCDFile
from biotite.structure.io.xtc import XTCFile

from biotools.mdtools import plot_trajectory
from biotools.md_simulations import plot_trajectory as md_plot_trajectory


@pytest.mark.parametrize("reader,suffix", [(DCDFile, ".dcd"), (XTCFile, ".xtc")])
def test_plot_trajectory_selects_frames(tmp_path, reader, suffix) -> None:
    topology = tmp_path / "topology.pdb"
    topology.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    coordinates = np.array(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 2, 0], [1, 2, 0]],
            [[0, 3, 0], [1, 3, 0]],
        ],
        dtype=np.float32,
    )
    file = reader()
    file.set_coord(coordinates)
    trajectory = tmp_path / ("trajectory" + suffix)
    file.write(trajectory)

    assert md_plot_trajectory is plot_trajectory
    view = plot_trajectory(topology, trajectory, start=1, stop=4, step=2)
    html = view.write_html()
    assert "setCoordinates" in html
    assert "viewer_" in html
    assert "animate" in html
    assert "[0.0, 1.0, 0.0]" in html
    assert "[0.0, 3.0, 0.0]" in html
    assert "[0.0, 2.0, 0.0]" not in html

    with pytest.raises(ValueError, match="max_frames"):
        plot_trajectory(topology, trajectory, max_frames=2)

    wrong_topology = tmp_path / "wrong.pdb"
    wrong_topology.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"
    )
    with pytest.raises(ValueError, match="topology has 1"):
        plot_trajectory(wrong_topology, trajectory)
