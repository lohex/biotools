"""Interactive protein structure visualization."""

from __future__ import annotations

from io import StringIO
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure


def plot_structure(structure: Structure) -> Any:
    """Create an interactive ``py3Dmol`` view for a structure.

    Args:
        structure: Biopython structure to render as a spectrum-colored cartoon.

    Returns:
        Configured ``py3Dmol.view`` instance.
    """
    import py3Dmol
    from Bio.PDB import PDBIO

    buffer = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(buffer)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(buffer.getvalue(), "pdb")
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view
