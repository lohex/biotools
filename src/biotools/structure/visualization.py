"""Interactive protein structure visualization."""

from io import StringIO


def plot_structure(structure):
    """Create a ``py3Dmol`` view for a structure."""
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
