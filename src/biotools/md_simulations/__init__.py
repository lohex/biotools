"""Implementation modules for molecular-dynamics simulations."""


__all__ = ["plot_trajectory"]


def __getattr__(name: str):
    if name == "plot_trajectory":
        from .plotting import plot_trajectory

        return plot_trajectory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
