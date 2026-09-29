"""Public molecular-dynamics preparation and simulation API.

Implementation details live in small modules under
:mod:`biotools.md_simulations`.  This facade preserves the established
``biotools.mdtools`` import path.
"""

from .md_simulations.common import (
    Ensemble,
    MDInput,
    MDStageResult,
    ResumeMode,
    SimulationConfig,
)
from .md_simulations.equilibration import (
    EquilibrationAssessment,
    EquilibrationCriteria,
    EquilibrationMonitor,
    EquilibrationProgress,
    EquilibrationResult,
    EquilibrationSample,
    MonitorCallback,
    StabilityMonitor,
    equilibrate,
)
from .md_simulations.minimization import (
    MinimizationResult,
    MinimizationSample,
    _classify_minimization_termination,
    _get_raw_state_diagnostics,
    _IterationReporter,
    _should_restart_optimizer,
    minimize,
)
from .md_simulations.plotting import plot_md_result, plot_trajectory
from .md_simulations.preparation import fix_pdb, model_solvent
from .md_simulations.production import ProductionResult, run_production
from .md_simulations.soft_equilibration import soft_equilibrate_nvt

__all__ = [
    "Ensemble",
    "EquilibrationAssessment",
    "EquilibrationCriteria",
    "EquilibrationMonitor",
    "EquilibrationProgress",
    "EquilibrationResult",
    "EquilibrationSample",
    "MDInput",
    "MDStageResult",
    "MinimizationResult",
    "MinimizationSample",
    "MonitorCallback",
    "ProductionResult",
    "ResumeMode",
    "SimulationConfig",
    "StabilityMonitor",
    "equilibrate",
    "fix_pdb",
    "minimize",
    "model_solvent",
    "plot_md_result",
    "plot_trajectory",
    "run_production",
    "soft_equilibrate_nvt",
]
