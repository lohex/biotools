"""Typed contact-analysis API.

The functions in this package complement the backwards-compatible
``characterize_*_contacts`` helpers.  They expose the rule profile,
diagnostics, stable atom identities, and all atomic observations.
"""

from .analysis import analyze_contacts, prepare_contact_system
from .aggregate import aggregate_contact_features, ContactFeatureSet
from .config import ContactConfig, get_contact_config
from .models import (
    AtomReference,
    ContactAnalysisResult,
    ContactDiagnostic,
    ContactObservation,
    ContactSystem,
    CriterionResult,
    GeometryMeasurement,
)

__all__ = [
    "analyze_contacts",
    "aggregate_contact_features",
    "AtomReference",
    "ContactAnalysisResult",
    "ContactConfig",
    "ContactDiagnostic",
    "ContactFeatureSet",
    "ContactObservation",
    "ContactSystem",
    "CriterionResult",
    "GeometryMeasurement",
    "get_contact_config",
    "prepare_contact_system",
]
