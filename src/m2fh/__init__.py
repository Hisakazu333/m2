"""Finite-horizon Bayes control; no fixed-confidence certification claims."""
from .model import Model, Outcome, rational
from .solver import Solver

__version__ = "0.1.0"
__all__ = ["Model", "Outcome", "Solver", "rational"]
