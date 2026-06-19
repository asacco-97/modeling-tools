"""Model-agnostic tabular model evaluation utilities."""

from pe_tools.evaluation.evaluator import ModelEvaluator, evaluate_model
from pe_tools.evaluation.metrics import gini_index, normalized_gini
from pe_tools.evaluation.result import EvaluationResult

__all__ = [
    "EvaluationResult",
    "ModelEvaluator",
    "evaluate_model",
    "gini_index",
    "normalized_gini",
]
