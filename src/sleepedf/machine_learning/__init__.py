"""Model construction and subject-separated evaluation."""
from .baseline import default_baseline
from .evaluation import evaluate_loso

__all__ = ["default_baseline", "evaluate_loso"]
