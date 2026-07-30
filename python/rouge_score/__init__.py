"""ROUGE metrics accelerated by Mojo."""

from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import AggregateScore, BootstrapAggregator, Score

__all__ = [
    "AggregateScore",
    "BootstrapAggregator",
    "RougeScorer",
    "Score",
]
