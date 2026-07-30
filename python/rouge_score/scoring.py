"""Score containers and bootstrap aggregation compatible with rouge-score."""

from __future__ import annotations

import abc
import collections
from typing import Dict

import numpy as np


class Score(collections.namedtuple("Score", ["precision", "recall", "fmeasure"])):
    """Precision, recall, and harmonic-mean score."""


class BaseScorer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def score(self, target: str, prediction: str) -> Dict[str, Score]:
        raise NotImplementedError


class AggregateScore(collections.namedtuple("AggregateScore", ["low", "mid", "high"])):
    """Low, median, and high bootstrap intervals."""


class BootstrapAggregator:
    def __init__(self, confidence_interval=0.95, n_samples=1000):
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError("confidence_interval must be in range [0, 1]")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self._n_samples = n_samples
        self._confidence_interval = confidence_interval
        self._scores = collections.defaultdict(list)

    def add_scores(self, scores):
        for score_type, score in scores.items():
            self._scores[score_type].append(score)

    def aggregate(self):
        result = {}
        for score_type, scores in self._scores.items():
            score_matrix = np.vstack(tuple(scores))
            percentiles = self._bootstrap_resample(score_matrix)
            intervals = tuple(
                scores[0].__class__(*percentiles[j, :]) for j in range(3)
            )
            result[score_type] = AggregateScore(
                low=intervals[0], mid=intervals[1], high=intervals[2]
            )
        return result

    def _bootstrap_resample(self, matrix):
        sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
        for i in range(self._n_samples):
            sample_idx = np.random.choice(
                np.arange(matrix.shape[0]), size=matrix.shape[0]
            )
            sample_mean[i, :] = np.mean(matrix[sample_idx, :], axis=0)
        percentile_delta = (1 - self._confidence_interval) / 2
        q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
        return np.percentile(sample_mean, q, axis=0)


def fmeasure(precision, recall):
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0
