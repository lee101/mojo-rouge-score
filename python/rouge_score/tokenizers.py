"""Tokenizer classes compatible with rouge-score."""

from __future__ import annotations

import abc

from nltk.stem import porter

from rouge_score import tokenize


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
    def __init__(self, use_stemmer=False):
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return tokenize.tokenize(text, self._stemmer)
