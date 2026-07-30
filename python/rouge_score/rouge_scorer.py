"""ROUGE-N, ROUGE-L, and ROUGE-Lsum scoring with Mojo kernels."""

from __future__ import annotations

import collections
import itertools
import re

import nltk
import numpy as np

from rouge_score import scoring, tokenizers
from rouge_score import _lib


def _ensure_str(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _encode_many(sequences):
    token_ids = dict.fromkeys(itertools.chain.from_iterable(sequences))
    for value, token in enumerate(token_ids, 1):
        token_ids[token] = value
    lookup = token_ids.__getitem__
    return [
        np.fromiter(
            map(lookup, sequence),
            dtype=np.int64,
            count=len(sequence),
        )
        for sequence in sequences
    ]


def _score_from_hits(hits, target_count, prediction_count):
    precision = hits / max(prediction_count, 1)
    recall = hits / max(target_count, 1)
    return scoring.Score(
        precision=precision,
        recall=recall,
        fmeasure=scoring.fmeasure(precision, recall),
    )


def _ngram_overlap_ids(target, prediction, n):
    target_count = max(target.size - n + 1, 0)
    prediction_count = max(prediction.size - n + 1, 0)
    if not target_count or not prediction_count:
        return 0
    if n == 1:
        vocabulary_size = max(
            int(target.max(initial=0)),
            int(prediction.max(initial=0)),
        )
        counts = np.zeros(vocabulary_size + 1, dtype=np.int64)
        return int(
            _lib.unigram_overlap(target, prediction, counts)
        )
    capacity = 1
    while capacity * 3 < target_count * 4:
        capacity <<= 1
    starts = np.zeros(capacity, dtype=np.int64)
    counts = np.zeros(capacity, dtype=np.int64)
    return int(
        _lib.ngram_overlap(target, prediction, n, starts, counts)
    )


def _score_ngram_ids(target, prediction, n):
    target_count = max(target.size - n + 1, 0)
    prediction_count = max(prediction.size - n + 1, 0)
    hits = _ngram_overlap_ids(target, prediction, n)
    return _score_from_hits(hits, target_count, prediction_count)


def _score_lcs_ids(target, prediction):
    if not target.size or not prediction.size:
        return scoring.Score(precision=0, recall=0, fmeasure=0)
    row = np.empty(prediction.size + 1, dtype=np.int64)
    hits = int(
        _lib.lcs_length(target, prediction, row)
    )
    return _score_from_hits(hits, target.size, prediction.size)


def _lcs_indices_ids(ref, candidate):
    if not ref.size or not candidate.size:
        return np.empty(0, dtype=np.int64)
    table = np.empty((ref.size + 1) * (candidate.size + 1), dtype=np.int64)
    indices = np.empty(min(ref.size, candidate.size), dtype=np.int64)
    length = int(
        _lib.lcs_indices(ref, candidate, table, indices)
    )
    return indices[:length]


class RougeScorer(scoring.BaseScorer):
    def __init__(
        self, rouge_types, use_stemmer=False, split_summaries=False, tokenizer=None
    ):
        self.rouge_types = rouge_types
        self._tokenizer = tokenizer or tokenizers.DefaultTokenizer(use_stemmer)
        self._split_summaries = split_summaries

    def score_multi(self, targets, prediction):
        score_dicts = [self.score(target, prediction) for target in targets]
        max_score = {}
        for key in self.rouge_types:
            index = np.argmax([scores[key].fmeasure for scores in score_dicts])
            max_score[key] = score_dicts[index][key]
        return max_score

    def score(self, target, prediction):
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
            encoded_pair = None
        else:
            target_tokens = self._tokenizer.tokenize(target)
            prediction_tokens = self._tokenizer.tokenize(prediction)
            encoded_pair = _encode_many([target_tokens, prediction_tokens])

        result = {}
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                result[rouge_type] = _score_lcs_ids(*encoded_pair)
            elif rouge_type == "rougeLsum":
                def get_sents(text):
                    if self._split_summaries:
                        sents = nltk.sent_tokenize(text)
                    else:
                        sents = _ensure_str(text).split("\n")
                    return [sentence for sentence in sents if len(sentence)]

                target_sentences = [
                    self._tokenizer.tokenize(sentence) for sentence in get_sents(target)
                ]
                prediction_sentences = [
                    self._tokenizer.tokenize(sentence)
                    for sentence in get_sents(prediction)
                ]
                result[rouge_type] = _summary_level_lcs(
                    target_sentences, prediction_sentences
                )
            elif re.match(r"rouge[0-9]$", _ensure_str(rouge_type)):
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                result[rouge_type] = _score_ngram_ids(*encoded_pair, n)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
        return result


def _create_ngrams(tokens, n):
    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def _score_ngrams(target_ngrams, prediction_ngrams):
    hits = sum(
        min(count, prediction_ngrams[ngram])
        for ngram, count in target_ngrams.items()
    )
    return _score_from_hits(
        hits, sum(target_ngrams.values()), sum(prediction_ngrams.values())
    )


def _score_lcs(target_tokens, prediction_tokens):
    if not target_tokens or not prediction_tokens:
        return scoring.Score(precision=0, recall=0, fmeasure=0)
    return _score_lcs_ids(*_encode_many([target_tokens, prediction_tokens]))


def _lcs_table(ref, can):
    table = [[0] * (len(can) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(can) + 1):
            if ref[i - 1] == can[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table


def _backtrack_norec(table, ref, can):
    i = len(ref)
    j = len(can)
    lcs = []
    while i > 0 and j > 0:
        if ref[i - 1] == can[j - 1]:
            lcs.insert(0, i - 1)
            i -= 1
            j -= 1
        elif table[i][j - 1] > table[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return lcs


def lcs_ind(ref, can):
    if not ref or not can:
        return []
    encoded = _encode_many([ref, can])
    return _lcs_indices_ids(*encoded).tolist()


def _find_union(lcs_list):
    return sorted(list(set().union(*lcs_list)))


def _union_lcs(ref, c_list):
    lcs_list = [lcs_ind(ref, candidate) for candidate in c_list]
    return [ref[i] for i in _find_union(lcs_list)]


def _summary_level_lcs(ref_sent, can_sent):
    if not ref_sent or not can_sent:
        return scoring.Score(precision=0, recall=0, fmeasure=0)
    target_count = sum(map(len, ref_sent))
    prediction_count = sum(map(len, can_sent))
    if not target_count or not prediction_count:
        return scoring.Score(precision=0, recall=0, fmeasure=0)

    all_encoded = _encode_many([*ref_sent, *can_sent])
    encoded_ref = all_encoded[: len(ref_sent)]
    encoded_can = all_encoded[len(ref_sent) :]
    target_counts = collections.Counter(
        int(token) for sentence in encoded_ref for token in sentence
    )
    prediction_counts = collections.Counter(
        int(token) for sentence in encoded_can for token in sentence
    )

    hits = 0
    for ref in encoded_ref:
        selected = np.zeros(ref.size, dtype=np.bool_)
        for candidate in encoded_can:
            selected[_lcs_indices_ids(ref, candidate)] = True
        for token in ref[selected]:
            key = int(token)
            if prediction_counts[key] > 0 and target_counts[key] > 0:
                hits += 1
                prediction_counts[key] -= 1
                target_counts[key] -= 1
    return _score_from_hits(hits, target_count, prediction_count)
