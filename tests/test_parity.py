import random
import re

import numpy as np
import pytest

from conftest import (
    upstream_rouge_scorer,
    upstream_scoring,
    upstream_tokenize,
    upstream_tokenizers,
)
from rouge_score import _lib, rouge_scorer, scoring, tokenize, tokenizers


ALL_TYPES = [*(f"rouge{n}" for n in range(1, 10)), "rougeL"]


def assert_scores_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=0.0, rel=0.0)
        assert isinstance(actual[key], scoring.Score)


@pytest.mark.parametrize(
    ("target", "prediction"),
    [
        ("The quick brown fox jumps over the lazy dog.", "The quick brown dog."),
        ("", ""),
        ("one two three", ""),
        ("", "one two three"),
        ("a a a a b b", "a a b b b"),
        ("Café déjà vu — 東京 123", "CAFE deja vu 123"),
        (b"Bytes INPUT 123", b"bytes output 123"),
    ],
)
def test_all_score_types_match_upstream(target, prediction):
    actual = rouge_scorer.RougeScorer(ALL_TYPES).score(target, prediction)
    expected = upstream_rouge_scorer.RougeScorer(ALL_TYPES).score(
        target, prediction
    )
    assert_scores_equal(actual, expected)


def test_randomized_scores_match_upstream():
    rng = random.Random(17)
    vocabulary = ["alpha", "beta", "gamma", "delta", "42", "x"]
    ours = rouge_scorer.RougeScorer(ALL_TYPES)
    theirs = upstream_rouge_scorer.RougeScorer(ALL_TYPES)
    for _ in range(100):
        target = " ".join(
            rng.choice(vocabulary) for _ in range(rng.randrange(0, 90))
        )
        prediction = " ".join(
            rng.choice(vocabulary) for _ in range(rng.randrange(0, 90))
        )
        assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


def test_repeated_ngrams_use_multiset_intersection():
    target = "a b a b a b a"
    prediction = "a b a x a b"
    ours = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"])
    theirs = upstream_rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"])
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


def test_compact_unigram_counts_match_upstream():
    target = " ".join(["a"] * 257 + ["b"] * 129 + ["c"])
    prediction = " ".join(["a"] * 128 + ["b"] * 258 + ["d"])
    ours = rouge_scorer.RougeScorer(["rouge1"])
    theirs = upstream_rouge_scorer.RougeScorer(["rouge1"])
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


def test_simd_ngram_comparison_scalar_tail_matches_upstream():
    target = "a b c d e f g h i j a b c d e x g h i j"
    prediction = "a b c d e y g h i j a b c d e f g h i j"
    rouge_types = ["rouge5", "rouge9"]
    ours = rouge_scorer.RougeScorer(rouge_types)
    theirs = upstream_rouge_scorer.RougeScorer(rouge_types)
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


@pytest.mark.parametrize(
    "bad",
    [
        np.arange(6, dtype=np.int32),
        np.arange(12, dtype=np.int64)[::2],
        np.arange(6, dtype=np.int64).reshape(2, 3),
    ],
)
def test_ffi_rejects_incompatible_input_buffers(bad):
    good = np.arange(6, dtype=np.int64)
    row = np.empty(7, dtype=np.int64)
    with pytest.raises((TypeError, ValueError)):
        _lib.lcs_length(bad, good, row)


def test_ffi_rejects_read_only_or_wrong_sized_output_buffers():
    tokens = np.arange(6, dtype=np.int64)
    row = np.empty(7, dtype=np.int64)
    row.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        _lib.lcs_length(tokens, tokens, row)
    with pytest.raises(ValueError, match="row length"):
        _lib.lcs_length(tokens, tokens, np.empty(6, dtype=np.int64))


def test_ffi_rejects_unigram_ids_outside_counts_buffer():
    target = np.array([0, 3], dtype=np.int64)
    prediction = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        _lib.unigram_overlap(target, prediction, np.zeros(3, dtype=np.int64))


def test_ffi_rejects_invalid_hash_table_layout():
    tokens = np.arange(8, dtype=np.int64)
    with pytest.raises(ValueError, match="power of two"):
        _lib.ngram_overlap(
            tokens,
            tokens,
            2,
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        )


def test_native_ffi_rejects_null_pointers_and_negative_lengths():
    assert _lib.lib().mrs_lcs_length(0, 1, 0, 1, 0) == -1
    assert _lib.lib().mrs_lcs_length(0, -1, 0, 0, 0) == -1


def test_ffi_kernel_failures_are_not_silently_accepted():
    with pytest.raises(RuntimeError, match="invalid result"):
        _lib._checked_result("test", -1, 3)


def test_base_scorer_contract_matches_upstream():
    assert issubclass(rouge_scorer.RougeScorer, scoring.BaseScorer)
    with pytest.raises(TypeError):
        scoring.BaseScorer()


def test_stemming_matches_upstream():
    target = "connections connected connecting easily"
    prediction = "connection connect connects easy"
    ours = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    theirs = upstream_rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


class PipeTokenizer:
    def tokenize(self, text):
        return text.lower().split("|")


def test_custom_tokenizer_matches_upstream():
    ours = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=PipeTokenizer()
    )
    theirs = upstream_rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=PipeTokenizer()
    )
    assert_scores_equal(
        ours.score("A|B|A|C", "A|A|B|D"),
        theirs.score("A|B|A|C", "A|A|B|D"),
    )


def test_score_multi_selects_best_target_per_metric():
    targets = ["one two three four", "one two x x", "three four five"]
    prediction = "one two three"
    ours = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    theirs = upstream_rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    assert_scores_equal(
        ours.score_multi(targets, prediction),
        theirs.score_multi(targets, prediction),
    )


@pytest.mark.parametrize("rouge_type", ["rouge0", "rouge10", "ROUGE1", "bleu"])
def test_invalid_metric_matches_upstream(rouge_type):
    with pytest.raises(ValueError) as ours:
        rouge_scorer.RougeScorer([rouge_type]).score("a", "a")
    with pytest.raises(ValueError) as theirs:
        upstream_rouge_scorer.RougeScorer([rouge_type]).score("a", "a")
    assert str(ours.value) == str(theirs.value)


@pytest.mark.parametrize(
    ("ref", "candidate"),
    [
        (list("abcbdab"), list("bdcaba")),
        (list("aaaa"), list("aa")),
        (list("XMJYAUZ"), list("MZJAWXU")),
        ([], list("abc")),
        (list("abc"), []),
    ],
)
def test_lcs_indices_match_upstream(ref, candidate):
    assert rouge_scorer.lcs_ind(ref, candidate) == upstream_rouge_scorer.lcs_ind(
        ref, candidate
    )


def test_lcs_table_and_backtracking_match_upstream():
    ref = list("abcbdab")
    candidate = list("bdcaba")
    ours = rouge_scorer._lcs_table(ref, candidate)
    theirs = upstream_rouge_scorer._lcs_table(ref, candidate)
    assert ours == theirs
    assert rouge_scorer._backtrack_norec(
        ours, ref, candidate
    ) == upstream_rouge_scorer._backtrack_norec(theirs, ref, candidate)


def test_randomized_summary_lcs_matches_upstream():
    rng = random.Random(91)
    vocabulary = ["a", "b", "c", "d", "e"]
    for _ in range(100):
        refs = [
            rng.choices(vocabulary, k=rng.randrange(0, 18))
            for _ in range(rng.randrange(0, 6))
        ]
        candidates = [
            rng.choices(vocabulary, k=rng.randrange(0, 18))
            for _ in range(rng.randrange(0, 6))
        ]
        actual = rouge_scorer._summary_level_lcs(refs, candidates)
        expected = upstream_rouge_scorer._summary_level_lcs(refs, candidates)
        assert actual == pytest.approx(expected, abs=0.0, rel=0.0)


def test_rouge_lsum_newline_parity():
    target = "the cat sat on the mat\nthe dog ran home\none repeated token"
    prediction = "the dog sat on the mat\nthe cat ran away\none token token"
    ours = rouge_scorer.RougeScorer(["rougeLsum"])
    theirs = upstream_rouge_scorer.RougeScorer(["rougeLsum"])
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


def test_rouge_lsum_sentence_splitting_parity(monkeypatch):
    def sentence_split(text):
        return [part.strip() for part in text.split(".") if part.strip()]

    monkeypatch.setattr(rouge_scorer.nltk, "sent_tokenize", sentence_split)
    ours = rouge_scorer.RougeScorer(["rougeLsum"], split_summaries=True)
    theirs = upstream_rouge_scorer.RougeScorer(
        ["rougeLsum"], split_summaries=True
    )
    target = "The cat sat down. The dog ran home."
    prediction = "The dog sat down. The cat stayed home."
    assert_scores_equal(ours.score(target, prediction), theirs.score(target, prediction))


def test_union_lcs_and_double_count_prevention_match_upstream():
    refs = [["a", "b", "a"], ["a", "c"]]
    candidates = [["a", "b"], ["b", "a", "c"]]
    assert rouge_scorer._union_lcs(
        refs[0], candidates
    ) == upstream_rouge_scorer._union_lcs(refs[0], candidates)
    assert rouge_scorer._summary_level_lcs(
        refs, candidates
    ) == upstream_rouge_scorer._summary_level_lcs(refs, candidates)


@pytest.mark.parametrize(
    "text",
    [
        "Hello, WORLD! 123",
        "résumé naïve café",
        "snake_case and-dashes",
        "",
        b"BYTE text 42",
    ],
)
def test_default_tokenization_matches_upstream(text):
    assert tokenize.tokenize(text, None) == upstream_tokenize.tokenize(text, None)
    assert tokenizers.DefaultTokenizer().tokenize(
        text
    ) == upstream_tokenizers.DefaultTokenizer().tokenize(text)


def test_create_and_score_ngrams_match_upstream():
    target = "a b a b a".split()
    prediction = "a b b a".split()
    for n in (1, 2, 3, 7):
        ours_target = rouge_scorer._create_ngrams(target, n)
        ours_prediction = rouge_scorer._create_ngrams(prediction, n)
        theirs_target = upstream_rouge_scorer._create_ngrams(target, n)
        theirs_prediction = upstream_rouge_scorer._create_ngrams(prediction, n)
        assert ours_target == theirs_target
        assert ours_prediction == theirs_prediction
        assert rouge_scorer._score_ngrams(
            ours_target, ours_prediction
        ) == upstream_rouge_scorer._score_ngrams(
            theirs_target, theirs_prediction
        )


def test_score_containers_match_upstream_shape():
    score = scoring.Score(0.1, 0.2, 0.3)
    aggregate = scoring.AggregateScore(score, score, score)
    assert score._fields == upstream_scoring.Score._fields
    assert aggregate._fields == upstream_scoring.AggregateScore._fields
    assert tuple(score) == (0.1, 0.2, 0.3)


@pytest.mark.parametrize(
    ("confidence", "samples", "message"),
    [
        (-0.1, 100, "confidence_interval must be in range [0, 1]"),
        (1.1, 100, "confidence_interval must be in range [0, 1]"),
        (0.95, 0, "n_samples must be positive"),
    ],
)
def test_bootstrap_validation_matches_upstream(confidence, samples, message):
    with pytest.raises(ValueError, match=r"^" + re.escape(message) + r"$"):
        scoring.BootstrapAggregator(confidence, samples)


def test_bootstrap_aggregation_matches_upstream():
    samples = [
        {
            "rouge1": scoring.Score(0.2 + i * 0.05, 0.3, 0.4),
            "rougeL": scoring.Score(0.1, 0.2 + i * 0.03, 0.25),
        }
        for i in range(8)
    ]
    ours = scoring.BootstrapAggregator(confidence_interval=0.9, n_samples=200)
    theirs = upstream_scoring.BootstrapAggregator(
        confidence_interval=0.9, n_samples=200
    )
    for sample in samples:
        ours.add_scores(sample)
        theirs.add_scores(
            {
                key: upstream_scoring.Score(*value)
                for key, value in sample.items()
            }
        )
    np.random.seed(123)
    actual = ours.aggregate()
    np.random.seed(123)
    expected = theirs.aggregate()
    for key in actual:
        assert np.asarray(actual[key]) == pytest.approx(np.asarray(expected[key]))


def test_empty_bootstrap_aggregation_matches_upstream():
    assert scoring.BootstrapAggregator().aggregate() == {}


def test_fmeasure_matches_upstream():
    for precision, recall in [(0, 0), (1, 1), (0.25, 0.75), (0, 1)]:
        assert scoring.fmeasure(precision, recall) == upstream_scoring.fmeasure(
            precision, recall
        )
