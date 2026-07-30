"""ctypes bindings for the Mojo ROUGE kernels."""

from __future__ import annotations

import ctypes
import os
import operator

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_ROUGE_SCORE_LIB",
    os.path.join(ROOT, "dist", "libmojo-rouge-score.so"),
)

I = ctypes.c_int64

_SIGNATURES = {
    "mrs_unigram_overlap": ([I, I, I, I, I], I),
    "mrs_ngram_overlap": ([I, I, I, I, I, I, I, I], I),
    "mrs_lcs_length": ([I, I, I, I, I], I),
    "mrs_lcs_indices": ([I, I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            raise RuntimeError(
                f"Mojo library not found at {LIB_PATH}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def _length(value, name: str) -> int:
    length = operator.index(value)
    if length < 0 or length > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside the int64 range")
    return length


def _array(array: np.ndarray, name: str, *, writable: bool = False) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.dtype != np.dtype(np.int64):
        raise TypeError(f"{name} must have dtype int64")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return array


def _addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def _checked_result(name: str, result: int, maximum: int) -> int:
    result = int(result)
    if result < 0 or result > maximum:
        raise RuntimeError(f"{name} kernel returned invalid result {result}")
    return result


def unigram_overlap(
    target: np.ndarray, prediction: np.ndarray, counts: np.ndarray
) -> int:
    target = _array(target, "target")
    prediction = _array(prediction, "prediction")
    counts = _array(counts, "counts", writable=True)
    target_len = _length(target.size, "target length")
    prediction_len = _length(prediction.size, "prediction length")
    if target_len and (target.min() < 0 or target.max() >= counts.size):
        raise ValueError("target token IDs are outside the counts buffer")
    if prediction_len and (prediction.min() < 0 or prediction.max() >= counts.size):
        raise ValueError("prediction token IDs are outside the counts buffer")
    result = lib().mrs_unigram_overlap(
        _addr(target), target_len, _addr(prediction), prediction_len, _addr(counts)
    )
    return _checked_result("unigram overlap", result, min(target_len, prediction_len))


def ngram_overlap(
    target: np.ndarray,
    prediction: np.ndarray,
    n: int,
    starts: np.ndarray,
    counts: np.ndarray,
) -> int:
    target = _array(target, "target")
    prediction = _array(prediction, "prediction")
    starts = _array(starts, "starts", writable=True)
    counts = _array(counts, "counts", writable=True)
    n = _length(n, "n")
    if n == 0:
        raise ValueError("n must be positive")
    if starts.size != counts.size:
        raise ValueError("starts and counts must have equal lengths")
    capacity = _length(starts.size, "hash-table capacity")
    if capacity == 0 or capacity & (capacity - 1):
        raise ValueError("hash-table capacity must be a nonzero power of two")
    target_len = _length(target.size, "target length")
    prediction_len = _length(prediction.size, "prediction length")
    result = lib().mrs_ngram_overlap(
        _addr(target),
        target_len,
        _addr(prediction),
        prediction_len,
        n,
        _addr(starts),
        _addr(counts),
        capacity,
    )
    maximum = min(max(target_len - n + 1, 0), max(prediction_len - n + 1, 0))
    return _checked_result("ngram overlap", result, maximum)


def lcs_length(
    target: np.ndarray, prediction: np.ndarray, row: np.ndarray
) -> int:
    target = _array(target, "target")
    prediction = _array(prediction, "prediction")
    row = _array(row, "row", writable=True)
    if row.size != prediction.size + 1:
        raise ValueError("row length must equal prediction length plus one")
    target_len = _length(target.size, "target length")
    prediction_len = _length(prediction.size, "prediction length")
    result = lib().mrs_lcs_length(
        _addr(target), target_len, _addr(prediction), prediction_len, _addr(row)
    )
    return _checked_result("LCS length", result, min(target_len, prediction_len))


def lcs_indices(
    ref: np.ndarray,
    candidate: np.ndarray,
    table: np.ndarray,
    indices: np.ndarray,
) -> int:
    ref = _array(ref, "ref")
    candidate = _array(candidate, "candidate")
    table = _array(table, "table", writable=True)
    indices = _array(indices, "indices", writable=True)
    ref_len = _length(ref.size, "reference length")
    candidate_len = _length(candidate.size, "candidate length")
    if table.size != (ref_len + 1) * (candidate_len + 1):
        raise ValueError("table has the wrong length")
    if indices.size < min(ref_len, candidate_len):
        raise ValueError("indices buffer is too short")
    result = lib().mrs_lcs_indices(
        _addr(ref),
        ref_len,
        _addr(candidate),
        candidate_len,
        _addr(table),
        _addr(indices),
    )
    return _checked_result("LCS indices", result, min(ref_len, candidate_len))
