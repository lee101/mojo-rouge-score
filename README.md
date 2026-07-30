# mojo-rouge-score

`mojo-rouge-score` is a Mojo port of the compute-heavy parts of Google's
[`rouge-score`](https://pypi.org/project/rouge-score/) package. It keeps the familiar
`rouge_score` Python API while moving exact n-gram multiset intersection and longest
common subsequence dynamic programming into a compiled Mojo shared library.

This is an API-compatible subset for programmatic scoring, especially when ROUGE-L is
applied to long documents. It is not an approximation: the test suite compares behavior
and numerical results directly with `rouge-score==0.1.2`.

## Coverage

Implemented:

- `rouge_score.rouge_scorer.RougeScorer`
- `rouge1` through `rouge9`, `rougeL`, and `rougeLsum` (each is covered by a parity test)
- default tokenization, optional Porter stemming, and custom tokenizer objects
- newline-separated summaries and upstream's `split_summaries=True` behavior
- `score`, `score_multi`, and the upstream LCS/n-gram helper functions
- `Score`, `AggregateScore`, `BaseScorer`, `BootstrapAggregator`, and `fmeasure`

Not implemented:

- the `rouge_score.rouge` command-line application
- `rouge_score.io` file-pattern and CSV helpers
- legacy pyrouge file conversion
- metrics not provided by upstream 0.1.2, such as ROUGE-W and skip-bigram ROUGE
- platforms other than Linux, GPU execution, and a prebuilt binary wheel

The compatibility target is the Python API in upstream `rouge-score==0.1.2`, not every
file shipped in that distribution. Private scorer helpers used by downstream code are
also parity-tested, but only the items in the implemented list above are supported.

## Install

The repository pins the Mojo nightly used to build the library:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-rouge-score.so`. The Pixi environment sets
`PYTHONPATH=python`, so imports resolve to this implementation. Mojo is required to build
from source; this repository does not currently publish a standalone pip-installable
binary.

## Usage

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)
scores = scorer.score(
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown dog jumps on the log.",
)

print(scores["rougeL"])
# Score(precision=0.625, recall=0.5555555555555556,
#       fmeasure=0.5882352941176471)
```

Save the example as a script and run it with `pixi run python script.py`. The example
above was executed on the same environment used for the test and benchmark runs.

Run the validation suite with:

```bash
pixi run build
pixi run test
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz using
Python 3.13.14. Times include tokenization, Python-to-integer token encoding, FFI calls,
and score construction. Each cell is the best of three runs on identical input.

| case | mojo-rouge-score | rouge-score 0.1.2 | speedup |
| --- | ---: | ---: | ---: |
| ROUGE-1, 200k tokens | 191.27 ms | 1053.19 ms | 5.51x |
| ROUGE-2, 200k tokens | 186.86 ms | 721.68 ms | 3.86x |
| ROUGE-1/2 together, 200k | 180.89 ms | 889.62 ms | 4.92x |
| ROUGE-L, 2,500 tokens | 21.77 ms | 1662.39 ms | 76.35x |
| ROUGE-Lsum, 20x100 tokens | 21.75 ms | 603.67 ms | 27.76x |

These are end-to-end timings, not kernel-only timings. Run `pixi run bench` to reproduce
the cases; normal scheduler and system-load variation applies.

## How it works

Python performs upstream-compatible normalization and stemming, interns the combined
target and prediction vocabulary with a Python dictionary, and creates contiguous
`int64` NumPy arrays. Checked wrappers validate dtype, shape, contiguity, writeability,
lengths, and output capacity before buffers cross the C ABI. The arrays remain referenced
for the duration of each synchronous `ctypes` call; the shared library never stores or
owns their memory.

ROUGE-1 uses a compact caller-allocated direct-count table. Longer ROUGE-N metrics use
open addressing; hashes choose candidate slots, but every collision is resolved by
comparing the complete n-gram token sequence, so scoring remains exact. Counts are
decremented while scanning prediction n-grams to compute the multiset intersection
without constructing Python tuples.

ROUGE-L uses a two-row-equivalent in-place DP buffer, reducing auxiliary storage from
quadratic to linear when only the LCS length is needed. ROUGE-Lsum uses a full contiguous
table only for each sentence pair whose LCS indices must be backtracked; those indices are
unioned with the same tie-breaking and double-count prevention as upstream.

MIT.
