"""Benchmarks against upstream rouge-score 0.1.2 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")

sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.curdir) != os.path.abspath(PYTHON_DIR)
]
from rouge_score import rouge_scorer as upstream_rouge_scorer

for module_name in list(sys.modules):
    if module_name == "rouge_score" or module_name.startswith("rouge_score."):
        del sys.modules[module_name]
sys.path.insert(0, PYTHON_DIR)

from rouge_score import rouge_scorer


def best_time(function, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def make_text(length, offset=0):
    vocabulary = [f"token{i}" for i in range(4096)]
    return " ".join(vocabulary[(i * 37 + offset) % len(vocabulary)] for i in range(length))


def benchmark(name, rouge_types, target, prediction, repeats=3):
    ours = rouge_scorer.RougeScorer(rouge_types)
    theirs = upstream_rouge_scorer.RougeScorer(rouge_types)
    ours.score(target, prediction)
    theirs.score(target, prediction)
    mojo_seconds = best_time(lambda: ours.score(target, prediction), repeats)
    upstream_seconds = best_time(lambda: theirs.score(target, prediction), repeats)
    assert ours.score(target, prediction) == theirs.score(target, prediction)
    return name, mojo_seconds, upstream_seconds


def main():
    cases = [
        benchmark(
            "ROUGE-1, 200k tokens",
            ["rouge1"],
            make_text(200_000),
            make_text(200_000, 11),
        ),
        benchmark(
            "ROUGE-2, 200k tokens",
            ["rouge2"],
            make_text(200_000),
            make_text(200_000, 11),
        ),
        benchmark(
            "ROUGE-1/2 together, 200k",
            ["rouge1", "rouge2"],
            make_text(200_000),
            make_text(200_000, 11),
        ),
        benchmark(
            "ROUGE-L, 2,500 tokens",
            ["rougeL"],
            make_text(2_500),
            make_text(2_500, 11),
        ),
        benchmark(
            "ROUGE-Lsum, 20x100 tokens",
            ["rougeLsum"],
            "\n".join(make_text(100, i * 7) for i in range(20)),
            "\n".join(make_text(100, i * 7 + 11) for i in range(20)),
        ),
    ]

    machine = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    machine = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    print(f"Machine: {machine}; Python {platform.python_version()}")
    print()
    print("| case | mojo-rouge-score | rouge-score 0.1.2 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_seconds, upstream_seconds in cases:
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {upstream_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
