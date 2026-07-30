"""Default text normalization compatible with rouge-score."""

from __future__ import annotations

import re

NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text, stemmer):
    text = text.lower()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    tokens = TOKEN_RE.findall(text)
    if stemmer:
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
    return tokens
