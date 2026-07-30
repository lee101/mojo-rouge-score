import os
import sys

PYTHON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
)

sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.curdir) != os.path.abspath(PYTHON_DIR)
]

from rouge_score import rouge_scorer as upstream_rouge_scorer
from rouge_score import scoring as upstream_scoring
from rouge_score import tokenize as upstream_tokenize
from rouge_score import tokenizers as upstream_tokenizers

for module_name in list(sys.modules):
    if module_name == "rouge_score" or module_name.startswith("rouge_score."):
        del sys.modules[module_name]

sys.path.insert(0, PYTHON_DIR)
