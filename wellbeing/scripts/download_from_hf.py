#!/usr/bin/env python3
"""Download per-model option files, generations, and results from the
companion public HF dataset (mmazeika/wellbeing-results) into this repo.

The dataset mirrors the wellbeing/ directory tree, so files land exactly
where the framework's scripts expect them. After this finishes you can run
the analyzers (e.g. `analysis/ai_wellbeing_index.py`) or any of the
`scripts/run_*.sh` drivers without re-running the upstream pipelines.

Requires:
    pip install huggingface_hub
    Authenticate: `huggingface-cli login`, or set `HF_TOKEN` env var.

Usage:
    python wellbeing/scripts/download_from_hf.py
    HF_TOKEN=<your-token> python wellbeing/scripts/download_from_hf.py
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub not installed; run: pip install huggingface_hub")

REPO_ID = "mmazeika/wellbeing-results"
WELLBEING_ROOT = Path(__file__).resolve().parent.parent  # the wellbeing/ subdir

# Don't fetch HF-side metadata files — they would clobber the framework's
# own README.md / .gitattributes at wellbeing/ root.
HF_METADATA_FILES = ["README.md", ".gitattributes"]

print(f"Downloading {REPO_ID} into {WELLBEING_ROOT} ...")
local_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(WELLBEING_ROOT),
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=HF_METADATA_FILES,
)
print(f"Done. Files placed under {local_path}")
