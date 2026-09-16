#!/usr/bin/env python3
"""viz/lmeval_sidecar_runner.py — Production lm-eval wrapper with sidecar capture.

This script replaces `python -m lm_eval` as the harness entrypoint.
It:
  1. Asserts lm_eval == 0.4.12 (fail closed otherwise).
  2. Adds the viz/ directory to sys.path so lmeval_sidecar is importable.
  3. Imports lmeval_sidecar (triggers @register_model("sidecar-chat-completions")).
  4. Calls lm_eval CLI (passes through all sys.argv).

Caching is explicitly disabled: the model class rejects cache args at construction time.
Never passes credentials in argv.
LMEVAL_SIDECAR_PATH must be set in the environment by the caller.

Usage (exactly replaces `python -m lm_eval`):
    LMEVAL_SIDECAR_PATH=/run/raw/response_metadata.jsonl \\
    OPENAI_API_KEY=dummy \\
    python viz/lmeval_sidecar_runner.py \\
        --model sidecar-chat-completions \\
        --model_args "model=...,base_url=..." \\
        ...
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# 1. Add viz/ to sys.path so lmeval_sidecar package is importable
# ---------------------------------------------------------------------------
_VIZ_DIR = pathlib.Path(__file__).parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

# ---------------------------------------------------------------------------
# 2. Assert version (fail closed before doing anything else)
# ---------------------------------------------------------------------------
from lmeval_sidecar.capture import assert_lm_eval_version  # noqa: E402
assert_lm_eval_version()

# ---------------------------------------------------------------------------
# 3. Register the sidecar model with lm_eval BEFORE get_model is called
# ---------------------------------------------------------------------------
import lmeval_sidecar  # noqa: F401, E402 — triggers @register_model

# ---------------------------------------------------------------------------
# 4. Hand off to lm_eval CLI (all argv forwarded)
# ---------------------------------------------------------------------------
from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
