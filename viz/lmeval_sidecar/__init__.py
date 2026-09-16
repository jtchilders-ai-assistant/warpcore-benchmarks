# viz/lmeval_sidecar/__init__.py
# Auto-registers the sidecar model when this package is imported.
# Also exports version assertion for use by the runner.
from .capture import SidecarChatCompletion, assert_lm_eval_version  # noqa: F401
