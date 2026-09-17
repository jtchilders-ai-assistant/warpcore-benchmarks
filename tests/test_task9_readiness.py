"""Task 9: the live Qwen serving profile is pinned for closure runs."""

from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).parent.parent
VIZ = REPO / "viz"
if str(VIZ) not in sys.path:
    sys.path.insert(0, str(VIZ))

from schema_helpers import SCHEMA_ADAPTER, validate_doc
from create_campaign import _image_digest


ADAPTER_PATH = REPO / "adapters" / "qwen3.6-35b-a3b.yaml"


def _adapter() -> dict:
    return yaml.safe_load(ADAPTER_PATH.read_text())


def test_live_qwen_adapter_is_canonical_and_schema_valid():
    adapter = _adapter()
    assert validate_doc(adapter, SCHEMA_ADAPTER) is None
    assert adapter["campaign_status"] == "canonical"
    assert "noncanonical_reason" not in adapter


def test_live_qwen_adapter_pins_observed_model_and_image_identity():
    adapter = _adapter()
    assert adapter["model"]["revision"] == "95a723d08a9490559dae23d0cff1d9466213d989"
    assert adapter["serving"]["image"] == (
        "sha256:6a5355182ae6aba054d02066ed6e8c4c6a5355a737dca56afdcb899dfe1acdd6"
    )


def test_live_qwen_adapter_matches_observed_effective_capacity():
    serving = _adapter()["serving"]
    assert serving["engine_version"] == "0.17.1rc1.dev96+g57431d823.d20260312"
    assert serving["max_model_len"] == 262144
    assert serving["gpu_memory_utilization"] == 0.8
    assert serving["max_num_seqs"] == 128
    assert serving["reasoning_parser"] == "qwen3"
    assert serving["tool_call_parser"] == "qwen3_xml"


def test_local_docker_image_id_is_preserved_in_campaign_manifest_digest():
    image_id = _adapter()["serving"]["image"]
    assert _image_digest(image_id) == image_id


def test_contract_cli_accepts_canonical_adapter_with_measured_prompt_evidence():
    result = subprocess.run(
        [
            "/usr/bin/python3",
            str(REPO / "viz" / "validate_suite.py"),
            str(REPO / "suite" / "warpcore-v1.yaml"),
            "--adapter",
            str(ADAPTER_PATH),
            "--prompt-tokens",
            "gsm8k=256,ifeval=373,gpqa_diamond=2808",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
