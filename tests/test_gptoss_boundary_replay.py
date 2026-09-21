import io
import json
import tarfile
from pathlib import Path

import pytest

from viz.gptoss_boundary_replay import (
    BoundaryError,
    build_replay_request,
    classify_response,
    extract_replay_fixture,
    reassemble_sse,
)


def _response(arguments: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": arguments},
                }],
            },
        }],
    }


def _trajectory() -> dict:
    malformed = _response('{"command":"printf x"]}')
    return {
        "instance_id": "example__repo-1",
        "info": {
            "config": {
                "model": {
                    "model_name": "hosted_vllm/openai/gpt-oss-120b",
                    "model_kwargs": {
                        "api_base": "http://127.0.0.1:18000/v1",
                        "api_key": "warpcore",
                        "drop_params": True,
                        "max_tokens": 32768,
                        "parallel_tool_calls": True,
                        "temperature": 0,
                        "timeout": 1800,
                    },
                }
            }
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }],
                "extra": {"response": _response('{"command":"pwd"}')},
            },
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": "<returncode>0</returncode>",
                "extra": {"raw_output": "secret transient output"},
            },
            {
                "role": "user",
                "content": "Tool call error",
                "extra": {"interrupt_type": "FormatError", "response": malformed},
            },
        ],
    }


def test_extracts_exact_prefix_before_first_malformed_response(tmp_path: Path):
    archive = tmp_path / "trajectories.tar.gz"
    raw = json.dumps(_trajectory()).encode()
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("example__repo-1/example__repo-1.traj.json")
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))

    fixture = extract_replay_fixture(archive, "example__repo-1")

    assert fixture["source_message_index"] == 4
    assert fixture["request"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-0",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-0", "content": "<returncode>0</returncode>"},
    ]
    assert fixture["request"]["model"] == "openai/gpt-oss-120b"
    assert fixture["request"]["temperature"] == 0
    assert fixture["request"]["max_tokens"] == 32768
    assert fixture["request"]["parallel_tool_calls"] is True
    assert fixture["source_response"] == _trajectory()["messages"][4]["extra"]["response"]
    assert fixture["source_classification"]["all_arguments_valid_json"] is False


def test_build_request_rejects_unreplayable_model_name():
    traj = _trajectory()
    traj["info"]["config"]["model"]["model_name"] = "anthropic/other"
    with pytest.raises(BoundaryError, match="hosted_vllm"):
        build_replay_request(traj, 4)


def test_classifies_missing_and_malformed_tool_arguments_fail_closed():
    assert classify_response(_response('{"command":"pwd"}'))["all_arguments_valid_json"] is True
    assert classify_response(_response('{"command":"pwd"]}'))["all_arguments_valid_json"] is False
    missing = _response('{"command":"pwd"}')
    del missing["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    result = classify_response(missing)
    assert result["all_arguments_valid_json"] is False
    assert result["tool_calls"][0]["error"] == "arguments is not a string"


def test_reassembles_streamed_tool_arguments_by_choice_and_call_index():
    events = [
        {"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call-1", "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"printf '},
        }]}}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": 'x"}'},
        }]}, "finish_reason": "tool_calls"}]},
    ]
    raw = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    assembled = reassemble_sse(raw.encode())

    assert assembled["choices"][0]["finish_reason"] == "tool_calls"
    assert assembled["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "bash", "arguments": '{"command":"printf x"}'
    }
    assert classify_response(assembled)["all_arguments_valid_json"] is True


def test_reassembler_rejects_invalid_sse_json():
    with pytest.raises(BoundaryError, match="invalid SSE JSON"):
        reassemble_sse(b"data: {not-json}\n\n")
