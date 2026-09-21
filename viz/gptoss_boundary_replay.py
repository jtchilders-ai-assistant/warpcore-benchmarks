#!/usr/bin/env python3
"""Extract and compare gpt-oss malformed-tool-call replay boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"}
            },
            "required": ["command"],
        },
    },
}


class BoundaryError(ValueError):
    """Raised when retained or captured evidence is incomplete or ambiguous."""


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def classify_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        choices = response["choices"]
        message = choices[0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BoundaryError("response lacks choices[0].message") from exc
    calls = message.get("tool_calls") or []
    classified = []
    for index, call in enumerate(calls):
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        item = {
            "index": index,
            "name": function.get("name") if isinstance(function, dict) else None,
            "arguments": arguments,
            "arguments_sha256": (
                hashlib.sha256(arguments.encode()).hexdigest() if isinstance(arguments, str) else None
            ),
            "valid_json": False,
            "error": None,
        }
        if not isinstance(arguments, str):
            item["error"] = "arguments is not a string"
        else:
            try:
                json.loads(arguments)
                item["valid_json"] = True
            except json.JSONDecodeError as exc:
                item["error"] = f"{exc.msg}: line {exc.lineno} column {exc.colno} (char {exc.pos})"
        classified.append(item)
    return {
        "response_sha256": canonical_digest(response),
        "finish_reason": choices[0].get("finish_reason"),
        "tool_call_count": len(classified),
        "all_arguments_valid_json": bool(classified) and all(x["valid_json"] for x in classified),
        "tool_calls": classified,
    }


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key != "extra"}


def build_replay_request(trajectory: dict[str, Any], malformed_index: int) -> dict[str, Any]:
    try:
        model_config = trajectory["info"]["config"]["model"]
        model_name = model_config["model_name"]
        kwargs = model_config["model_kwargs"]
        messages = trajectory["messages"]
    except (KeyError, TypeError) as exc:
        raise BoundaryError("trajectory lacks model configuration or messages") from exc
    prefix = "hosted_vllm/"
    if not isinstance(model_name, str) or not model_name.startswith(prefix):
        raise BoundaryError("only hosted_vllm model names are replayable")
    served_model = model_name[len(prefix) :]
    if not served_model:
        raise BoundaryError("hosted_vllm model name must have a nonempty served-model suffix")
    if not isinstance(messages, list) or malformed_index <= 0 or malformed_index >= len(messages):
        raise BoundaryError("malformed response index is outside the trajectory")
    request = {
        "model": served_model,
        "messages": [_clean_message(message) for message in messages[:malformed_index]],
        "tools": [BASH_TOOL],
    }
    for key in ("temperature", "max_tokens", "parallel_tool_calls"):
        if key not in kwargs:
            raise BoundaryError(f"trajectory model configuration lacks {key}")
        request[key] = kwargs[key]
    return request


def extract_replay_fixture(archive: Path, instance_id: str) -> dict[str, Any]:
    member_name = f"{instance_id}/{instance_id}.traj.json"
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.name == member_name]
        if len(members) != 1:
            raise BoundaryError(
                f"trajectory member must occur exactly once: {member_name} (found {len(members)})"
            )
        member = members[0]
        if not member.isfile():
            raise BoundaryError(f"trajectory is not a regular file: {member_name}")
        source = bundle.extractfile(member)
        if source is None:
            raise BoundaryError(f"trajectory is not a regular file: {member_name}")
        trajectory = json.load(source)
    candidates = []
    for index, message in enumerate(trajectory.get("messages", [])):
        extra = message.get("extra") if isinstance(message, dict) else None
        response = extra.get("response") if isinstance(extra, dict) else None
        if not isinstance(response, dict):
            continue
        classification = classify_response(response)
        if classification["tool_call_count"] and not classification["all_arguments_valid_json"]:
            candidates.append((index, response, classification))
    if not candidates:
        raise BoundaryError(f"no malformed retained response in {instance_id}")
    malformed_index, response, classification = candidates[0]
    request = build_replay_request(trajectory, malformed_index)
    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "source_member": member_name,
        "source_message_index": malformed_index,
        "request": request,
        "request_sha256": canonical_digest(request),
        "source_response": response,
        "source_classification": classification,
    }


def reassemble_sse(raw: bytes) -> dict[str, Any]:
    messages: dict[int, dict[str, Any]] = {}
    finish_reasons: dict[int, Any] = {}
    response_id = None
    saw_done = False
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BoundaryError(f"invalid SSE JSON at line {line_number}: {exc}") from exc
        response_id = response_id or event.get("id")
        for choice in event.get("choices") or []:
            choice_index = choice.get("index", 0)
            if not isinstance(choice_index, int) or not 0 <= choice_index < 32:
                raise BoundaryError("streamed choice index must be an integer in [0, 31]")
            message = messages.setdefault(
                choice_index, {"role": "assistant", "content": None, "tool_calls": []}
            )
            delta = choice.get("delta") or {}
            if delta.get("role") is not None:
                message["role"] = delta["role"]
            if delta.get("content") is not None:
                message["content"] = (message.get("content") or "") + delta["content"]
            for reasoning_key in ("reasoning", "reasoning_content"):
                if delta.get(reasoning_key) is not None:
                    message[reasoning_key] = (
                        message.get(reasoning_key) or ""
                    ) + delta[reasoning_key]
            for fragment in delta.get("tool_calls") or []:
                call_index = fragment.get("index")
                if not isinstance(call_index, int) or call_index < 0:
                    raise BoundaryError("streamed tool call lacks a nonnegative integer index")
                if call_index > len(message["tool_calls"]):
                    raise BoundaryError("streamed tool-call indices must be contiguous")
                if call_index == len(message["tool_calls"]):
                    message["tool_calls"].append(
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                call = message["tool_calls"][call_index]
                if fragment.get("id") is not None:
                    call["id"] = fragment["id"]
                if fragment.get("type") is not None:
                    call["type"] = fragment["type"]
                function = fragment.get("function") or {}
                if function.get("name") is not None:
                    call["function"]["name"] += function["name"]
                if function.get("arguments") is not None:
                    call["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason") is not None:
                finish_reasons[choice_index] = choice["finish_reason"]
    if not saw_done:
        raise BoundaryError("SSE stream lacks [DONE]")
    if not messages:
        raise BoundaryError("SSE stream contains no choices")
    return {
        "id": response_id,
        "object": "chat.completion",
        "choices": [
            {
                "index": index,
                "finish_reason": finish_reasons.get(index),
                "message": messages[index],
            }
            for index in sorted(messages)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = extract_replay_fixture(args.archive, args.instance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} request_sha256={fixture['request_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
