"""Regression tests for the production SWE-bench generation/tool-call preflight."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
_VIZ = _REPO / "viz"
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

import run_swebench  # noqa: E402


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _SubprocessResult:
    returncode = 0
    stdout = "x86_64"
    stderr = ""


class TestProductionGenerationToolProbe(unittest.TestCase):
    def test_v2_cli_skips_prestate_qualification_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            fake_runner = Mock()
            fake_runner.run.return_value = 0
            with patch(
                "swebench_qualification.verify_qualification_for_launch",
                side_effect=AssertionError("v2 must not consult qualification"),
            ), patch("create_campaign.create_campaign", return_value=run_dir), patch(
                "run_swebench.SwebenchRunner", return_value=fake_runner
            ):
                rc = run_swebench.main([
                    "--suite", str(_REPO / "suite" / "warpcore-v2.yaml"),
                    "--adapter", str(_REPO / "adapters" / "ornith-1.5-35b-a3b.yaml"),
                    "--endpoint", "http://endpoint.example/v1",
                    "--run-id", "v2-no-qualification",
                    "--repo", str(_REPO),
                    "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                    "--allow-no-screen",
                ])
            self.assertEqual(rc, 0)
            fake_runner.run.assert_called_once_with()

    def test_v2_cli_preflight_failure_blocks_generation(self):
        run_id = "v2-preflight-failure-test"
        run_dir = (
            _REPO / "results" / "ornith-1.5-35b-a3b" / "runs" /
            "warpcore-v2" / "swebench" / run_id
        )
        if run_dir.exists():
            raise AssertionError(f"test residue exists: {run_dir}")
        run_dir.mkdir(parents=True)
        try:
            status = {
                "schema_version": 1,
                "run_id": run_id,
                "suite_id": "warpcore-v2",
                "execution_state": "planned",
                "lifecycle": "current",
                "history": [{
                    "state": "planned",
                    "timestamp": "2026-09-29T00:00:00Z",
                    "note": "test",
                }],
            }
            (run_dir / "status.json").write_text(json.dumps(status))
            generation = Mock(return_value=0)
            with patch(
                "create_campaign.create_campaign", return_value=run_dir
            ), patch.object(
                run_swebench.SwebenchRunner,
                "_run_preflight",
                return_value=run_swebench.EXIT_DEFECT,
            ), patch.object(
                run_swebench.SwebenchRunner,
                "_run_generation",
                generation,
            ):
                rc = run_swebench.main([
                    "--suite", str(_REPO / "suite" / "warpcore-v2.yaml"),
                    "--adapter", str(_REPO / "adapters" / "ornith-1.5-35b-a3b.yaml"),
                    "--endpoint", "http://endpoint.example/v1",
                    "--run-dir", str(run_dir),
                    "--repo", str(_REPO),
                    "--prompt-tokens", "gsm8k=500,ifeval=2000,gpqa_diamond=1000",
                    "--allow-no-screen",
                    "--resume",
                ])
            self.assertEqual(rc, run_swebench.EXIT_DEFECT)
            generation.assert_not_called()
        finally:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = pathlib.Path(self.tmp.name) / "run"
        self.runner = run_swebench.SwebenchRunner(
            suite_path=_REPO / "suite" / "warpcore-v2.yaml",
            adapter_path=_REPO / "adapters" / "ornith-1.5-35b-a3b.yaml",
            endpoint="http://endpoint.example/v1",
            run_dir=self.run_dir,
            repo=_REPO,
            allow_no_screen=True,
            prompt_token_maxima={"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, chat_payload: dict, mini_version: str = "2.4.6"):
        models = {"data": [{"id": "ornith-ai/Ornith-1.5-35B-A3B-FP8"}]}
        calls = []

        def urlopen(req, **_kwargs):
            calls.append(req)
            if req.full_url.endswith("/models"):
                return _Response(models)
            if req.full_url.endswith("/chat/completions"):
                return _Response(chat_payload)
            raise AssertionError(req.full_url)

        with patch.object(
            self.runner,
            "_probe_mini_swe_agent_version",
            return_value=(0, mini_version),
        ), patch("subprocess.run", return_value=_SubprocessResult()), patch(
            "urllib.request.urlopen", side_effect=urlopen
        ):
            rc = self.runner._run_preflight()
        return rc, calls

    def test_constructor_rejects_invalid_explicit_interpreter(self):
        with self.assertRaisesRegex(ValueError, "python_executable"):
            run_swebench.SwebenchRunner(
                suite_path=_REPO / "suite" / "warpcore-v2.yaml",
                adapter_path=_REPO / "adapters" / "ornith-1.5-35b-a3b.yaml",
                endpoint="http://endpoint.example/v1",
                run_dir=self.run_dir,
                repo=_REPO,
                python_executable="/definitely/missing/python",
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000},
            )

    def test_constructor_preserves_interpreter_symlink_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            target = tmp_path / "python-target"
            target.write_text("#!/bin/sh\nexit 0\n")
            target.chmod(0o700)
            venv_python = tmp_path / "venv-python"
            venv_python.symlink_to(target)
            runner = run_swebench.SwebenchRunner(
                suite_path=_REPO / "suite" / "warpcore-v2.yaml",
                adapter_path=_REPO / "adapters" / "ornith-1.5-35b-a3b.yaml",
                endpoint="http://endpoint.example/v1",
                run_dir=self.run_dir,
                repo=_REPO,
                python_executable=str(venv_python),
                allow_no_screen=True,
                prompt_token_maxima={"gsm8k": 500, "ifeval": 2000, "gpqa_diamond": 1000},
            )
            self.assertEqual(runner._python_executable, str(venv_python.absolute()))
            self.assertNotEqual(runner._python_executable, str(target.resolve()))

    def test_generation_argv_uses_explicit_pinned_interpreter(self):
        pinned = "/opt/pinned/bin/python"
        self.runner._python_executable = pinned
        argv = self.runner._build_generation_argv("/tmp/config.yaml", self.run_dir / "raw")
        self.assertEqual(argv[0], pinned)

    def test_preflight_rejects_wrong_mini_swe_agent_version(self):
        payload = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{
                    "function": {"name": "bash", "arguments": '{"command":"true"}'},
                }]},
            }]
        }
        rc, _calls = self._run(payload, mini_version="9.9.9")
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)

    def test_preflight_inconclusive_when_mini_version_probe_fails(self):
        payload = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{
                    "function": {"name": "bash", "arguments": '{"command":"true"}'},
                }]},
            }]
        }
        with patch.object(
            self.runner,
            "_probe_mini_swe_agent_version",
            return_value=(run_swebench.EXIT_INCONCLUSIVE, ""),
        ):
            rc = self.runner._run_preflight()
        self.assertEqual(rc, run_swebench.EXIT_INCONCLUSIVE)

    def test_preflight_image_cache_uses_pinned_interpreter(self):
        pinned = "/opt/pinned/bin/python"
        self.runner._python_executable = pinned
        payload = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{
                    "function": {"name": "bash", "arguments": '{"command":"true"}'},
                }]},
            }]
        }
        models = {"data": [{"id": "ornith-ai/Ornith-1.5-35B-A3B-FP8"}]}
        subprocess_argv = []

        def run(argv, **_kwargs):
            subprocess_argv.append(argv)
            return _SubprocessResult()

        def urlopen(req, **_kwargs):
            return _Response(models if req.full_url.endswith("/models") else payload)

        with patch.object(
            self.runner, "_probe_mini_swe_agent_version", return_value=(0, "2.4.6")
        ), patch("subprocess.run", side_effect=run), patch(
            "urllib.request.urlopen", side_effect=urlopen
        ):
            self.assertEqual(self.runner._run_preflight(), 0)
        image_probe = next(argv for argv in subprocess_argv if "swebench_preflight.py" in " ".join(argv))
        self.assertEqual(image_probe[0], pinned)

    def test_preflight_requires_real_parsed_bash_tool_call(self):
        payload = {
            "choices": [{"finish_reason": "stop", "message": {"content": "hello", "tool_calls": []}}]
        }
        rc, calls = self._run(payload)
        self.assertEqual(rc, run_swebench.EXIT_DEFECT)
        self.assertTrue(any(r.full_url.endswith("/chat/completions") for r in calls))

    def test_preflight_accepts_valid_bash_tool_call(self):
        payload = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"printf SWEBENCH_PREFLIGHT_OK"}'},
                    }],
                },
            }]
        }
        rc, calls = self._run(payload)
        self.assertEqual(rc, 0)
        chat = next(r for r in calls if r.full_url.endswith("/chat/completions"))
        body = json.loads(chat.data)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["tools"][0]["function"]["name"], "bash")


if __name__ == "__main__":
    unittest.main()
