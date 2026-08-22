"""GB10 (SM121) MoE backend fix for vLLM 0.27.2rc1 aarch64 nightly.

WHY
  torch.cuda.get_arch_list() in this image = [sm_80, sm_90, sm_100, sm_120].
  GB10 is sm_121. FlashInfer CUTLASS/TRTLLM MoE kernels are arch-exact SASS
  with no sm_121 cubin and no PTX fallback -> selecting them dies at the
  first decode with:
      CUDA error: no kernel image is available for execution on the device
  (Upstream already excludes FLASHINFER_B12X from NvFP4 auto-selection
   pending an "upstream CUTLASS SM121 MMA op guard"; the other FlashInfer
   MoE backends were simply never given the same guard.)

THE MIXED-CHECKPOINT CATCH-22
  poolside/Laguna-S-2.1-NVFP4 quantizes layers 0-39 experts to NVFP4 but
  leaves layers 40-47 experts BF16, producing two MoE method objects with
  DISJOINT legal backend sets, while --moe-backend is a single global flag:
      marlin -> ValueError: not supported for unquantized MoE
      triton -> ValueError: not supported for NvFP4 MoE
  Omitting the flag lets each oracle auto-select, but both then choose
  FlashInfer -> the no-kernel-image crash above.

FIX
  Launch with --moe-backend marlin (GB10-stable NvFP4 path, already proven
  on Nemotron-3-Super-120B) and alias marlin -> TRITON for the unquantized
  group only. Gated strictly to device capability (12, 1); on any other GPU
  this file is a no-op.

  Injected via sitecustomize so it also applies inside the SPAWNED
  EngineCore child process (a plain monkeypatch in the parent would be lost).
"""
import sys

_TARGET = "vllm.model_executor.layers.fused_moe.oracle.unquantized"


def _patch_unquantized(mod):
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if torch.cuda.get_device_capability(0) != (12, 1):
            return
    except Exception:
        return

    orig = mod.map_unquantized_backend

    def patched(runner_backend):
        if runner_backend == "marlin":
            print(
                "[sm121_moe_fix] unquantized MoE: aliasing marlin -> TRITON "
                "(no sm_121 FlashInfer cubin)",
                flush=True,
            )
            return mod.UnquantizedMoeBackend.TRITON
        return orig(runner_backend)

    mod.map_unquantized_backend = patched
    print("[sm121_moe_fix] patched map_unquantized_backend", flush=True)


class _Finder:
    """Post-import hook: patch the oracle right after it executes."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None and spec.loader is not None:
                break
        else:
            return None

        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            try:
                _patch_unquantized(module)
            except Exception as e:
                print(f"[sm121_moe_fix] patch FAILED: {e}", flush=True)

        loader.exec_module = exec_module
        return spec


if not any(type(f).__name__ == "_Finder" for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
