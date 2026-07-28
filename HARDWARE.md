# Hardware — Warpcore (NVIDIA DGX Spark / GB10)

Specs captured 2026-07-27 from the live host.

| Component | Detail |
| --------- | ------ |
| Host alias | `warpcore` (ANL network, `140.221.17.30`) |
| Machine | NVIDIA DGX Spark |
| GPU | **NVIDIA GB10** (Grace-Blackwell superchip), driver **580.142** |
| GPU memory | Unified LPDDR5X, reported `N/A` by `nvidia-smi` (normal for GB10). ~128 GB total unified; **~110 GB usable** ceiling for model weights + KV cache in practice |
| CPU | ARM **Cortex-X925**, `aarch64`, 20 cores |
| System memory | 121 GiB total (unified with GPU) |
| OS | Ubuntu 24.04.4 LTS, kernel `6.17.0-1014-nvidia` |
| Serving stack | vLLM in Docker, container `vllm_prebuilt` (image `eugr/spark-vllm:latest`) |
| vLLM version | `0.23.1rc1.dev961+gbc6fbf472.d20260708` (nightly arm64) |

## Notes

- The GB10 is a **single unified-memory** device: GPU and CPU share the LPDDR5X pool, so `nvidia-smi`
  memory queries return `N/A` — this is expected, not a fault.
- The production container runs the **MARLIN** MoE backend + TRITON attention path (the crash-fixed
  build), not CUTLASS/FlashInfer. See ISSUES.md for why this matters.
- Memory sizing rule for "will model X fit": dense = `params_B × bytes_per_param + ~5GB KV ≤ 110GB`
  (BF16=2, FP8=1, INT4=0.5 bytes/param); MoE = ALL expert weights must fit (`total_params × bytes`,
  active-param count is irrelevant for sizing).
- No systemd unit / auto-restart for the vLLM server: if it dies it stays dead until manually
  restarted via the `~/spark-vllm-docker` recipe tooling.
