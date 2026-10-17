# FLUX 3 Action: facts we rely on

Collected 2026-10-17 from the sources below. FLUX code pinned at
`black-forest-labs/flux-action@830f1ce` (2026-10-02). Re-check a source before trusting
anything here that it does not support.

| Fact | Source |
|---|---|
| 7B world action model. Text encoder is frozen Qwen3-VL-4B; video VAE is frozen | blog |
| Released policies: `black-forest-labs/flux-3-action-droid`, `black-forest-labs/flux-3-action-so101`; shared encoders in `black-forest-labs/flux-3-action-base` | flux-action README, docs/setup.md |
| DROID packages: root (BF16, 4 steps + guidance), `variants/fp8r`, `variants/gd`, `variants/gd-fp8r`, `variants/sd` (1 step), `variants/sd-fp8r` | docs/setup.md |
| Requires Linux + NVIDIA GPU. Reference environment: H200, CUDA 12.8, Python 3.12, torch 2.10.0+cu128, transformers 5.16.1, NATTEN 0.21.6 (`+torch2100cu128` wheel, installed separately and again after every `uv sync`) | docs/setup.md |
| FFmpeg with an AV1 decoder is needed for the recorded-data example | docs/setup.md |
| A first inference needs no robot: download the public DROID sample (~800 MB), `prepare-droid`, `make_observation.py`, `flux-action infer` | docs/setup.md, docs/prepare.md |
| DROID reference settings: sampler `cosmos_unipc`, 4 steps, shift 5.0, guidance 4.0 / 1.0, `n_action_steps` 32 | docs/setup.md |
| Eager runs are reproducible. `--compile-dit` is faster but differs by ~0.015 rad between runs | docs/setup.md |
| Serving: `uv run --extra serve flux-action serve-robolab --checkpoint <ckpt> --port 8000 [--subfolder variants/gd-fp8r] [--log-dir ...]` | docs/setup.md |
| `--offload-text-encoder` keeps the text encoder on CPU between caption-cache misses | docs/setup.md |
| VRAM is **not published**. `report.json` records GPU memory, so we measure it | — |
| Weights license: FLUX Kommunity License v1.0. Code: Apache-2.0 | blog, GitHub |
| RoboLab-120 success: FLUX 3 Action 42.9 %, π0.5 28.0 %, GR00T N1.6 7.2 % | blog |
| LeRobot integration (policy class, PEFT recipes): https://huggingface.co/docs/lerobot/main/en/flux3 | flux-action README |

## Sources

- Blog: https://huggingface.co/blog/black-forest-labs/flux-3-action
- Code: https://github.com/black-forest-labs/flux-action (`docs/setup.md`, `docs/prepare.md`, `docs/so101-lora.md`, `docs/embodiments.md`)
- Weights: https://huggingface.co/collections/black-forest-labs/flux-3-action
- BFL docs: https://docs.bfl.ai/flux_3/flux3_action_overview
- RoboLab leaderboard: https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html
