# Status 2026-10-17 15:45: first FLUX inference on an H100

**TL;DR.** The SO-101 policy runs. On a Runpod H100 in Paris: 0.58 s per action chunk (42 × 6) eager, 25.9 GB of GPU memory,
2 minutes to load. Nothing is compiled or tuned yet. AWS is still waiting on the GPU quota, so compute is on Runpod for now.

| | |
|---|---|
| Pod | `rmhs4cu2cejemv`, H100 80 GB HBM3, EU-FR-1, $3.99/h, running continuously |
| SSH | `ssh root@<pod-host> -p <ssh-port>` (alerest285, saadl27 ×2, joelleayoub keys). The port changes if the pod restarts |
| Disk | `/workspace` 187 GB, persists across stop. FLUX at `/workspace/flux-action`, weights at `/workspace/hf` |
| Env | `FLUX_DIR=/workspace/flux-action HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache` |
| Weights | `flux-3-action-so101` @0713a7a + `flux-3-action-base` encoders (gated, alerest's HF token on the pod) |

## Measured (infra/gpu/smoke-so101.sh, synthetic observation, eager)

- load 123 s · first plan 1.94 s · then p50 0.58 s, max 0.61 s · peak GPU 25.9 GB · output `(1, 42, 6)`, finite

## Next

1. T10 (the SO-101 server): try `prepare_inference(compile=True)` and FP8 to cut the 0.58 s
2. Ping from the Mac at the robot through the tunnel (round trip = network + server)
3. Runpod console admin for Saad and Joelle needs a Runpod team (alerest285, console only)
