"""Engine flux-action (the default, decision 03): BFL's standalone SO-101 loader, no LeRobot.

The path verified in flux-action 830f1ce: ``flux_action.inference.so101.load_policy(repo, revision=...)``,
``set_text_encoder_offload(True)`` before ``.to(device)`` (the move then leaves Qwen on the CPU), then
``predict_action_chunk`` on an explicit window ``{"state": (1, 8, 6), "command_history": (1, 8, 6),
"images.scene": u1 (1, 8, 3, H, W), "images.wrist": ..., "task": [task]}``. It returns (1, 42, 6) absolute
commands: joint deltas integrated from ``command_history[:, -1]``, gripper absolute.

The shared encoders come from the checkpoint config (``video_vae_id``, ``text_encoder_id``, Hub specs
``repo_id[:file][@revision]``, resolved by models/runtime.py and models/text_encoder.py). The config's own
pin wins; ``base_revision`` pins a spec that has none, and a different pin is an error unless forced.

``compile`` is experimental: flux-action compiles only the released DROID geometry (prepare_inference), so
this wraps the SO-101 DiT forward in ``torch.compile`` the way LeRobot's compile_model does. If the first
warmup fails compiled, the engine goes back to eager and says so in the log and in hello.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .base import (
    Engine,
    capture_sampler_video,
    override_normalization,
    override_sampling,
    pin_revision,
    take_captured_video,
    window_order,
)


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _num(x):
    return int(x) if float(x).is_integer() else float(x)


class FluxActionEngine(Engine):
    name = "flux-action"

    def __init__(
        self,
        checkpoint: str,
        revision: str | None = None,
        base_revision: str | None = None,
        device: str = "cuda",
        compile: bool = False,
        offload_text_encoder: bool = True,
        force_base_revision: bool = False,
        guidance: tuple[float, float] | None = None,
        steps: int | None = None,
        norm_stats: dict | None = None,
        capture_video: bool = False,
    ):
        super().__init__()
        import flux_action
        import torch
        from flux_action.hub import resolve_policy_directory
        from flux_action.inference import so101
        from flux_action.processing.history import snapshot_indices

        self.torch = torch
        self.device = torch.device(device)
        if revision and Path(checkpoint).expanduser().is_dir():
            print(f"flux-action: {checkpoint} is a local directory, --revision ignored", flush=True)
            revision = None
        started = time.perf_counter()
        bundle = resolve_policy_directory(checkpoint, revision=revision)
        raw = json.loads((bundle / "config.json").read_text())
        print(f"flux-action: policy package {bundle}", flush=True)
        print(
            f"flux-action: checkpoint config encoders {raw.get('video_vae_id')} and {raw.get('text_encoder_id')}",
            flush=True,
        )
        vae_id, vae_note = pin_revision(raw.get("video_vae_id"), base_revision, force_base_revision)
        te_id, te_note = pin_revision(raw.get("text_encoder_id"), base_revision, force_base_revision)
        for note in (vae_note, te_note):
            if note:
                print(f"flux-action: {note}", flush=True)
        policy = self._load(so101, bundle, raw, vae_id, te_id)
        if offload_text_encoder:
            policy.set_text_encoder_offload(True)
        policy.to(self.device)
        self.policy = policy
        cfg = policy.config
        self.capture_video, self.last_video = bool(capture_video), None
        if capture_video:
            from flux_action.inference import sampling

            capture_sampler_video(sampling)
        for note in override_sampling(cfg, guidance, steps) + override_normalization(cfg, norm_stats):
            print(f"flux-action: {note}", flush=True)
        cfg.validate_inference()
        self.compiled = bool(compile)
        if compile:
            # Experimental (module docstring): an instance attribute, so nn.Module.__call__ uses it and
            # `del policy.dit.forward` restores the eager forward.
            policy.dit.forward = torch.compile(policy.dit.forward)
        self.snapshots = snapshot_indices(cfg)
        self.camera_keys = list(cfg.camera_keys)
        self.cameras = [key.removeprefix("images.") for key in self.camera_keys]
        problems = [
            f"{name} {have} (so101.v1 needs {want})"
            for name, have, want in (
                ("n_obs_steps", cfg.n_obs_steps, 8),
                ("snapshots", self.snapshots, [0, 7]),
                ("chunk_size", cfg.chunk_size, 42),
                ("action_dim", cfg.action_dim, 6),
                ("cameras", self.cameras, ["scene", "wrist"]),
            )
            if have != want
        ]
        if problems:
            raise RuntimeError("checkpoint does not match the wire protocol: " + "; ".join(problems))
        self.order = window_order(cfg.n_obs_steps, self.snapshots)
        guidance_action = cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
        self.meta = {
            "checkpoint": str(checkpoint),
            "revision": revision,
            "chunk": cfg.chunk_size,
            "exec": cfg.n_action_steps,
            "fps": _num(cfg.fps),
            "n_obs": cfg.n_obs_steps,
            "snapshots": list(self.snapshots),
            "cameras": self.cameras,
            "guidance": [float(cfg.guidance_scale), float(guidance_action)],
            "steps": cfg.num_inference_steps,
            "sampler": cfg.sampler,
            "sampler_shift": cfg.sampler_shift,
            "seed": cfg.inference_seed,
            "dtype": cfg.torch_dtype,
            "video_vae_id": vae_id,
            "text_encoder_id": te_id,
            "config_video_vae_id": raw.get("video_vae_id"),
            "config_text_encoder_id": raw.get("text_encoder_id"),
            "compile": "dit (experimental)" if compile else False,
            "offload_text_encoder": bool(offload_text_encoder),
            "device": str(self.device),
            "gpu": gpu,
            "torch": torch.__version__,
            "flux_action_commit": _git_commit(Path(flux_action.__file__).parent),
            "latent_hw": list(cfg.latent_hw),  # --save-latents: tokens reshape "(t h w) c" with t = n_tokens / (h w)
            "state_q01": list(cfg.state_normalization["q01"]),
            "state_q99": list(cfg.state_normalization["q99"]),
            "action_q01": list(cfg.action_normalization["q01"]),
            "action_q99": list(cfg.action_normalization["q99"]),
        }
        where = "CPU (offloaded)" if offload_text_encoder else str(self.device)
        print(
            f"flux-action: loaded in {time.perf_counter() - started:.1f} s; DiT and VAE on {self.device}"
            f"{f' ({gpu})' if gpu else ''}, text encoder on {where}",
            flush=True,
        )
        print(
            f"flux-action: sampling {cfg.sampler}, {cfg.num_inference_steps} steps, guidance "
            f"{cfg.guidance_scale}/{guidance_action}, shift {cfg.sampler_shift}, seed {cfg.inference_seed}, "
            f"{cfg.torch_dtype}, {'compiled DiT, experimental (the first plan compiles it)' if compile else 'eager'}",
            flush=True,
        )

    def warmup(self, task: str) -> None:
        """Engine.warmup; a compiled DiT that fails it is replaced by the eager forward and warmed again."""
        try:
            super().warmup(task)
        except Exception as e:
            if not self.compiled:
                raise
            error = f"{type(e).__name__}: {e}"[:500]
            print(f"flux-action: WARNING: the compiled DiT failed in warmup ({error}); falling back to eager", flush=True)
            del self.policy.dit.forward
            self.compiled = False
            self.first_plan_ms = None
            self.meta["compile"] = False
            self.meta["compile_failed"] = error
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            super().warmup(task)

    @staticmethod
    def _load(so101, bundle: Path, raw: dict, vae_id, te_id):
        if (vae_id, te_id) == (raw.get("video_vae_id"), raw.get("text_encoder_id")):
            return so101.load_policy(bundle)
        # load_policy builds its PolicyConfig from config.json; set the pinned encoder specs on the way
        # into the constructor, which loads the encoders.
        real = so101.FluxActionPolicy

        def build(config, **kwargs):
            config.video_vae_id, config.text_encoder_id = vae_id, te_id
            return real(config, **kwargs)

        so101.FluxActionPolicy = build
        try:
            return so101.load_policy(bundle)
        finally:
            so101.FluxActionPolicy = real

    def infer(self, state, cmd_hist, frames, task):
        torch, dev = self.torch, self.device
        batch = {
            "state": torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))[None].to(dev),
            "command_history": torch.from_numpy(np.ascontiguousarray(cmd_hist, dtype=np.float32))[None].to(dev),
            "task": [task],
        }
        for cam, key in zip(self.cameras, self.camera_keys, strict=True):
            window = np.ascontiguousarray(frames[cam][self.order].transpose(0, 3, 1, 2))  # (8, 3, H, W) uint8
            # uint8 as in offline.load_observation: the history profile normalizes the bytes itself.
            batch[key] = torch.from_numpy(window)[None].to(dev)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
            self.last_video = take_captured_video() if self.capture_video else None
        return actions[0].float().cpu().numpy()
