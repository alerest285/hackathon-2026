"""Engine lerobot: LeRobot main (146d847) Flux3Policy with its saved processors. The A/B option for
``compile_model`` and the only path that loads a PEFT/LoRA adapter (milestone 4).

Stateless call, as in tests/policies/flux3/test_flux3_lora.py: raw ``observation.state`` (1, 8, 6),
``observation.command_history`` (1, 8, 6) and ``observation.images.<cam>`` uint8 (1, 8, 3, H, W) go
through the preprocessor, whose history step takes the explicit-window branch (``output_anchor =
command_history[:, -1]``, its deques untouched); ``predict_action_chunk`` returns normalized deltas and
the postprocessor integrates them from that anchor (gripper absolute).
"""

import subprocess
import time
from pathlib import Path

import numpy as np

from .base import Engine, override_sampling, pin_revision, snapshot_indices, window_order


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


class LeRobotEngine(Engine):
    name = "lerobot"

    def __init__(
        self,
        checkpoint: str,
        revision: str | None = None,
        base_revision: str | None = None,
        device: str = "cuda",
        compile: bool = False,
        adapter: str | None = None,
        offload_text_encoder: bool = True,
        force_base_revision: bool = False,
        guidance: tuple[float, float] | None = None,
        steps: int | None = None,
        norm_stats: dict | None = None,
    ):
        super().__init__()
        import lerobot
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3 import Flux3Config, Flux3Policy
        from lerobot.policies.flux3.processor_flux3 import ObservationHistoryNormalizerProcessorStep

        self.torch = torch
        checkpoint = str(Path(checkpoint).expanduser()) if checkpoint.startswith("~") else checkpoint
        if adapter and adapter.startswith("~"):
            adapter = str(Path(adapter).expanduser())
        if revision and Path(checkpoint).is_dir():
            print(f"lerobot: {checkpoint} is a local directory, --revision ignored", flush=True)
            revision = None
        started = time.perf_counter()
        base, base_rev, settings, settings_rev = checkpoint, revision, checkpoint, revision
        peft_config = None
        if adapter:
            # Like lerobot.policies.factory.make_policy with use_peft: the adapter names its base, and
            # lerobot-train saves the policy config and processors next to the adapter weights.
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(adapter)
            named, pinned = peft_config.base_model_name_or_path, peft_config.revision
            if named and named != checkpoint:
                # A commit of --checkpoint means nothing for another repo: the adapter's own pin, as make_policy.
                base, base_rev = named, pinned
                if not pinned and not Path(named).is_dir():
                    print(f"lerobot: WARNING: adapter base {named} has no pinned revision; loading its main", flush=True)
            elif pinned and pinned != revision:
                if revision:
                    print(f"lerobot: WARNING: the adapter pins {checkpoint}@{pinned}; using it, not --revision "
                          f"{revision[:8]}", flush=True)
                base_rev = pinned
            if base_rev and Path(base).is_dir():
                base_rev = None
            settings, settings_rev = adapter, None
            print(f"lerobot: adapter {adapter} on base {base}{f'@{base_rev}' if base_rev else ''}", flush=True)
        cfg = Flux3Config.from_pretrained(settings, revision=settings_rev)
        cfg.device = device
        cfg.compile_model = bool(compile)  # read by the constructor (encoders) and by _inference_dit
        config_vae, config_te = cfg.video_vae_id, cfg.text_encoder_id
        print(f"lerobot: checkpoint config encoders {config_vae} and {config_te}", flush=True)
        cfg.video_vae_id, vae_note = pin_revision(cfg.video_vae_id, base_revision, force_base_revision)
        cfg.text_encoder_id, te_note = pin_revision(cfg.text_encoder_id, base_revision, force_base_revision)
        for note in (vae_note, te_note):
            if note:
                print(f"lerobot: {note}", flush=True)
        policy = Flux3Policy.from_pretrained(base, config=cfg, revision=base_rev)
        if adapter:
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, adapter, config=peft_config, is_trainable=False)
            policy.eval()
        inner = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        if offload_text_encoder:
            inner.frozen.text_encoder.cpu()  # docs/source/flux3.mdx: uncached captions then encode on the CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.policy, self.config = policy, inner.config
        self.pre, self.post = make_pre_post_processors(
            inner.config,
            pretrained_path=settings,
            pretrained_revision=settings_rev,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        cfg = inner.config
        for note in override_sampling(cfg, guidance, steps):
            print(f"lerobot: {note}", flush=True)
        if norm_stats:
            self._apply_norm_stats(norm_stats, check_only=bool(adapter))
        self.camera_keys = list(cfg.camera_order)
        self.cameras = [key.removeprefix("observation.images.") for key in self.camera_keys]
        self.snapshots = snapshot_indices(cfg.n_obs_steps, cfg.history_snapshots)
        problems = [
            f"{name} {have} (so101.v1 needs {want})"
            for name, have, want in (
                ("conditioning", cfg.conditioning, "history"),
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
        history = next(s for s in self.pre.steps if isinstance(s, ObservationHistoryNormalizerProcessorStep))
        quantiles = {k: v.float().tolist() for k, v in history.state_dict().items()}
        guidance_action = cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action
        gpu = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None
        self.meta = {
            "checkpoint": str(checkpoint),
            "revision": revision,
            "adapter": adapter,
            "base": str(base),
            "base_revision": base_rev,
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
            "dtype": str(cfg.dtype).removeprefix("torch."),
            "video_vae_id": cfg.video_vae_id,
            "text_encoder_id": cfg.text_encoder_id,
            "config_video_vae_id": config_vae,
            "config_text_encoder_id": config_te,
            "compile": "dit+encoders" if compile else False,
            "offload_text_encoder": bool(offload_text_encoder),
            "device": str(device),
            "gpu": gpu,
            "torch": torch.__version__,
            "lerobot_commit": _git_commit(Path(lerobot.__file__).parent),
            "state_q01": quantiles["state.q01"],
            "state_q99": quantiles["state.q99"],
            "action_q01": quantiles["action.q01"],
            "action_q99": quantiles["action.q99"],
        }
        where = "CPU (offloaded)" if offload_text_encoder else str(device)
        print(
            f"lerobot: loaded in {time.perf_counter() - started:.1f} s on {device}{f' ({gpu})' if gpu else ''}, "
            f"text encoder on {where}",
            flush=True,
        )
        print(
            f"lerobot: sampling {cfg.sampler}, {cfg.num_inference_steps} steps, guidance "
            f"{cfg.guidance_scale}/{guidance_action}, shift {cfg.sampler_shift}, seed {cfg.inference_seed}, "
            f"{self.meta['dtype']}, {'compile_model (the first plan compiles)' if compile else 'eager'}",
            flush=True,
        )

    def _apply_norm_stats(self, stats: dict, check_only: bool) -> None:
        """Override the quantiles of every history normalizer step (pre and post), or, with an adapter, only check
        that they already equal `stats`: an adapter must run with the normalization it was trained with."""
        torch = self.torch
        steps = [s for s in (*self.pre.steps, *self.post.steps)
                 if getattr(s, "streams", None) and hasattr(s, "load_state_dict")]
        if not steps:
            raise RuntimeError("--norm-stats: no history normalizer step in this checkpoint's processors")
        for step in steps:
            current = step.state_dict()
            new = dict(current)
            for key, value in current.items():
                stream, q = key.split(".", 1)
                if stream in stats:
                    new[key] = torch.tensor(stats[stream][q], dtype=value.dtype)
            if check_only:
                bad = [k for k in new if not torch.allclose(new[k].float(), current[k].float(), atol=1e-4)]
                if bad:
                    raise RuntimeError(f"--norm-stats differ from the adapter's own normalization ({', '.join(bad)} in "
                                       f"{type(step).__name__}); serve the adapter with the base it was trained on")
            else:
                step.load_state_dict(new)
        print(f"lerobot: normalization {'checked against' if check_only else 'overridden by'} --norm-stats "
              f"({', '.join(stats)}) in {len(steps)} processor steps", flush=True)

    def infer(self, state, cmd_hist, frames, task):
        torch = self.torch
        obs = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))[None],
            "observation.command_history": torch.from_numpy(np.ascontiguousarray(cmd_hist, dtype=np.float32))[None],
            "task": task,
        }
        for cam, key in zip(self.cameras, self.camera_keys, strict=True):
            window = np.ascontiguousarray(frames[cam][self.order].transpose(0, 3, 1, 2))  # (8, 3, H, W) uint8
            obs[key] = torch.from_numpy(window)[None]  # the saved device step moves it; uint8 stays uint8
        with torch.inference_mode():
            self.pre.reset()
            prepared = self.pre(obs)
            actions = self.post(self.policy.predict_action_chunk(prepared))
        return actions[0].float().cpu().numpy()
