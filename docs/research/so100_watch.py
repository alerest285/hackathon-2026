"""Watch a console run: save camera frames + joint state from the running console, then make contact sheets to look at.

    # terminal 2, while console/start.sh runs (dry or live):
    python3 docs/research/so100_watch.py record                      # Ctrl-C to stop (or it stops when the console does)
    # afterwards (needs Pillow):
    uv run --no-project --with pillow python docs/research/so100_watch.py sheet ~/hackathon-data/watch/<stamp>

record: polls http://127.0.0.1:8765 (/status, /cam/scene.jpg, /cam/wrist.jpg) at --hz (default 4) into
~/hackathon-data/watch/<stamp>/: scene/NNNNN.jpg, wrist/NNNNN.jpg, status.jsonl (time, tick, live, task, model-space
state and command, native raw_state, held, last_chunk). Read-only: it never talks to the arm or the server.
About 40 MB per minute at 4 Hz; stops after --max-s (default 300).
sheet: contact sheets (scene and wrist side by side, time and native joints printed on each tile), plus summary.txt
(per-joint native range and the frames where each joint was at its min/max).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def get(url, timeout=1.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def record(a):
    out = Path(a.out).expanduser() / time.strftime("%Y%m%dT%H%M%S")
    for cam in ("scene", "wrist"):
        (out / cam).mkdir(parents=True, exist_ok=True)
    print(f"recording {a.console} at {a.hz} Hz into {out} (Ctrl-C stops)")
    n, misses, t0 = 0, 0, time.time()
    with open(out / "status.jsonl", "w") as f:
        try:
            while time.time() - t0 < a.max_s:
                t = time.time()
                try:
                    s = json.loads(get(a.console + "/status"))
                    for cam in ("scene", "wrist"):
                        (out / cam / f"{n:05d}.jpg").write_bytes(get(f"{a.console}/cam/{cam}.jpg?{int(t * 1000)}"))
                    misses = 0
                except OSError as e:
                    misses += 1
                    if misses * (1 / a.hz) > 5:
                        print(f"console unreachable for 5 s ({e}); stopping")
                        break
                    time.sleep(1 / a.hz)
                    continue
                keep = {k: s.get(k) for k in ("tick", "live", "task", "state", "cmd", "raw_state", "held", "remaining",
                                              "last_chunk", "out_of_range", "fault", "error")}
                f.write(json.dumps({"n": n, "t": round(t - t0, 3), **keep}) + "\n")
                f.flush()
                n += 1
                if n % (a.hz * 10) == 0:
                    print(f"  {n} frames, {t - t0:.0f} s, live={s.get('live')} task={s.get('task')!r}", flush=True)
                time.sleep(max(0.0, 1 / a.hz - (time.time() - t)))
        except KeyboardInterrupt:
            pass
    print(f"saved {n} frames to {out}")


def native(r):
    """Native degrees: raw_state when a joint map is on, else state (the console sends native values unmapped)."""
    return r.get("raw_state") or r.get("state") or {}


def sheet(a):
    from PIL import Image, ImageDraw

    d = Path(a.dir).expanduser()
    rows = [json.loads(l) for l in open(d / "status.jsonl")]
    rows = [r for r in rows if (d / "scene" / f"{r['n']:05d}.jpg").exists()]
    if a.from_s is not None:
        rows = [r for r in rows if r["t"] >= a.from_s]
    if a.to_s is not None:
        rows = [r for r in rows if r["t"] <= a.to_s]
    pick = [rows[round(i * (len(rows) - 1) / max(1, a.tiles - 1))] for i in range(min(a.tiles, len(rows)))]
    tw, th = 480, 180  # one tile: scene 240x180 + wrist 240x180
    cols = 4
    for page in range(0, len(pick), cols * 6):
        part = pick[page:page + cols * 6]
        img = Image.new("RGB", (cols * tw, ((len(part) + cols - 1) // cols) * (th + 30)), "white")
        dr = ImageDraw.Draw(img)
        for i, r in enumerate(part):
            x, y = (i % cols) * tw, (i // cols) * (th + 30)
            for k, cam in enumerate(("scene", "wrist")):
                im = Image.open(d / cam / f"{r['n']:05d}.jpg").convert("RGB").resize((240, 180))
                img.paste(im, (x + k * 240, y))
            raw = native(r)
            txt = f"t={r['t']:.1f}s {'LIVE' if r.get('live') else 'dry'} " + " ".join(
                f"{j[:2] if j != 'wrist_roll' else 'wr'}{raw.get(j, float('nan')):.0f}" for j in J)
            dr.text((x + 3, y + th + 3), txt, fill="black")
            dr.text((x + 3, y + th + 15), "native deg: pan lift elbow wflex wroll grip", fill="gray")
        p = d / f"sheet_{page // (cols * 6):02d}.png"
        img.save(p)
        print("wrote", p)
    lines = [f"{len(rows)} samples, {rows[0]['t']:.1f}..{rows[-1]['t']:.1f} s, task {rows[-1].get('task')!r}"]
    for j in J:
        v = [(native(r)[j], r["t"], r["n"]) for r in rows if j in native(r)]
        if v:
            lo, hi = min(v), max(v)
            lines.append(f"{j:14s} native {lo[0]:7.1f} (t={lo[1]:.1f}s frame {lo[2]}) .. {hi[0]:7.1f} (t={hi[1]:.1f}s frame {hi[2]})")
    held = sum(1 for r in rows if r.get("held"))
    lines.append(f"held (waiting for a plan) in {held}/{len(rows)} samples")
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("record")
r.add_argument("--console", default="http://127.0.0.1:8765")
r.add_argument("--hz", type=int, default=4)
r.add_argument("--max-s", type=float, default=300)
r.add_argument("--out", default="~/hackathon-data/watch")
s = sub.add_parser("sheet")
s.add_argument("dir")
s.add_argument("--tiles", type=int, default=24)
s.add_argument("--from-s", type=float)
s.add_argument("--to-s", type=float)
a = p.parse_args()
sys.exit(record(a) if a.cmd == "record" else sheet(a))
