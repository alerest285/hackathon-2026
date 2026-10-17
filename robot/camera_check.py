"""Find the cameras, then save what the model will see.

    uv run --python 3.13 python camera_check.py                      # list working indices 0..5
    uv run --python 3.13 python camera_check.py --scene 0 --wrist 1  # save frames + measured fps

With --scene/--wrist it writes runs/camera-check-<stamp>/scene.png and wrist.png (full resolution) and
composed.png: scene | wrist side by side at 256x256 each, decoded from the bytes the client sends. The left
half must be the fixed scene camera. A swapped pair cannot be fixed by renaming: swap the indices.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def list_cameras(max_index: int) -> int:
    from so101_client.cameras import capture_mode, open_capture

    found = 0
    for i in range(max_index + 1):
        try:
            cap = open_capture(i)
        except RuntimeError:
            print(f"  {i}: -")
            continue
        ok = False
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        w, h, fps = capture_mode(cap)
        cap.release()
        if ok:
            found += 1
            print(f"  {i}: {frame.shape[1]}x{frame.shape[0]} (reports {w}x{h} @ {fps:.0f} fps)")
        else:
            print(f"  {i}: opens but returns no frames")
    if not found:
        print("no camera returned frames. On macOS give the terminal camera permission "
              "(System Settings > Privacy & Security > Camera) and try again.")
    return 0 if found else 1


def check(scene, wrist, seconds: float, image_format: str, image_size: int, out_dir: Path) -> int:
    import cv2

    from so101_client.cameras import Cameras, compose, decode_image

    cams = Cameras({"scene": scene, "wrist": wrist}, image_format=image_format, image_size=image_size)
    cams.start()
    try:
        first = cams.latest()
        t0 = time.perf_counter()
        time.sleep(seconds)
        last = cams.latest()
        dt = time.perf_counter() - t0
    finally:
        cams.stop()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scene", "wrist"):
        fps = (last[name].seq - first[name].seq) / dt
        w, h, reported = cams.modes[name]
        cv2.imwrite(str(out_dir / f"{name}.png"), last[name].bgr)
        print(f"  {name}: source {cams.sources[name]}, {last[name].bgr.shape[1]}x{last[name].bgr.shape[0]}, "
              f"measured {fps:.1f} fps (reports {reported:.0f}), capture errors {cams.errors[name]}")
    composed = compose(decode_image(last["scene"].encoded, image_format), decode_image(last["wrist"].encoded, image_format))
    cv2.imwrite(str(out_dir / "composed.png"), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_dir / 'scene.png'}, {out_dir / 'wrist.png'}")
    print(f"wrote {out_dir / 'composed.png'} ({composed.shape[1]}x{composed.shape[0]}): left = scene, right = wrist")
    print("Open composed.png and check the left half is the fixed scene camera. If not, swap --scene and --wrist.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", help="scene camera index or path")
    p.add_argument("--wrist", help="wrist camera index or path")
    p.add_argument("--max-index", type=int, default=5)
    p.add_argument("--seconds", type=float, default=3.0, help="how long to measure fps")
    p.add_argument("--image-format", choices=("jpeg", "png", "raw"), default="jpeg")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--out", type=Path, help="default: robot/runs/camera-check-<stamp>/")
    a = p.parse_args(argv)
    if (a.scene is None) != (a.wrist is None):
        p.error("give both --scene and --wrist, or neither to list cameras")
    if a.scene is None:
        print(f"probing camera indices 0..{a.max_index}")
        return list_cameras(a.max_index)
    out = a.out or HERE / "runs" / f"camera-check-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    return check(a.scene, a.wrist, a.seconds, a.image_format, a.image_size, out)


if __name__ == "__main__":
    sys.exit(main())
