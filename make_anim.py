import argparse
from pathlib import Path
import re
import numpy as np
import imageio.v2 as imageio
from PIL import Image

'''

example usage: python make_anim.py pngs_here 'number_(\d+)\.png' --gif --fps 1

'''
# ============================================================
# CLI arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, help="Directory containing PNG files")
parser.add_argument("pattern", type=str, help=r"Regex pattern with (\d+) for index")
parser.add_argument("--fps", type=float, default=0.5, help="Frames per second")
parser.add_argument("--gif", action="store_true", help="Write GIF")
parser.add_argument("--mp4", action="store_true", help="Write MP4")
parser.add_argument("--output", type=str, default="animation", help="Output basename")
args = parser.parse_args()

if not args.gif and not args.mp4:
    raise RuntimeError("Specify at least one of --gif or --mp4")

# ============================================================
# find all matching PNG files and sort by extracted integer index
# ============================================================
input_dir = Path(args.input_dir)
pattern = re.compile(args.pattern)

files_with_index = []
for f in input_dir.glob("*.png"):
    m = pattern.search(f.name)
    if m:
        idx = int(m.group(1))
        files_with_index.append((idx, f))

if not files_with_index:
    raise RuntimeError("No matching PNG files found")

files_with_index.sort(key=lambda x: x[0])
files = [f for _, f in files_with_index]

print("Using files:")
for idx, f in files_with_index:
    print("  {}   (index={})".format(f.name, idx))

# ============================================================
# load frames as RGB
# ============================================================
raw_frames = [np.array(Image.open(f).convert("RGB")) for f in files]

print("\nFrame shapes:")
for f, frame in zip(files, raw_frames):
    print("  {}  {}".format(f.name, frame.shape))

# ============================================================
# pad all frames to common size
# ============================================================
max_h = max(frame.shape[0] for frame in raw_frames)
max_w = max(frame.shape[1] for frame in raw_frames)

frames = []
for f, frame in zip(files, raw_frames):
    h, w, c = frame.shape
    if c != 3:
        raise RuntimeError("Unexpected channel count in {}: {}".format(f.name, frame.shape))

    if h == max_h and w == max_w:
        frames.append(frame)
        continue

    print("Padding {} from {} to ({}, {}, 3)".format(f.name, frame.shape, max_h, max_w))

    canvas = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
    y0 = (max_h - h) // 2
    x0 = (max_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w, :] = frame
    frames.append(canvas)

# ============================================================
# write outputs
# ============================================================
if args.gif:
    gif_name = Path(args.output).with_suffix(".gif")
    imageio.mimsave(gif_name, frames, duration=1.0 / args.fps, loop=0)
    print("\nSaved {}".format(gif_name))

if args.mp4:
    mp4_name = Path(args.output).with_suffix(".mp4")
    with imageio.get_writer(mp4_name, fps=args.fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print("Saved {}".format(mp4_name))