import imageio.v2 as imageio
from pathlib import Path
from natsort import natsorted
import sys

fpath = sys.argv[1]
png_dir = Path(fpath)
png_files = natsorted(png_dir.glob("*.png"))

frames = [imageio.imread(f) for f in png_files]

imageio.mimsave("evolution.gif", frames, duration=1/30, loop=0)