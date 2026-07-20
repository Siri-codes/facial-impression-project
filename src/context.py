from PIL import Image
import numpy as np
from config import IMAGE_DIR, DATA_DIR, SUBSAMPLE_SEED, CONTEXT_GRID, CONTEXT_SEED

def context_ids(all_ids, exclude=(), seed=CONTEXT_SEED, n=25):
    """Fixed, reproducible sample of n stimulus ids."""
    pool = np.array([i for i in all_ids if i not in set(exclude)])
    return sorted(np.random.default_rng(seed).choice(pool, size=n, replace=False))

def make_context_grid(stimulus_ids, thumb=200, out_path=CONTEXT_GRID):
    grid = Image.new("RGB", (thumb*5, thumb*5), "white")
    for i, sid in enumerate(list(stimulus_ids)[:25]):
        im = Image.open(IMAGE_DIR / f"{sid}.jpg").resize((thumb, thumb))
        grid.paste(im, ((i % 5) * thumb, (i // 5) * thumb))
    grid.save(out_path, quality=85)
    return out_path