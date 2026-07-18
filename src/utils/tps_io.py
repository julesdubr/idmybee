"""Minimal TPS file reader."""
import numpy as np


def parse_tps(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    specimens = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('LM='):
            lm = int(line.split('=')[1])
            coords = []
            for j in range(1, lm + 1):
                parts = lines[i + j].strip().split()
                coords.append((float(parts[0]), float(parts[1])))
            i += lm + 1
            image, specid = None, None
            while i < n and not lines[i].strip().startswith('LM='):
                l = lines[i].strip()
                if l.startswith('IMAGE='):
                    image = l.split('=', 1)[1]
                elif l.startswith('ID='):
                    specid = l.split('=', 1)[1]
                i += 1
            # reshape(-1, 2) keeps shape (0, 2) instead of (0,) when lm == 0,
            # so downstream code fails on an explicit check rather than a
            # cryptic "mean of empty slice" / matmul error.
            specimens.append({'lm': lm, 'coords': np.array(coords).reshape(-1, 2),
                               'image': image, 'id': specid})
        else:
            i += 1
    return specimens


def as_array(specimens):
    """(n_specimens, n_landmarks, 2) array. Assumes constant landmark count."""
    return np.stack([s['coords'] for s in specimens])


def filter_valid(specimens, expected_lm, label=""):
    """Split specimens into (valid, skipped) based on landmark count.
    Skipped specimens (LM=0 failed detections, or any unexpected count)
    are printed with their IMAGE=/ID= so nothing silently disappears.
    """
    valid, skipped = [], []
    for s in specimens:
        if s['lm'] == expected_lm and s['coords'].shape == (expected_lm, 2):
            valid.append(s)
        else:
            skipped.append(s)
    if skipped:
        print(f"[{label}] skipped {len(skipped)}/{len(specimens)} specimen(s) "
              f"with LM != {expected_lm}:")
        for s in skipped[:20]:
            print(f"    LM={s['lm']}  ID={s['id']}  IMAGE={s['image']}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")
    return valid, skipped