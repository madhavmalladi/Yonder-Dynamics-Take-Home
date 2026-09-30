"""Step 1: explore the dataset (read-only) and check for train/valid leakage.

Reports:
  * images / boxes per class per split
  * box size and image brightness/contrast statistics
  * leakage: (a) same Roboflow source filename in both splits,
             (b) near-duplicate images across splits via a 64-bit difference hash
Writes a text summary and a few plots to outputs/eda/.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("Sampled-YD-Object-Detection-2")
OUT = Path("outputs/eda")
CLASSES = ["bottle", "mallet"]
SPLITS = ["train", "valid"]
HASH_DIST = 6  # max Hamming distance (of 64 bits) to call two images near-duplicates


def source_name(path: Path) -> str:
    """Roboflow names files '<original>_<ext>.rf.<hash>.jpg'; strip the suffix."""
    return re.sub(r"_(jpg|jpeg|png)(_jpg)?\.rf\.[0-9a-f]+$", "", path.stem, flags=re.I)


def dhash(img_gray: np.ndarray) -> int:
    small = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load_split(split: str):
    rows = []
    for img_path in sorted((ROOT / split / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = []
        label_path = ROOT / split / "labels" / (img_path.stem + ".txt")
        for line in label_path.read_text().splitlines():
            if line.strip():
                c, x, y, w, h = line.split()
                boxes.append((int(c), float(x), float(y), float(w), float(h)))
        rows.append(
            dict(
                path=img_path,
                src=source_name(img_path),
                hash=dhash(gray),
                mean=float(gray.mean()),
                std=float(gray.std()),
                boxes=boxes,
            )
        )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(s) for s in SPLITS}
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    # --- counts ---
    for s in SPLITS:
        boxes = [b for r in data[s] for b in r["boxes"]]
        cnt = Counter(b[0] for b in boxes)
        imgs_per_class = Counter(c for r in data[s] for c in {b[0] for b in r["boxes"]})
        both = sum(1 for r in data[s] if len({b[0] for b in r["boxes"]}) > 1)
        log(f"[{s}] images={len(data[s])} boxes={len(boxes)} "
            + " ".join(f"{CLASSES[c]}_boxes={cnt[c]}(imgs={imgs_per_class[c]})" for c in range(2))
            + f" imgs_with_both={both}")

    # --- box size / image stats ---
    log("\nBox size (normalized w*h area) and image brightness, per split/class:")
    for s in SPLITS:
        for c in range(2):
            areas = np.array([b[3] * b[4] for r in data[s] for b in r["boxes"] if b[0] == c])
            log(f"  {s:5s} {CLASSES[c]:6s} area median={np.median(areas):.3f} "
                f"p10={np.percentile(areas, 10):.3f} p90={np.percentile(areas, 90):.3f}")
        m = np.array([r["mean"] for r in data[s]])
        sd = np.array([r["std"] for r in data[s]])
        log(f"  {s:5s} gray mean: mean={m.mean():.1f} min={m.min():.0f} max={m.max():.0f} "
            f"| gray std (contrast): mean={sd.mean():.1f} min={sd.min():.0f}")
        log(f"        dark images (mean<70): {(m < 70).sum()}  bright (mean>190): {(m > 190).sum()}")

    # --- leakage check ---
    log("\nLeakage check:")
    tr_src, va_src = defaultdict(list), defaultdict(list)
    for r in data["train"]:
        tr_src[r["src"]].append(r)
    for r in data["valid"]:
        va_src[r["src"]].append(r)
    shared = set(tr_src) & set(va_src)
    va_imgs_shared = sum(len(va_src[k]) for k in shared)
    log(f"  unique source names: train={len(tr_src)} valid={len(va_src)}")
    log(f"  source names present in BOTH splits: {len(shared)} "
        f"-> {va_imgs_shared}/{len(data['valid'])} valid images share a source with train")
    dup_in_train = sum(1 for v in tr_src.values() if len(v) > 1)
    log(f"  train sources with >1 image (augmented copies): {dup_in_train}")

    tr_hashes = np.array([r["hash"] for r in data["train"]], dtype=object)
    near = []
    for r in data["valid"]:
        dists = [bin(r["hash"] ^ h).count("1") for h in tr_hashes]
        j = int(np.argmin(dists))
        near.append((dists[j], r["path"].name, data["train"][j]["path"].name))
    n_near = sum(1 for d, _, _ in near if d <= HASH_DIST)
    log(f"  valid images with a train near-duplicate (dHash dist <= {HASH_DIST}): "
        f"{n_near}/{len(near)}")
    log("  closest 5 pairs: ")
    for d, v, t in sorted(near)[:5]:
        log(f"    dist={d}  valid={v[:40]}  train={t[:40]}")

    # --- plots ---
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s in SPLITS:
        ax[0].hist([r["mean"] for r in data[s]], bins=30, alpha=0.6, label=s)
        ax[1].hist([r["std"] for r in data[s]], bins=30, alpha=0.6, label=s)
    ax[0].set_title("Image mean gray level")
    ax[1].set_title("Image gray std (contrast)")
    ax[0].legend()
    for c in range(2):
        a = [b[3] * b[4] for r in data["train"] for b in r["boxes"] if b[0] == c]
        ax[2].hist(a, bins=30, alpha=0.6, label=CLASSES[c])
    ax[2].set_title("Train box area (normalized)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png", dpi=120)

    # contact sheet of random train images with boxes drawn
    rng = np.random.default_rng(0)
    pick = rng.choice(len(data["train"]), 16, replace=False)
    tiles = []
    for i in pick:
        r = data["train"][i]
        img = cv2.imread(str(r["path"]))
        h, w = img.shape[:2]
        for c, x, y, bw, bh in r["boxes"]:
            p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
            p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
            cv2.rectangle(img, p1, p2, (0, 200, 0) if c == 0 else (0, 0, 230), 2)
        tiles.append(cv2.resize(img, (256, 256)))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)])
    cv2.imwrite(str(OUT / "samples_train.jpg"), sheet)

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    log(f"\nWrote {OUT}/summary.txt, distributions.png, samples_train.jpg")


if __name__ == "__main__":
    main()
