"""Synthesize 'hard photo' training examples by compositing real document
images onto real photo backgrounds with perspective warp, occlusion, and
harsh lighting -- simulating a handheld phone photo of a bill/receipt taken
in a cluttered, dim environment (folder, table, hand/glass partially
covering it).

This directly targets a generalization gap found by manual testing: the
trained classifier, built only from clean scans / well-lit document photos,
misclassified two real handheld restaurant-bill photos as *not* invoices.
Neither exists as a public dataset at any scale, so we synthesize it from
data we already have: positive/negative document images (from the labeled
pool) as foregrounds, and the coco_photos hard-negative images (from the
same pool) as backgrounds.

Ground truth is known by construction (we chose the source document), so
labeled rows are appended directly to data/labeled/manifest.csv -- no
teacher re-labeling needed.

Usage:
    python scripts/synthesize_hard_examples.py --n-positive 900 --n-negative 400
"""

import argparse
import csv
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
RAW_DIR = REPO_ROOT / "data" / "raw" / "pool"
MAX_SIDE = 512

LIGHTING = A.Compose(
    [
        A.RandomGamma(gamma_limit=(50, 160), p=0.8),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ImageCompression(quality_range=(35, 85), p=0.6),
    ]
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def warp_and_mask(doc: np.ndarray, canvas_w: int, canvas_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize `doc` to a random fraction of the canvas, apply a random
    perspective tilt, and return (warped_bgr, alpha_mask) both cropped to
    their own bounding box."""
    doc_h, doc_w = doc.shape[:2]
    scale = random.uniform(0.4, 0.85)
    target_w = max(20, int(canvas_w * scale))
    target_h = max(20, int(target_w * doc_h / doc_w))
    if target_h > canvas_h * 0.95:
        target_h = int(canvas_h * 0.95)
        target_w = max(20, int(target_h * doc_w / doc_h))
    doc_resized = cv2.resize(doc, (target_w, target_h))

    src = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
    jitter = 0.15 * min(target_w, target_h)
    dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
    dst -= dst.min(axis=0)
    out_w = max(1, int(dst[:, 0].max()) + 1)
    out_h = max(1, int(dst[:, 1].max()) + 1)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc_resized, M, (out_w, out_h), borderValue=(0, 0, 0))
    mask_full = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_full, M, (out_w, out_h), borderValue=0)
    return warped, mask


def add_occlusion(canvas: np.ndarray, region_mask: np.ndarray) -> None:
    """Draw 1-2 semi-opaque blobs over part of `region_mask`'s footprint,
    simulating a hand/object partially covering the document. Mutates
    `canvas` in place."""
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return
    for _ in range(random.randint(1, 2)):
        idx = random.randrange(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        radius_x = random.randint(20, max(21, region_mask.shape[1] // 4))
        radius_y = random.randint(20, max(21, region_mask.shape[0] // 4))
        color = random.choice([(180, 140, 120), (60, 55, 50), (210, 190, 170)])
        alpha = random.uniform(0.55, 0.9)
        overlay = canvas.copy()
        cv2.ellipse(overlay, (cx, cy), (radius_x, radius_y), random.randint(0, 360), 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def composite(doc: np.ndarray, bg: np.ndarray) -> np.ndarray:
    bg = cv2.resize(bg, (max(bg.shape[1], 600), max(bg.shape[0], 600)))
    canvas = bg.copy()
    canvas_h, canvas_w = canvas.shape[:2]

    warped, mask = warp_and_mask(doc, canvas_w, canvas_h)
    wh, ww = warped.shape[:2]
    if ww >= canvas_w or wh >= canvas_h:
        scale = min((canvas_w - 1) / ww, (canvas_h - 1) / wh)
        warped = cv2.resize(warped, (max(1, int(ww * scale)), max(1, int(wh * scale))))
        mask = cv2.resize(mask, (warped.shape[1], warped.shape[0]))
        wh, ww = warped.shape[:2]

    x0 = random.randint(0, canvas_w - ww)
    y0 = random.randint(0, canvas_h - wh)

    # soft drop shadow: blurred, offset copy of the mask, darkened into the canvas
    shadow_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    sx, sy = x0 + random.randint(5, 15), y0 + random.randint(5, 15)
    sx_end, sy_end = min(canvas_w, sx + ww), min(canvas_h, sy + wh)
    if sx < canvas_w and sy < canvas_h:
        shadow_mask[sy:sy_end, sx : sx + (sx_end - sx)] = mask[: sy_end - sy, : sx_end - sx]
    shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 0)
    shadow_strength = (shadow_mask.astype(np.float32) / 255.0 * 0.5)[..., None]
    canvas = (canvas.astype(np.float32) * (1 - shadow_strength)).astype(np.uint8)

    roi = canvas[y0 : y0 + wh, x0 : x0 + ww]
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = (warped.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    canvas[y0 : y0 + wh, x0 : x0 + ww] = blended

    full_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    full_mask[y0 : y0 + wh, x0 : x0 + ww] = mask
    if random.random() < 0.6:
        add_occlusion(canvas, full_mask)

    canvas = LIGHTING(image=canvas)["image"]
    return canvas


def load_manifest_rows() -> list[dict]:
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-positive", type=int, default=900)
    parser.add_argument("--n-negative", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_manifest_rows()
    # Only ever synthesize from *real* documents, never from a previous run's
    # synthetic output -- otherwise reruns would compound lineage chains and
    # make split-leakage prevention (via source_doc) impossible to track.
    real_rows = [r for r in rows if not r["source"].startswith("synthetic_")]
    positives = [r for r in real_rows if r["invoice_label"] == "1"]
    negatives = [r for r in real_rows if r["invoice_label"] == "0" and r["source"] != "coco_photos"]
    backgrounds = [r for r in real_rows if r["source"] == "coco_photos"]

    if not positives or not negatives or not backgrounds:
        raise SystemExit(
            f"Need positives ({len(positives)}), non-photo negatives ({len(negatives)}), "
            f"and coco_photos backgrounds ({len(backgrounds)}) in the manifest -- run build_dataset.py first."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(MANIFEST, "a", newline="") as f:
        writer = csv.writer(f)

        jobs = [(1, positives, args.n_positive), (0, negatives, args.n_negative)]
        for label, doc_pool, n in jobs:
            kind = "hard_pos" if label == 1 else "hard_neg"
            print(f"Synthesizing {n} synthetic_{kind} examples...")
            for i in range(n):
                doc_row = random.choice(doc_pool)
                bg_row = random.choice(backgrounds)
                try:
                    doc = load_rgb(REPO_ROOT / doc_row["filepath"])
                    bg = load_rgb(REPO_ROOT / bg_row["filepath"])
                    composited = composite(doc, bg)
                except Exception as e:  # noqa: BLE001 - one bad pair shouldn't kill the run
                    print(f"  skip (compose error): {e}")
                    continue

                img = Image.fromarray(composited)
                img.thumbnail((MAX_SIDE, MAX_SIDE))
                fname = f"syn_{kind}_{i:05d}.jpg"
                fpath = RAW_DIR / fname
                img.save(fpath, format="JPEG", quality=88)

                writer.writerow(
                    [
                        str(fpath.relative_to(REPO_ROOT)),
                        f"synthetic_{kind}",
                        "synthetic_composite",
                        label,
                        "1.0000",
                        doc_row["filepath"],  # source_doc: ties this composite to its real source image
                    ]
                )
                n_written += 1
                if (i + 1) % 200 == 0:
                    f.flush()
                    print(f"  {kind}: {i + 1}/{n}")
            f.flush()

    print(f"\nWrote {n_written} synthetic rows to {MANIFEST}")


if __name__ == "__main__":
    main()
