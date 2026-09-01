"""Create leakage-free bill screenshots from existing split-aware bill images.

Each underlying source document contributes at most one rendered screenshot.
Its original train/validation/test assignment is retained, preventing a bill
source from appearing in more than one split.  The surrounding UI is neutral
so the model must recognize the bill content rather than keywords in chrome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = Path("/tmp/verifier_v2_production_local_splits.csv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "screenshot_positive_bills"
FIELDS = (
    "filepath",
    "split",
    "parent_filepath",
    "source_doc",
    "class_index",
    "class_name",
    "context",
    "sha256",
)
CANVAS_SIZE = (720, 1440)
CONTEXTS = ("gallery", "pdf_viewer", "email_attachment", "chat_attachment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


FONT_SMALL = font(22)
FONT_MEDIUM = font(28)
FONT_MEDIUM_BOLD = font(28, bold=True)


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as source:
        return list(csv.DictReader(source))


def source_key(row: dict) -> str:
    return row.get("source_doc") or row["filepath"]


def unique_bill_sources(rows: list[dict], seed: int) -> list[dict]:
    bills = [row for row in rows if int(row["class_index"]) in (0, 1)]
    key_splits: dict[str, set[str]] = {}
    for row in bills:
        key_splits.setdefault(source_key(row), set()).add(row["split"])
    leaked = {key: splits for key, splits in key_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(leaked.items())[:5]
        raise RuntimeError(f"Underlying bill sources cross splits: {preview}")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in bills:
        grouped.setdefault((row["split"], source_key(row)), []).append(row)
    representatives = []
    for key in sorted(grouped):
        source = key[1]
        candidates = sorted(
            grouped[key],
            key=lambda row: (
                row["filepath"] != source,
                "syn_hard_neg" in row["filepath"],
                row["filepath"],
            ),
        )
        representatives.append(candidates[0])
    return representatives


def draw_status_bar(canvas: Image.Image, draw: ImageDraw.ImageDraw, dark: bool) -> None:
    foreground = (245, 245, 245) if dark else (25, 25, 25)
    draw.text((28, 14), "9:41", fill=foreground, font=FONT_SMALL)
    draw.rounded_rectangle((625, 18, 681, 40), radius=6, outline=foreground, width=3)
    draw.rectangle((681, 24, 686, 34), fill=foreground)
    draw.rectangle((632, 24, 670, 34), fill=foreground)
    draw.ellipse((570, 20, 590, 40), outline=foreground, width=3)
    draw.arc((530, 19, 560, 45), 205, 335, fill=foreground, width=3)


def paste_document(
    canvas: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
    rng: random.Random,
    fit: bool,
    shadow: bool = False,
) -> None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if fit:
        rendered = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    else:
        scale = width / source.width
        rendered = source.resize(
            (width, max(1, round(source.height * scale))), Image.Resampling.LANCZOS
        )
        if rendered.height > height:
            maximum = rendered.height - height
            crop_top = round(maximum * rng.uniform(0.0, 0.55))
            rendered = rendered.crop((0, crop_top, width, crop_top + height))
    x = left + (width - rendered.width) // 2
    y = top + (height - rendered.height) // 2
    if shadow:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (x + 8, y + 10, x + rendered.width + 8, y + rendered.height + 10),
            radius=8,
            fill=(0, 0, 0, 48),
        )
        canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))
    canvas.paste(rendered, (x, y))


def render_gallery(source: Image.Image, rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    draw_status_bar(canvas, draw, dark=True)
    draw.text((28, 70), "‹", fill=(245, 245, 245), font=FONT_MEDIUM_BOLD)
    draw.text((300, 70), "Today", fill=(245, 245, 245), font=FONT_MEDIUM_BOLD)
    draw.text((590, 70), "•••", fill=(245, 245, 245), font=FONT_MEDIUM)
    paste_document(canvas, source, (18, 130, 702, 1305), rng, fit=rng.random() < 0.45)
    draw.line((0, 1340, 720, 1340), fill=(55, 58, 64), width=2)
    for x, label in ((95, "↥"), (260, "♡"), (455, "ⓘ"), (625, "⌫")):
        draw.text((x, 1360), label, fill=(230, 230, 230), font=FONT_MEDIUM)
    return canvas


def render_pdf(source: Image.Image, rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (218, 220, 224))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 720, 120), fill=(248, 248, 248))
    draw_status_bar(canvas, draw, dark=False)
    draw.text((25, 68), "‹ Files", fill=(20, 100, 210), font=FONT_MEDIUM)
    draw.text((284, 70), "Document", fill=(30, 30, 30), font=FONT_MEDIUM_BOLD)
    draw.text((650, 70), "•••", fill=(30, 30, 30), font=FONT_MEDIUM)
    paste_document(canvas, source, (44, 148, 676, 1328), rng, fit=True, shadow=True)
    draw.rectangle((0, 1360, 720, 1440), fill=(248, 248, 248))
    draw.text((115, 1380), "↥", fill=(40, 100, 200), font=FONT_MEDIUM)
    draw.text((330, 1380), "⌕", fill=(40, 100, 200), font=FONT_MEDIUM)
    draw.text((555, 1380), "✎", fill=(40, 100, 200), font=FONT_MEDIUM)
    return canvas


def render_email(source: Image.Image, rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw_status_bar(canvas, draw, dark=False)
    draw.line((0, 55, 720, 55), fill=(220, 220, 220), width=2)
    draw.text((24, 70), "‹ Inbox", fill=(30, 105, 220), font=FONT_MEDIUM)
    draw.text((620, 70), "•••", fill=(45, 45, 45), font=FONT_MEDIUM)
    draw.text((28, 130), "Your document", fill=(25, 25, 25), font=FONT_MEDIUM_BOLD)
    draw.ellipse((30, 185, 82, 237), fill=(75, 120, 190))
    draw.text((100, 187), "Account Services", fill=(35, 35, 35), font=FONT_SMALL)
    draw.text((100, 218), "to me", fill=(125, 125, 125), font=FONT_SMALL)
    draw.text((32, 270), "Please find the requested attachment below.", fill=(55, 55, 55), font=FONT_SMALL)
    paste_document(canvas, source, (34, 330, 686, 1335), rng, fit=rng.random() < 0.55, shadow=True)
    return canvas


def render_chat(source: Image.Image, rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (227, 236, 225))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 720, 120), fill=(248, 248, 248))
    draw_status_bar(canvas, draw, dark=False)
    draw.text((24, 70), "‹", fill=(25, 105, 210), font=FONT_MEDIUM_BOLD)
    draw.ellipse((65, 62, 112, 109), fill=(105, 140, 175))
    draw.text((130, 70), "Documents", fill=(25, 25, 25), font=FONT_MEDIUM_BOLD)
    draw.rounded_rectangle((24, 145, 696, 1340), radius=20, fill=(255, 255, 255))
    paste_document(canvas, source, (40, 165, 680, 1285), rng, fit=rng.random() < 0.50)
    draw.text((590, 1300), "9:40", fill=(130, 130, 130), font=FONT_SMALL)
    draw.rectangle((0, 1360, 720, 1440), fill=(248, 248, 248))
    draw.rounded_rectangle((65, 1372, 640, 1428), radius=24, outline=(190, 190, 190), width=2)
    return canvas


RENDERERS = {
    "gallery": render_gallery,
    "pdf_viewer": render_pdf,
    "email_attachment": render_email,
    "chat_attachment": render_chat,
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representatives = unique_bill_sources(load_rows(args.splits), args.seed)
    manifest_rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    contexts = {context: 0 for context in CONTEXTS}
    for index, row in enumerate(representatives, start=1):
        identity = f"{args.seed}:{row['split']}:{source_key(row)}"
        digest = hashlib.sha256(identity.encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        context = CONTEXTS[int(digest[16:24], 16) % len(CONTEXTS)]
        destination = output_dir / f"{digest[:24]}.jpg"
        if args.overwrite or not destination.exists():
            with Image.open(REPO_ROOT / row["filepath"]) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
                rendered = RENDERERS[context](source, rng)
                rendered.save(destination, format="JPEG", quality=91, optimize=True)
        with open(destination, "rb") as image_file:
            image_digest = hashlib.sha256(image_file.read()).hexdigest()
        manifest_rows.append(
            {
                "filepath": str(destination.relative_to(REPO_ROOT)),
                "split": row["split"],
                "parent_filepath": row["filepath"],
                "source_doc": source_key(row),
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "context": context,
                "sha256": image_digest,
            }
        )
        counts[row["split"]] += 1
        contexts[context] += 1
        if index % 250 == 0:
            print(f"Rendered {index}/{len(representatives)}")

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved {len(manifest_rows)} screenshot bills counts={counts}")
    print(f"Contexts={contexts}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
