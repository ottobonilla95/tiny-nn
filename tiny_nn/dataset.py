"""Independent test renderer: vector drawing at 8× resolution with antialiasing.

This module never imports the weight compiler or its geometric templates.
Labels come from drawing commands, not predictions. No examples fit weights.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import ALL_LABELS, GRID_SIZE, LABELS

PROFILES = ("clean", "thick", "noise", "gaps")
DATASET_VERSION = 1
SCALE = 8


def pixel_hash(pixels):
    return hashlib.sha256(np.asarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def render_shape(label, rng, profile="clean"):
    if label not in LABELS or profile not in PROFILES:
        raise ValueError("Unknown shape or difficulty profile")
    size = rng.uniform(7, 13)
    stroke = rng.uniform(1.8, 2.7) if profile == "thick" else rng.uniform(0.9, 1.55)
    width = stroke if label == "vertical" else size
    height = stroke if label == "horizontal" else size
    left, top = rng.uniform(0.3, GRID_SIZE - width - 0.3), rng.uniform(0.3, GRID_SIZE - height - 0.3)
    image = Image.new("L", (GRID_SIZE * SCALE,) * 2)
    draw = ImageDraw.Draw(image)

    def rectangle(x, y, w, h):
        draw.rectangle((round(x * SCALE), round(y * SCALE), round((x + w) * SCALE) - 1,
                        round((y + h) * SCALE) - 1), fill=255)

    box = (round(left * SCALE), round(top * SCALE), round((left + width) * SCALE) - 1,
           round((top + height) * SCALE) - 1)
    if label in ("horizontal", "vertical"):
        rectangle(left, top, width, height)
    elif label == "cross":
        rectangle(left, top + (size - stroke) / 2, size, stroke)
        rectangle(left + (size - stroke) / 2, top, stroke, size)
    elif label == "square":
        draw.rectangle(box, outline=255, width=round(stroke * SCALE))
    else:
        draw.ellipse(box, outline=255, width=round(stroke * SCALE))
    pixels = np.array(image.resize((GRID_SIZE,) * 2, Image.Resampling.LANCZOS)).copy()
    if profile == "noise":
        count = int(rng.integers(2, 6))
        locations = rng.choice(pixels.size, size=count, replace=False)
        pixels.flat[locations] = 255 - pixels.flat[locations]
    elif profile == "gaps":
        locations = np.flatnonzero(pixels.ravel() >= 128)
        count = min(len(locations), int(rng.integers(1, 4)))
        pixels.flat[rng.choice(locations, size=count, replace=False)] = 0
    return pixels, {"size": float(size), "stroke": float(stroke), "left": float(left), "top": float(top)}


def render_unknown(kind, rng):
    if kind == "blank":
        return np.zeros((GRID_SIZE,) * 2, dtype=np.uint8)
    if kind == "filled":
        return np.full((GRID_SIZE,) * 2, 255, dtype=np.uint8)
    if kind == "speckle":
        return (rng.random((GRID_SIZE,) * 2) < rng.uniform(0.12, 0.55)).astype(np.uint8) * 255
    image = Image.new("L", (GRID_SIZE * SCALE,) * 2)
    draw = ImageDraw.Draw(image)
    stroke = int(rng.integers(SCALE, 2 * SCALE + 1))
    if kind == "diagonal":
        start = (int(rng.uniform(1, 4) * SCALE), int(rng.uniform(1, 4) * SCALE))
        end = (int(rng.uniform(11, 15) * SCALE), int(rng.uniform(11, 15) * SCALE))
        if rng.random() < 0.5:
            start = (start[0], GRID_SIZE * SCALE - start[1])
            end = (end[0], GRID_SIZE * SCALE - end[1])
        draw.line([start, end], fill=255, width=stroke)
    elif kind == "triangle":
        size = rng.uniform(8, 13) * SCALE
        x, y = rng.uniform(0.5, 2) * SCALE, rng.uniform(0.5, 2) * SCALE
        vertices = [(x + size / 2, y), (x + size, y + size), (x, y + size)]
        draw.line(vertices + vertices[:1], fill=255, width=stroke, joint="curve")
    elif kind == "blob":
        x, y = rng.uniform(1, 5) * SCALE, rng.uniform(1, 5) * SCALE
        size = rng.uniform(6, 10) * SCALE
        draw.ellipse((x, y, x + size, y + size), fill=255)
    elif kind == "scribble":
        points = [tuple(point) for point in rng.uniform(1, 15, (6, 2)) * SCALE]
        draw.line(points, fill=255, width=stroke, joint="curve")
    else:
        raise ValueError(f"Unknown negative kind: {kind}")
    return np.array(image.resize((GRID_SIZE,) * 2, Image.Resampling.LANCZOS))


def generate_dataset(output, split, per_case=50, seed=None, exclude=None):
    if per_case < 1:
        raise ValueError("per_case must be positive")
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    seed = (1729 if split == "dev" else 8675309) if seed is None else seed
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError(f"Dataset already exists at {output}; choose a new directory to preserve it")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    excluded = set()
    excluded_id = None
    if exclude:
        manifest = json.loads((Path(exclude) / "manifest.json").read_text())
        excluded = {row["sha256"] for row in manifest["samples"]}
        excluded_id = manifest["dataset_id"]
    seen = set(excluded)
    rng = np.random.default_rng(seed)
    rows, images, targets = [], [], []

    def add(pixels, label, profile, parameters):
        digest = pixel_hash(pixels)
        if digest in seen:
            return False
        seen.add(digest)
        filename = f"images/{len(rows):05d}_{label}.png"
        Image.fromarray(pixels).save(output / filename)
        rows.append({"file": filename, "label": label, "profile": profile,
                     "sha256": digest, "parameters": parameters})
        images.append(pixels)
        targets.append(ALL_LABELS.index(label))
        return True

    for label in LABELS:
        for profile in PROFILES:
            count = 0
            for _ in range(per_case * 100):
                pixels, params = render_shape(label, rng, profile)
                count += add(pixels, label, profile, params)
                if count == per_case:
                    break
            if count != per_case:
                raise ValueError("Could not produce enough distinct examples")
    unknown_count = per_case * len(LABELS)
    count = 0
    for kind in ("blank", "filled"):
        if count < unknown_count:
            count += add(render_unknown(kind, rng), "unknown", kind, {})
    kinds = ("speckle", "diagonal", "triangle", "blob", "scribble")
    for attempt in range(unknown_count * 100):
        if count == unknown_count:
            break
        kind = kinds[attempt % len(kinds)]
        count += add(render_unknown(kind, rng), "unknown", kind, {})
    if count != unknown_count:
        raise ValueError("Could not produce enough distinct negative examples")
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row["sha256"] + row["label"] + row["profile"]).encode())
    manifest = {"version": DATASET_VERSION, "dataset_id": digest.hexdigest(),
                "split": split, "seed": seed, "per_case": per_case,
                "count": len(rows), "grid_size": GRID_SIZE,
                "excluded_dataset_id": excluded_id, "excluded_images": len(excluded),
                "renderer": "Pillow vector primitives at 8x resolution, Lanczos downsampling",
                "samples": rows}
    np.savez_compressed(output / "pixels.npz", pixels=np.stack(images),
                        targets=np.array(targets, dtype=np.int16))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "pixels.npz", allow_pickle=False) as arrays:
        pixels, targets = arrays["pixels"], arrays["targets"]
    rows = manifest["samples"]
    if pixels.shape != (len(rows), GRID_SIZE, GRID_SIZE) or targets.shape != (len(rows),):
        raise ValueError("Dataset shape does not match its manifest")
    digest = hashlib.sha256()
    for i, row in enumerate(rows):
        if pixel_hash(pixels[i]) != row["sha256"] or ALL_LABELS[int(targets[i])] != row["label"]:
            raise ValueError("Dataset does not match its manifest")
        digest.update((row["sha256"] + row["label"] + row["profile"]).encode())
    if digest.hexdigest() != manifest["dataset_id"]:
        raise ValueError("Dataset fingerprint mismatch")
    return pixels, targets, manifest


def load_image(path, invert=False):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white" if invert else "black")
        image = Image.alpha_composite(background, image).convert("L")
    if invert:
        image = ImageOps.invert(image)
    image = ImageOps.pad(image, (GRID_SIZE, GRID_SIZE), method=Image.Resampling.LANCZOS, color=0)
    return np.asarray(image, dtype=np.float32) / 255
