"""Evaluate frozen weights and write a self-contained, inspectable report."""

import base64
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ALL_LABELS, LABELS
from .dataset import load_dataset


def image_uri(pixels):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def breakdown(expected, predicted, names):
    result = {}
    for name in sorted(set(names)):
        selected = np.array([value == name for value in names])
        count = int(selected.sum())
        correct = int(np.sum(expected[selected] == predicted[selected]))
        result[name] = {"count": count, "correct": correct, "accuracy": correct / count}
    return result


def evaluate(model, dataset_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise ValueError(f"A report already exists at {output}; use a new output directory")
    pixels, targets, manifest = load_dataset(dataset_dir)
    start = time.perf_counter()
    scores, _ = model.forward(pixels.astype(np.float32) / 255)
    predictions = model.decisions(scores)
    duration = time.perf_counter() - start
    confusion = np.zeros((len(ALL_LABELS), len(ALL_LABELS)), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    profiles = [row["profile"] for row in manifest["samples"]]
    known = targets < len(LABELS)
    unknown = ~known
    accepted = predictions < len(LABELS)
    correct = targets == predictions
    failure_ids = np.flatnonzero(~correct).tolist()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": manifest["split"], "dataset_id": manifest["dataset_id"],
        "dataset_seed": manifest["seed"], "excluded_dataset_id": manifest["excluded_dataset_id"],
        "model": model.stats(), "samples": len(targets),
        "correct": int(correct.sum()), "accuracy": float(correct.mean()),
        "known_shape_accuracy": float(correct[known].mean()),
        "unknown_recall": float(correct[unknown].mean()),
        "known_rejection_rate": float((~accepted[known]).mean()),
        "coverage": float(accepted.mean()),
        "accuracy_when_accepted": float(correct[accepted].mean()) if accepted.any() else None,
        "inference_seconds": duration,
        "labels": ALL_LABELS, "confusion_matrix": confusion.tolist(),
        "by_class": breakdown(targets, predictions, [ALL_LABELS[i] for i in targets]),
        "by_profile": breakdown(targets, predictions, profiles),
        "failures": [{**manifest["samples"][i], "predicted": ALL_LABELS[predictions[i]],
                      "scores": {label: float(scores[i, j]) for j, label in enumerate(LABELS)}}
                     for i in failure_ids],
    }
    # Stress the same clean images with progressively more flipped pixels. This
    # is diagnostic evaluation only; these numbers never feed back into weights.
    clean_indices = [i for i, row in enumerate(manifest["samples"]) if row["profile"] == "clean"]
    clean = pixels[clean_indices].astype(np.float32) / 255
    stress = []
    rng = np.random.default_rng(manifest["seed"] + 414)
    ranks = np.argsort(rng.random((len(clean), 256)), axis=1)
    for count in (0, 3, 8, 13, 26):
        noisy = clean.copy().reshape(-1, 256)
        for row in range(len(noisy)):
            indices = ranks[row, :count]
            noisy[row, indices] = 1 - noisy[row, indices]
        noisy_scores, _ = model.forward(noisy)
        noisy_predictions = model.decisions(noisy_scores)
        stress.append({"flipped_pixels": count, "fraction": count / 256,
                       "accuracy": float(np.mean(noisy_predictions == targets[clean_indices]))})
    result["noise_sweep"] = stress
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    records = [{"file": row["file"], "expected": row["label"], "profile": row["profile"],
                "predicted": ALL_LABELS[predictions[i]], "correct": bool(correct[i]),
                "scores": {label: float(scores[i, j]) for j, label in enumerate(LABELS)}}
               for i, row in enumerate(manifest["samples"])]
    (output / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    write_html(result, pixels, targets, predictions, scores, profiles, output)
    write_montage(result, pixels, targets, predictions, output)
    return result


def write_html(result, pixels, targets, predictions, scores, profiles, output):
    def percent(value):
        return f"{value:.1%}"

    def table(rows):
        return "".join(f"<tr><th>{escape(name.replace('_', ' '))}</th><td>{row['correct']} / {row['count']}</td>"
                       f"<td>{percent(row['accuracy'])}</td></tr>" for name, row in rows.items())

    matrix = ""
    for name, values in zip(ALL_LABELS, result["confusion_matrix"]):
        total = sum(values)
        cells = "".join(f'<td style="background:rgba(62,81,179,{0.07 + .25 * value / max(total, 1):.3f})">{value}</td>'
                        for value in values)
        matrix += f"<tr><th>{name}</th>{cells}</tr>"
    failures = np.flatnonzero(targets != predictions).tolist()
    illustrated_failures = []
    for label in range(len(ALL_LABELS)):
        illustrated_failures.extend(np.flatnonzero((targets == label) & (targets != predictions))[:6].tolist())
    successes = []
    for i in range(len(ALL_LABELS)):
        successes.extend(np.flatnonzero((targets == i) & (predictions == i))[:3].tolist())

    def gallery(indices):
        return "".join(
            f'<figure><img src="{image_uri(pixels[i])}" alt="{ALL_LABELS[targets[i]]} example">'
            f'<figcaption>Expected: <b>{ALL_LABELS[targets[i]]}</b><br>'
            f'Output: <b>{ALL_LABELS[predictions[i]]}</b><br>{escape(profiles[i])}; '
            f'best match {float(scores[i].max()):.2f}</figcaption></figure>' for i in indices)

    stress = "".join(f"<tr><th>{row['flipped_pixels']} / 256</th><td>{percent(row['fraction'])}</td>"
                     f"<td>{percent(row['accuracy'])}</td></tr>" for row in result["noise_sweep"])
    model = result["model"]
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiny NN — {escape(result['split'])} evaluation</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f6;color:#183143;font:16px/1.6 'Avenir Next',Arial,sans-serif}}
main{{max-width:1100px;margin:auto;padding:48px 28px}}h1{{font-size:40px;letter-spacing:-1.5px;margin:0 0 12px}}
h2{{font-size:24px;margin:40px 0 14px}}p{{max-width:78ch}}.summary{{background:#fff;border-radius:16px;padding:24px 30px;margin:28px 0}}
.summary strong{{font-size:38px;color:#354cb2}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:36px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #ced8e2}}td{{text-align:right}}
.scroll{{overflow-x:auto}}.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:20px}}
figure{{margin:0}}img{{width:100%;max-width:170px;image-rendering:pixelated;border:8px solid #fff;border-radius:6px}}
figcaption{{font-size:13px;line-height:1.5}}code{{overflow-wrap:anywhere;font-size:12px}}footer{{margin-top:36px;font-size:13px}}
@media(max-width:650px){{main{{padding:28px 18px}}h1{{font-size:32px}}.columns{{grid-template-columns:1fr}}}}
</style><main>
<h1>Tiny NN: {escape(result['split'])} evaluation</h1>
<p>Weights constructed from geometry. No training, optimizer, pretrained checkpoint, or dataset fitting.
The test images use a separate antialiased vector renderer.</p>
<div class="summary"><strong>{percent(result['accuracy'])}</strong> overall accuracy
<p>{result['correct']:,} of {result['samples']:,} examples correct.
Known shapes: <b>{percent(result['known_shape_accuracy'])}</b>.
Unknown examples rejected: <b>{percent(result['unknown_recall'])}</b>.
Known shapes rejected: <b>{percent(result['known_rejection_rate'])}</b>.</p></div>
<p>{model['inputs']} inputs → {model['intensity_neurons']:,} fixed intensity ReLUs + one energy sum →
{model['template_neurons']:,} ReLU template detectors → 5 max-pool outputs → rejection rule.
{model['weights']:,} fixed connection weights and {model['biases']:,} biases; {model['hidden_neurons']:,} hidden units total.
A match score measures geometric similarity; it is not a probability.</p>
<div class="columns"><section><h2>By shape</h2><table><tr><th>Expected</th><th>Correct / total</th><th>Accuracy</th></tr>
{table(result['by_class'])}</table></section><section><h2>By condition</h2><table><tr><th>Condition</th><th>Correct / total</th><th>Accuracy</th></tr>
{table(result['by_profile'])}</table></section></div>
<h2>Confusion matrix</h2><p>Rows are expected labels. Columns are the network's outputs. “Unknown” includes low matches and ambiguous results.</p>
<div class="scroll"><table><tr><th>Expected ↓ / output →</th>{''.join('<th>' + name + '</th>' for name in ALL_LABELS)}</tr>{matrix}</table></div>
<h2>Noise stress test</h2><p>The same clean examples with an increasing number of randomly selected pixels inverted.</p>
<table><tr><th>Flipped pixels</th><th>Image fraction</th><th>Accuracy</th></tr>{stress}</table>
<h2>Correct examples</h2><div class="gallery">{gallery(successes)}</div>
<h2>Failures ({len(failures):,} total)</h2><p>Up to six failures per expected class, in dataset order. Every failure is recorded in metrics.json.</p>
<div class="gallery">{gallery(illustrated_failures) or '<p>No failures in this sample.</p>'}</div>
<h2>What this establishes</h2><p>These results cover synthetic, isolated, mostly axis-aligned shapes on a 16×16 grid.
They do not establish accuracy on freehand drawings, photographs, arbitrary rotations, overlapping shapes,
or all possible unknown images. Small circles and squares, broken strokes, and thin antialiased lines can be ambiguous.</p>
<p>Every class has the same number of known examples. Unknowns are 20% of this dataset. Template sizes are integers 7 through 13;
the independent renderer samples continuous sizes, positions, and stroke widths. Rejection requires a match of at least
{model['min_score']} and a lead of at least {model['min_margin']} over the next class.</p>
<footer>Model: <code>{model['model_id']}</code><br>Dataset: <code>{result['dataset_id']}</code><br>
Excluded development dataset: <code>{escape(str(result['excluded_dataset_id']))}</code><br>
Seed: {result['dataset_seed']}. Generated: {escape(result['created_utc'])}.</footer></main></html>"""
    (output / "report.html").write_text(content)


def write_montage(result, pixels, targets, predictions, output):
    """A shareable sample image containing both successes and failures."""
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 25)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default(size=25)
        font = ImageFont.load_default(size=14)
    indices = []
    for label in range(len(ALL_LABELS)):
        matches = np.flatnonzero((targets == label) & (targets == predictions))
        if len(matches):
            indices.append(int(matches[0]))
    for label in range(len(ALL_LABELS)):
        indices.extend(np.flatnonzero((targets == label) & (targets != predictions))[:2].tolist())
    columns, cell_w, cell_h = 6, 164, 188
    rows = (len(indices) + columns - 1) // columns
    image = Image.new("RGB", (columns * cell_w + 40, rows * cell_h + 112), "#edf2f6")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), f"Tiny NN / {result['accuracy']:.1%} on {result['samples']:,} {result['split']} examples",
              fill="#183143", font=title_font)
    draw.text((24, 55), "Constructed weights. Correct examples first, followed by actual failures.", fill="#536d80", font=font)
    for position, i in enumerate(indices):
        x, y = 24 + (position % columns) * cell_w, 92 + (position // columns) * cell_h
        thumbnail = Image.fromarray(pixels[i]).convert("RGB").resize((128, 128), Image.Resampling.NEAREST)
        image.paste(thumbnail, (x, y))
        draw.text((x, y + 134), f"Input: {ALL_LABELS[targets[i]]}", fill="#183143", font=font)
        draw.text((x, y + 154), f"Output: {ALL_LABELS[predictions[i]]}",
                  fill="#285a51" if targets[i] == predictions[i] else "#9c3c4d", font=font)
    image.save(output / "examples.png")
