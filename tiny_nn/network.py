"""Compile geometric prototypes into an affine/ReLU/max-pooling network.

A fixed bank of 8 ReLUs per pixel approximates x² by linear interpolation.
Its outputs sum to E(x), the image's squared intensity. For prototype t,
detector weights are [2*t, -1]/sum(t²), applied to [x, E(x)], with zero bias.
The score approximates 1 - sum((x-t)²)/sum(t²), tolerating antialiased edges.
ReLU discards negative matches, then max pooling gives one score per class.
The two fixed rejection thresholds are engineering choices, not learned values.
This is template matching expressed as a neural network, not a trained CNN.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from . import GRID_SIZE, LABELS

MODEL_VERSION = 2
ENERGY_STEPS = 8


def geometric_templates():
    """Construct prototypes analytically; deliberately do not use dataset.py."""
    for class_id, label in enumerate(LABELS):
        for size in range(7, 14):
            strokes = (1, 2) if label in ("circle", "square") else (1, 2, 3)
            for stroke in strokes:
                if label == "horizontal":
                    mask = np.ones((stroke, size), dtype=np.float32)
                elif label == "vertical":
                    mask = np.ones((size, stroke), dtype=np.float32)
                else:
                    yy, xx = np.mgrid[:size, :size]
                    if label == "square":
                        edge = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy])
                        mask = edge < stroke
                    elif label == "circle":
                        radius = size / 2
                        # Pixel coverage from analytic subpixel samples, not
                        # the independent dataset's Pillow ellipse renderer.
                        mask = np.zeros((size, size), dtype=np.float32)
                        for dy in (0.125, 0.375, 0.625, 0.875):
                            for dx in (0.125, 0.375, 0.625, 0.875):
                                distance = np.hypot(xx + dx - radius, yy + dy - radius)
                                mask += ((distance <= radius) & (distance >= radius - stroke)) / 16
                    else:
                        # Even-width bars have two possible half-pixel alignments
                        # through the bank's integer translations.
                        start = (size - stroke) // 2
                        mask = ((xx >= start) & (xx < start + stroke)) | (
                            (yy >= start) & (yy < start + stroke)
                        )
                height, width = mask.shape
                for top in range(GRID_SIZE - height + 1):
                    for left in range(GRID_SIZE - width + 1):
                        template = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                        template[top:top + height, left:left + width] = mask
                        yield class_id, template.ravel()


@dataclass(frozen=True)
class Network:
    weights: np.ndarray
    biases: np.ndarray
    class_ids: np.ndarray
    min_score: float = 0.55
    min_margin: float = 0.08
    energy_knots: np.ndarray = field(default_factory=lambda: np.arange(ENERGY_STEPS, dtype=np.float32) / ENERGY_STEPS)
    energy_coefficients: np.ndarray = field(default_factory=lambda: np.array([1, *([2] * (ENERGY_STEPS - 1))], dtype=np.float32) / ENERGY_STEPS)

    def __post_init__(self):
        count = len(self.class_ids)
        if self.weights.shape != (count, GRID_SIZE * GRID_SIZE + 1):
            raise ValueError("Invalid weight dimensions")
        if self.biases.shape != (count,) or self.class_ids.shape != (count,):
            raise ValueError("Invalid bias or class dimensions")
        if set(self.class_ids.tolist()) != set(range(len(LABELS))):
            raise ValueError("Every shape must have a template bank")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.biases).all():
            raise ValueError("Weights and biases must be finite")
        if self.energy_knots.shape != (ENERGY_STEPS,) or self.energy_coefficients.shape != (ENERGY_STEPS,):
            raise ValueError("Invalid intensity feature bank")
        if not np.isfinite(self.energy_knots).all() or not np.isfinite(self.energy_coefficients).all():
            raise ValueError("Intensity feature weights must be finite")
        if not (0 < self.min_score <= 1 and 0 <= self.min_margin <= 1):
            raise ValueError("Invalid rejection thresholds")
        for array in (self.weights, self.biases, self.class_ids, self.energy_knots, self.energy_coefficients):
            array.flags.writeable = False

    @classmethod
    def build(cls):
        entries = list(geometric_templates())
        class_ids = np.array([entry[0] for entry in entries], dtype=np.int16)
        templates = np.stack([entry[1] for entry in entries])
        energy = (templates ** 2).sum(axis=1, keepdims=True)
        weights = np.concatenate([2 * templates, -np.ones((len(templates), 1))], axis=1) / energy
        weights = weights.astype(np.float32)
        return cls(weights, np.zeros(len(entries), dtype=np.float32), class_ids)

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "version": MODEL_VERSION, "labels": LABELS, "grid_size": GRID_SIZE,
            "min_score": self.min_score, "min_margin": self.min_margin,
        }, sort_keys=True).encode())
        for value in (self.weights, self.biases, self.class_ids, self.energy_knots, self.energy_coefficients):
            digest.update(value.astype(value.dtype.newbyteorder("<")).tobytes())
        return digest.hexdigest()

    def stats(self):
        return {
            "model_id": self.fingerprint,
            "grid_size": GRID_SIZE,
            "inputs": GRID_SIZE ** 2,
            "hidden_neurons": len(self.class_ids) + GRID_SIZE ** 2 * ENERGY_STEPS + 1,
            "template_neurons": len(self.class_ids),
            "intensity_neurons": GRID_SIZE ** 2 * ENERGY_STEPS,
            "intensity_pool_neurons": 1,
            "class_outputs": len(LABELS),
            "weights": self.weights.size + 2 * GRID_SIZE ** 2 * ENERGY_STEPS,
            "detector_weights": self.weights.size,
            "biases": self.biases.size + GRID_SIZE ** 2 * ENERGY_STEPS + 1,
            "weight_count_note": "Logical connections, including zero detector weights and repeated shared intensity weights; max pooling has no weights.",
            "trainable_parameters": 0,
            "templates_per_class": {label: int(np.sum(self.class_ids == i)) for i, label in enumerate(LABELS)},
            "min_score": self.min_score,
            "min_margin": self.min_margin,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, biases=self.biases,
                            class_ids=self.class_ids, version=MODEL_VERSION,
                            labels=np.array(LABELS), grid_size=GRID_SIZE,
                            min_score=self.min_score, min_margin=self.min_margin,
                            energy_knots=self.energy_knots, energy_coefficients=self.energy_coefficients,
                            fingerprint=self.fingerprint)
        path.with_suffix(".json").write_text(json.dumps(self.stats(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != MODEL_VERSION or int(data["grid_size"]) != GRID_SIZE:
                raise ValueError("Unsupported model format")
            if tuple(data["labels"].tolist()) != LABELS:
                raise ValueError("Model labels do not match this version")
            model = cls(data["weights"], data["biases"], data["class_ids"],
                        float(data["min_score"]), float(data["min_margin"]),
                        data["energy_knots"], data["energy_coefficients"])
            if str(data["fingerprint"]) != model.fingerprint:
                raise ValueError("Model fingerprint mismatch")
            return model

    @staticmethod
    def _inputs(pixels):
        values = np.asarray(pixels, dtype=np.float32)
        if values.shape == (GRID_SIZE, GRID_SIZE):
            values = values.reshape(1, -1)
        elif values.ndim == 3 and values.shape[1:] == (GRID_SIZE, GRID_SIZE):
            values = values.reshape(len(values), -1)
        elif values.ndim == 1 and values.shape[0] == GRID_SIZE ** 2:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != GRID_SIZE ** 2 or not len(values):
            raise ValueError("Expected a 16×16 image or a batch of 16×16 images")
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("Pixel values must be finite numbers from 0 to 1")
        return values

    def forward(self, pixels, batch_size=128):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        values = self._inputs(pixels)
        scores = np.empty((len(values), len(LABELS)), dtype=np.float32)
        winners = np.empty_like(scores, dtype=np.int32)
        groups = [np.flatnonzero(self.class_ids == i) for i in range(len(LABELS))]
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            intensity = np.maximum(batch[:, :, None] - self.energy_knots, 0)
            energy = (intensity @ self.energy_coefficients).sum(axis=1, keepdims=True)
            features = np.concatenate([batch, energy], axis=1)
            hidden = np.maximum(features @ self.weights.T + self.biases, 0)
            for i, indices in enumerate(groups):
                best = np.argmax(hidden[:, indices], axis=1)
                winners[start:start + len(hidden), i] = indices[best]
                scores[start:start + len(hidden), i] = hidden[np.arange(len(hidden)), indices[best]]
        return np.clip(scores, 0, 1), winners

    def decisions(self, scores):
        winners = scores.argmax(axis=1)
        ordered = np.sort(scores, axis=1)
        accepted = (ordered[:, -1] >= self.min_score) & (
            ordered[:, -1] - ordered[:, -2] >= self.min_margin
        )
        return np.where(accepted, winners, len(LABELS))

    def predict(self, pixels):
        values = self._inputs(pixels)
        if len(values) != 1:
            raise ValueError("predict accepts one image; use forward for a batch")
        scores, detectors = self.forward(values)
        best = int(scores[0].argmax())
        decision = int(self.decisions(scores)[0])
        ordered = np.sort(scores[0])
        margin = float(ordered[-1] - ordered[-2])
        reason = "recognized"
        if scores[0, best] < self.min_score:
            reason = "weak_match"
        elif margin < self.min_margin:
            reason = "ambiguous"
        detector = int(detectors[0, best])
        return {
            "label": LABELS[decision] if decision < len(LABELS) else "unknown",
            "reason": reason,
            "best_candidate": LABELS[best],
            "match_score": float(scores[0, best]),
            "margin": margin,
            "scores": {label: float(scores[0, i]) for i, label in enumerate(LABELS)},
            "template_index": detector,
            "template": self.template(detector).tolist(),
        }

    def template(self, index):
        return (-self.weights[index, :-1] / (2 * self.weights[index, -1])).reshape(GRID_SIZE, GRID_SIZE)
