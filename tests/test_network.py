import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tiny_nn.dataset import generate_dataset, load_dataset, load_image
from tiny_nn.network import Network


def reference_images():
    """Simple external drawing commands, independent of the weight compiler."""
    shapes = {}
    for label in ("cross", "square", "circle", "horizontal", "vertical"):
        image = Image.new("L", (16, 16))
        draw = ImageDraw.Draw(image)
        if label in ("cross", "horizontal"):
            draw.line((3, 7, 12, 7), fill=255)
        if label in ("cross", "vertical"):
            draw.line((7, 3, 7, 12), fill=255)
        if label == "square":
            draw.rectangle((3, 3, 12, 12), outline=255)
        if label == "circle":
            draw.ellipse((3, 3, 12, 12), outline=255)
        shapes[label] = np.asarray(image, dtype=np.float32) / 255
    return shapes


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Network.build()

    def test_independent_shapes_survive_translation_and_one_missing_pixel(self):
        for label, pixels in reference_images().items():
            for dx, dy in ((0, 0), (-2, 1), (2, -2)):
                shifted = np.roll(pixels, (dy, dx), axis=(0, 1))
                with self.subTest(label=label, shift=(dx, dy)):
                    self.assertEqual(self.model.predict(shifted)["label"], label)
                damaged = shifted.copy()
                damaged.flat[np.flatnonzero(damaged)[0]] = 0
                with self.subTest(label=label, shift=(dx, dy), damage=True):
                    self.assertEqual(self.model.predict(damaged)["label"], label)

    def test_blank_dense_and_diagonal_inputs_are_unknown(self):
        for pixels in (np.zeros((16, 16)), np.ones((16, 16)), np.eye(16)):
            self.assertEqual(self.model.predict(pixels)["label"], "unknown")

    def test_rejection_distinguishes_weak_and_ambiguous_matches(self):
        scores = np.array([[.9, .89, .1, 0, 0], [.5, .1, 0, 0, 0], [.8, .1, 0, 0, 0]])
        np.testing.assert_array_equal(self.model.decisions(scores), [5, 5, 0])

    def test_batch_inference_agrees_with_individual_inference(self):
        images = list(reference_images().values())
        scores, _ = self.model.forward(images, batch_size=2)
        batch_labels = self.model.decisions(scores)
        np.testing.assert_array_equal(batch_labels, np.arange(5))
        for i, pixels in enumerate(images):
            single, _ = self.model.forward(pixels)
            np.testing.assert_allclose(scores[i], single[0], atol=2e-6)

    def test_intensity_relu_bank_has_bounded_square_approximation_error(self):
        x = np.linspace(0, 1, 1001, dtype=np.float32)
        approx = np.maximum(x[:, None] - self.model.energy_knots, 0) @ self.model.energy_coefficients
        error = approx - x * x
        self.assertGreaterEqual(error.min(), -1e-6)
        self.assertLessEqual(error.max(), 1 / (4 * 8 ** 2) + 1e-6)

    def test_save_reload_preserves_predictions_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            self.model.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.fingerprint, self.model.fingerprint)
            pixels = reference_images()["circle"]
            self.assertEqual(loaded.predict(pixels), self.model.predict(pixels))
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays["weights"][0, 0] += 0.25
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Network.load(path)

    def test_invalid_pixel_values_are_rejected(self):
        for pixels in (np.zeros((15, 16)), np.full((16, 16), np.nan), np.full((16, 16), np.inf),
                       np.full((16, 16), -1), np.full((16, 16), 255), np.zeros((0, 256))):
            with self.subTest(shape=pixels.shape):
                with self.assertRaises(ValueError):
                    self.model.predict(pixels)


class DatasetTests(unittest.TestCase):
    def test_datasets_are_reproducible_and_holdout_excludes_dev_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev = generate_dataset(root / "dev", "dev", per_case=1, seed=3)
            again = generate_dataset(root / "again", "dev", per_case=1, seed=3)
            test = generate_dataset(root / "test", "test", per_case=1, seed=3, exclude=root / "dev")
            self.assertEqual(dev["dataset_id"], again["dataset_id"])
            self.assertEqual(test["count"], 25)
            self.assertEqual(test["excluded_dataset_id"], dev["dataset_id"])
            self.assertFalse({r["sha256"] for r in dev["samples"]} & {r["sha256"] for r in test["samples"]})
            pixels, targets, _ = load_dataset(root / "test")
            self.assertEqual(pixels.shape, (25, 16, 16))
            np.testing.assert_array_equal(np.bincount(targets), [4, 4, 4, 4, 4, 5])
            with self.assertRaisesRegex(ValueError, "already exists"):
                generate_dataset(root / "dev", "dev", per_case=1)

    def test_image_polarity_and_alpha_are_handled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pixels = (reference_images()["cross"] * 255).astype(np.uint8)
            Image.fromarray(pixels).save(root / "light.png")
            Image.fromarray(255 - pixels).save(root / "dark.png")
            np.testing.assert_array_equal(load_image(root / "light.png"), load_image(root / "dark.png", invert=True))
            transparent = Image.new("RGBA", (16, 16))
            ImageDraw.Draw(transparent).line((3, 7, 12, 7), fill=(255, 255, 255, 255))
            transparent.save(root / "transparent.png")
            self.assertEqual(Network.build().predict(load_image(root / "transparent.png"))["label"], "horizontal")


if __name__ == "__main__":
    unittest.main()
