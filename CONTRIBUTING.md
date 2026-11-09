# Contributing to Tiny NN

The central experiment is simple: construct a useful neural network's weights from known rules, without fitting them to examples.

Useful changes include new geometric shape families, clearer explanations of a detector, better independent test generators, and improvements to the drawing lab.

## Get started

```sh
git clone https://github.com/ottobonilla95/tiny-nn.git
cd tiny-nn
uv sync
uv run tiny-nn build
uv run python -m unittest discover -s tests -v
uv run tiny-nn demo --open
```

## Keep the experiment honest

- Derive weights from rules and document the calculation. If you want to explore training, label it as a separate experiment.
- Keep dataset rendering independent of the model's template compiler.
- Use development examples to investigate failures. Once you inspect test results to change the model, treat those examples as development data and evaluate the final design on a fresh test set.
- Report rejected known shapes and incorrectly accepted unknowns alongside overall accuracy.
- Preserve the baseline in `results/baseline/`; record later experiments separately with their seeds and model fingerprints.

## Before opening a pull request

Run the automated tests and the relevant CLI commands. For model changes, include a description of the mathematical change, development results, and a fresh evaluation of the final model. For interface changes, check drawing, erasing, keyboard input, and a mobile viewport.

Keep generated weights, datasets, environments, and build outputs out of the commit. Small images or result files that explain a finding are welcome.

By contributing, you agree that your contributions may be distributed under the project's [MIT license](LICENSE).
