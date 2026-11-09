import argparse
import json
from pathlib import Path

from .dataset import generate_dataset, load_image
from .evaluate import evaluate
from .network import Network

DEFAULT_MODEL = Path("artifacts/model.npz")


def main():
    parser = argparse.ArgumentParser(description="Construct and test a visual neural network without training.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Construct weights directly from geometry")
    build.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    dataset = commands.add_parser("dataset", help="Generate independently drawn labeled PNGs")
    dataset.add_argument("--split", choices=("dev", "test"), required=True)
    dataset.add_argument("--output", type=Path)
    dataset.add_argument("--per-case", type=int, default=50, help="Examples per shape and difficulty; total is 25 × this")
    dataset.add_argument("--seed", type=int)
    dataset.add_argument("--exclude", type=Path, help="Exclude exact images from an earlier dataset")
    check = commands.add_parser("evaluate", help="Evaluate frozen weights and write JSON, HTML and image reports")
    check.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="Recognize a PNG/JPEG; by default a light shape on a dark background")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--invert", action="store_true", help="Use for dark ink on a light background")
    demo = commands.add_parser("demo", help="Open a local drawing lab (Ctrl-C to stop)")
    demo.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--open", action="store_true", help="Open the browser automatically")
    args = parser.parse_args()
    try:
        if args.command == "build":
            model = Network.build()
            model.save(args.output)
            print(json.dumps(model.stats(), indent=2))
            print(f"Saved weights to {args.output}")
        elif args.command == "dataset":
            output = args.output or Path("artifacts/data") / args.split
            result = generate_dataset(output, args.split, args.per_case, args.seed, args.exclude)
            print(f"Saved {result['count']:,} unique {args.split} images to {output}")
            print(f"Dataset: {result['dataset_id']}")
        elif args.command == "evaluate":
            result = evaluate(Network.load(args.model), args.data, args.output)
            print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['samples']})")
            print(f"Known shapes: {result['known_shape_accuracy']:.1%}")
            print(f"Unknown rejection: {result['unknown_recall']:.1%}")
            print(f"Report: {args.output / 'report.html'}")
        elif args.command == "predict":
            result = Network.load(args.model).predict(load_image(args.image, args.invert))
            result.pop("template")
            print(json.dumps(result, indent=2))
        elif args.command == "demo":
            from .web import serve
            serve(Network.load(args.model), args.port, args.open)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
