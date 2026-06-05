import argparse

import yaml

from src.training import run_train, run_evaluate, run_predict
from src.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", help="path to the image or folder (for predict mode)")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        run_train(config)
    elif args.mode == "evaluate":
        run_evaluate(config) 
    else:
        if not args.input:
            parser.error("--input is required for predict mode")
        run_predict(config, args.input)


if __name__ == "__main__":
    main()
