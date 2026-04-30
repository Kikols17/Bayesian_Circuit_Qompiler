from __future__ import annotations

import argparse

from src.pipeline.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Bayesian circuit experiment")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
