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
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume a previously detached navigator run by pointing at its output "
            "directory (must contain navigator_checkpoint.json)."
        ),
    )
    args = parser.parse_args()

    run_pipeline(args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
