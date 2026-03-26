"""Unified training launcher for baseline and AutoVLA tracks."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a training track.")
    parser.add_argument(
        "--track",
        choices=("baseline", "autovla"),
        required=True,
        help="Training stack to run.",
    )
    args = parser.parse_args()

    if args.track == "baseline":
        from train import main as run_baseline

        run_baseline()
        return

    from train_autovla import main as run_autovla

    run_autovla()


if __name__ == "__main__":
    main()
