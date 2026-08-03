#!/usr/bin/env python3
"""
Compute Zero-Point estimates from precomputed EU results.

Supports two families of zero-point methods:
  - ComboZP: Prospect-theory combination model (always attempted).
  - SR_ZP: Self-report sigmoid models (requires --sr_data).

Usage:
    python run.py --model_key qwen25-7b-instruct \\
        --save_dir results/zp_d2/qwen25-7b-instruct \\
        --eu_dir results/eu_d2/qwen25-7b-instruct
"""
import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"


def main():
    parser = argparse.ArgumentParser(description="Compute Zero-Point")
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--eu_dir", type=str, required=True,
                        help="Directory containing precomputed EU results")
    parser.add_argument("--hinge", type=str, default="hard",
                        choices=["hard", "expected"],
                        help="Combination-model hinge: 'hard' (means only, the "
                             "released metric) or 'expected' (variance-aware; used "
                             "by the stable AIWI pipeline).")

    args = parser.parse_args()

    from metrics.zero_point import run_zero_point

    os.makedirs(args.save_dir, exist_ok=True)

    run_zero_point(
        model_key=args.model_key,
        utilities_dir=Path(args.eu_dir),
        save_dir=args.save_dir,
        models_config_path=DEFAULT_MODELS_CONFIG,
        domain="experienced",
        skip_yes_no=True,  # We use SR_ZP instead
        hinge=args.hinge,
    )

    logger.info("ZP complete for %s", args.model_key)


if __name__ == "__main__":
    main()
