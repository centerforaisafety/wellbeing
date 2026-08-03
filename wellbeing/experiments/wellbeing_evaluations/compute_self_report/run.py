#!/usr/bin/env python3
"""
Compute Self-Report wellbeing ratings.

Usage:
    python run.py --model_key qwen25-7b-instruct --dataset d2_negative_500 --save_dir results/sr_d2/qwen25-7b-instruct
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"


def load_dataset_config(dataset_name: str, model_key: str) -> dict:
    with open(DATASETS_CONFIG, "r") as f:
        all_datasets = yaml.safe_load(f)
    if dataset_name not in all_datasets:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    ds = all_datasets[dataset_name]
    option_files = [PROJECT_ROOT / p.replace("<model_key>", model_key) for p in ds["option_files"]]
    result = {"name": dataset_name, "option_files": option_files}
    if "image_manifest" in ds:
        result["image_manifest"] = PROJECT_ROOT / ds["image_manifest"]
    if "audio_manifest" in ds:
        result["audio_manifest"] = PROJECT_ROOT / ds["audio_manifest"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute Self-Report")
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--battery_path", type=str, default=None,
                        help="Path to SR battery JSON (default: standard battery)")
    parser.add_argument("--strict_parse", action="store_true",
                        help="Only accept a bare number as a rating; hedging refusals "
                             "that mention the scale parse to None instead of 1")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw completions under results[exp_id]['per_question_raw']")
    args = parser.parse_args()

    from metrics.compute_metrics import (
        run_self_report,
        SR_DEFAULT_OPTIONS_FILE,
        DEFAULT_MODELS_CONFIG,
    )

    ds = load_dataset_config(args.dataset, args.model_key)
    options_file = ds["option_files"][0] if ds["option_files"] else SR_DEFAULT_OPTIONS_FILE

    os.makedirs(args.save_dir, exist_ok=True)

    sr_kwargs = dict(
        model_key=args.model_key,
        options_file=options_file,
        models_config_path=DEFAULT_MODELS_CONFIG,
        checkpoint_dir=args.save_dir,
        image_manifest_path=ds.get("image_manifest"),
        audio_manifest_path=ds.get("audio_manifest"),
        strict_parse=args.strict_parse,
        save_raw=args.save_raw,
    )
    if args.battery_path:
        sr_kwargs["battery_path"] = args.battery_path

    sr_output = run_self_report(**sr_kwargs)

    out_path = os.path.join(args.save_dir, "self_report_results.json")
    with open(out_path, "w") as f:
        json.dump(sr_output, f, indent=2)
    logger.info("Saved: %s", out_path)
    logger.info("SR complete for %s / %s", args.model_key, args.dataset)


if __name__ == "__main__":
    main()
