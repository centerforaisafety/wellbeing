#!/usr/bin/env python3
"""Run unified wellbeing experiments.

Usage:
    python run_experiments.py --experiments compute_experienced_utility_d3 --models qwen25-32b-instruct
    python run_experiments.py --experiments compute_experienced_utility_d3,compute_self_report_d3 --models qwen25-72b-instruct --slurm
    python run_experiments.py --list_experiments
    python run_experiments.py --list_models
"""

import argparse
import os
import sys
import yaml
from typing import Dict, List, NamedTuple, Optional

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.slurm import submit_slurm_job

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "experiments")


class ExperimentConfig(NamedTuple):
    script_path: str
    description: str = ""
    arguments: Optional[Dict] = None
    num_gpus: Optional[int] = None


def load_yaml_file(path: str) -> Dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_allowed_models() -> Dict[str, Dict]:
    """Load model configurations from models.yaml."""
    return load_yaml_file(os.path.join(CONFIGS_DIR, "models.yaml"))


def get_allowed_experiments() -> Dict[str, ExperimentConfig]:
    """Load experiment configurations from experiments.yaml."""
    data = load_yaml_file(os.path.join(CONFIGS_DIR, "experiments.yaml"))
    return {
        name: ExperimentConfig(
            script_path=config["script_path"],
            description=config.get("description", ""),
            arguments=config.get("arguments", {}),
            num_gpus=config.get("num_gpus"),
        )
        for name, config in data.items()
    }


def replace_template_values(config: Dict, model_key: str, model_config: Dict) -> Dict:
    """Replace <model_key> template in config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            value = value.replace("<model_key>", model_key)
            if key == "system_message" and not model_config.get("accepts_system_message", True):
                value = None
        result[key] = value
    return result


def get_gpu_count(model_config: Dict, experiment_config: Optional[ExperimentConfig] = None,
                  override: Optional[int] = None) -> int:
    """Get GPU count with priority: override > experiment > model > 0."""
    if override is not None:
        return override
    if experiment_config and experiment_config.num_gpus is not None:
        return experiment_config.num_gpus
    return model_config.get("gpu_count", 0)


def run_locally(script_path: str, experiment_args: Optional[Dict] = None,
                additional_args: Optional[List[str]] = None) -> None:
    """Run experiment locally."""
    import subprocess

    script_abs = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs)
    script_name = os.path.basename(script_abs)
    original_dir = os.getcwd()

    try:
        os.chdir(script_dir)
        cmd = ["python", "-u", script_name]
        if experiment_args:
            for key, value in experiment_args.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        if additional_args:
            cmd.extend(additional_args)

        process = subprocess.run(cmd)
        if process.returncode != 0:
            raise RuntimeError(f"Local execution failed for {script_name}")
    finally:
        os.chdir(original_dir)


def list_available_models() -> None:
    """Print available models grouped by type."""
    models = get_allowed_models()
    print("\nAvailable Models:")
    print("-" * 50)
    by_type = {}
    for name, config in models.items():
        by_type.setdefault(config["model_type"], []).append(name)
    for model_type, names in sorted(by_type.items()):
        print(f"\n{model_type.upper()}:")
        for name in sorted(names):
            print(f"  - {name}")


def list_available_experiments() -> None:
    """Print available experiments."""
    experiments = get_allowed_experiments()
    print("\nAvailable Experiments:")
    print("-" * 50)
    for name, config in sorted(experiments.items()):
        print(f"\n{name}:")
        print(f"  Script: {config.script_path}")
        if config.description:
            print(f"  Description: {config.description}")


def main():
    parser = argparse.ArgumentParser(description="Run unified wellbeing experiments")
    parser.add_argument("--experiments", type=str,
                        help="Comma-separated list of experiment names")
    parser.add_argument("--models", type=str,
                        help="Comma-separated list of model keys from models.yaml")
    parser.add_argument("--config", type=str,
                        help="Optional YAML file with argument overrides")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit as SLURM jobs")
    parser.add_argument("--time_limit", type=str, default="04:00:00",
                        help="SLURM time limit (default: 04:00:00)")
    parser.add_argument("--partition", type=str,
                        default=os.environ.get("GPU_PARTITION", "cais"),
                        help="SLURM partition (default: $GPU_PARTITION, else 'cais')")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--list_experiments", action="store_true",
                        help="List available experiments and exit")
    parser.add_argument("--override_gpu_count", type=int,
                        help="Override GPU count for all jobs")
    parser.add_argument("--overwrite_results", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--depends_on", type=str, default=None,
                        help="Comma-separated SLURM job IDs this submission must wait for "
                             "(translates to sbatch --dependency=afterok:...). SLURM mode only.")
    parser.add_argument("--additional_args", nargs=argparse.REMAINDER,
                        help="Additional arguments passed to experiment script")

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    if args.list_experiments:
        list_available_experiments()
        sys.exit(0)

    if not args.experiments or not args.models:
        parser.error("--experiments and --models are required (unless using --list_*)")

    model_keys = [m.strip() for m in args.models.split(",")]
    experiment_names = [e.strip() for e in args.experiments.split(",")]

    if not args.slurm:
        if len(model_keys) > 1 or len(experiment_names) > 1:
            print("Error: Multiple models/experiments require --slurm", file=sys.stderr)
            sys.exit(1)

    experiments = get_allowed_experiments()
    models = get_allowed_models()

    for name in experiment_names:
        if name not in experiments:
            print(f"Error: Unknown experiment '{name}'", file=sys.stderr)
            print("Use --list_experiments to see available experiments", file=sys.stderr)
            sys.exit(1)

    for key in model_keys:
        if key not in models:
            print(f"Error: Unknown model '{key}'", file=sys.stderr)
            print("Use --list_models to see available models", file=sys.stderr)
            sys.exit(1)

    for experiment_name in experiment_names:
        experiment_config = experiments[experiment_name]
        # Resolve script_path relative to experiments/ directory
        resolved_script = os.path.join(EXPERIMENTS_DIR, experiment_config.script_path)

        for model_key in model_keys:
            model_config = models[model_key]
            gpu_count = get_gpu_count(model_config, experiment_config, args.override_gpu_count)

            # Build experiment arguments with template substitution
            experiment_args = {}
            if experiment_config.arguments:
                experiment_args = replace_template_values(
                    experiment_config.arguments, model_key, model_config
                )

            # Merge config file overrides
            if args.config:
                overrides = load_yaml_file(args.config)
                overrides = replace_template_values(overrides, model_key, model_config)
                experiment_args.update(overrides)

            # Check for existing results
            save_dir = experiment_args.get("save_dir")
            if save_dir:
                script_abs = os.path.abspath(resolved_script)
                if not os.path.isabs(save_dir):
                    save_dir = os.path.join(os.path.dirname(script_abs), save_dir)
                if os.path.isdir(save_dir) and os.listdir(save_dir):
                    if not args.overwrite_results:
                        print(f"Skipping {experiment_name}/{model_key}: "
                              f"results exist in {save_dir} (use --overwrite_results)")
                        continue

            # Submit or run
            if args.slurm:
                job_name = f"wb_{experiment_name}_{model_key}"
                log_dir = os.path.join(SCRIPT_DIR, "slurm_outputs", experiment_name)
                deps = [j.strip() for j in args.depends_on.split(",") if j.strip()] if args.depends_on else None
                submit_slurm_job(
                    script_path=resolved_script,
                    gpu_count=gpu_count,
                    job_name=job_name,
                    time_limit=args.time_limit,
                    partition=args.partition,
                    experiment_args=experiment_args,
                    log_dir=log_dir,
                    depends_on=deps,
                )
            else:
                run_locally(
                    script_path=resolved_script,
                    experiment_args=experiment_args,
                    additional_args=args.additional_args,
                )


if __name__ == "__main__":
    main()
