"""
Benchmark script for comparing MAPF-GPT baseline vs. DecentralizedWrapper.

Runs the same eval suite as benchmark.py but injects MAPF-GPT-Safe alongside
the existing baseline algorithms so results appear side-by-side in the same
output files.

Usage
-----
# Default wrapper params (horizon=3, lambda_2=1.0, …)
python benchmark_safe.py

# Tune wrapper hyperparameters
python benchmark_safe.py --horizon 5 --lambda_2 2.0 --alpha 0.5 --priority_scheme random

# Run only specific eval suites
python benchmark_safe.py --folders 01-random 02-mazes

# Skip baseline (wrapper only)
python benchmark_safe.py --no_baseline

Arguments
---------
All wrapper hyperparameters correspond directly to DecentralizedWrapperConfig fields.
"""

import argparse
import collections
import copy
import json
from pathlib import Path

import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from gpt.safe_action_wrapper import (
    DecentralizedWrapperAlgo,
    DecentralizedWrapperConfig,
)

PROJECT_NAME = "Benchmark-Safe"
BASE_PATH = Path("eval_configs")

ALL_FOLDERS = [
    "01-random",
    "02-mazes",
    "03-warehouse",
    "04-movingai",
    "05-puzzles",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark DecentralizedWrapper vs. MAPF-GPT baseline"
    )
    p.add_argument(
        "--folders",
        nargs="+",
        default=ALL_FOLDERS,
        choices=ALL_FOLDERS,
        metavar="FOLDER",
        help=f"Eval suites to run (default: all). Choices: {ALL_FOLDERS}",
    )
    p.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the baseline MAPF-GPT algorithms from the YAML configs.",
    )
    p.add_argument("--disable_wandb", action="store_true")

    # Wrapper hyperparameters
    p.add_argument(
        "--path_to_weights",
        default=None,
        help="Override model weights path (default: use path from YAML config).",
    )
    p.add_argument(
        "--priority_scheme",
        default="index",
        choices=["index", "random"],
        help="Priority assignment: 'index' (agent 0 = highest) or 'random'.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Steps to simulate forward. 1 = single-step hard mask, >1 = risk map.",
    )
    p.add_argument(
        "--sim_num_agents",
        type=int,
        default=1,
        help="Agent slots in reduced context window for neighbor simulation.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Pruning threshold for trajectory tree (actions below this are dropped).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Decay factor for implicit risk propagation along goal paths.",
    )
    p.add_argument(
        "--lambda_1",
        type=float,
        default=1.0,
        help="Weight for policy log-probability in cost function.",
    )
    p.add_argument(
        "--lambda_2",
        type=float,
        default=1.0,
        help="Weight for risk map in cost function.",
    )
    p.add_argument(
        "--sequential_simulation",
        action="store_true",
        help="Simulate hp neighbors in priority order (default: independent).",
    )
    p.add_argument(
        "--conflict_radius",
        type=int,
        default=None,
        help=(
            "Chebyshev radius for trajectory tree simulation. Only hp neighbors "
            "within this distance trigger forward passes. Agents between "
            "conflict_radius and agents_radius are still in the ego context window "
            "but generate no simulation cost. None = use agents_radius (no filter)."
        ),
    )
    return p.parse_args()


def build_safe_algo_entry(baseline_entry: dict, args) -> dict:
    """
    Build a YAML-style algorithm config dict for MAPF-GPT-Safe, copying the
    parallel backend and process count from the baseline entry and overriding
    with any CLI-specified wrapper params.
    """
    entry = {
        "name": "MAPF-GPT-Safe",
        "parallel_backend": baseline_entry.get("parallel_backend", "balanced_dask"),
        "num_process": baseline_entry.get("num_process", 4),
        "path_to_weights": args.path_to_weights or baseline_entry.get("path_to_weights"),
        "priority_scheme": args.priority_scheme,
        "horizon": args.horizon,
        "sim_num_agents": args.sim_num_agents,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "lambda_1": args.lambda_1,
        "lambda_2": args.lambda_2,
        "sequential_simulation": args.sequential_simulation,
        "conflict_radius": args.conflict_radius,
    }
    return entry


def inject_safe_algorithms(evaluation_config: dict, args) -> dict:
    """
    For each baseline algorithm in evaluation_config['algorithms'], add a
    corresponding MAPF-GPT-Safe entry that uses the same weights and backend.
    If --no_baseline is set, remove the baseline entries.
    """
    cfg = copy.deepcopy(evaluation_config)
    original_algos = cfg["algorithms"]
    new_algos = {}

    for algo_name, algo_cfg in original_algos.items():
        if not args.no_baseline:
            new_algos[algo_name] = algo_cfg

        # Derive a safe-variant name, e.g. "MAPF-GPT-2M" -> "Safe-2M"
        suffix = algo_name.replace("MAPF-GPT", "").strip("-") or "2M"
        safe_name = f"Safe-{suffix}" if suffix else "Safe"

        new_algos[safe_name] = build_safe_algo_entry(algo_cfg, args)

    cfg["algorithms"] = new_algos
    return cfg


def ensure_weights(eval_config):
    """Pre-download any weights referenced in the config."""
    for algo_cfg in eval_config["algorithms"].values():
        ToolboxRegistry.create_algorithm(algo_cfg["name"], **algo_cfg)


def main():
    args = parse_args()

    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_eval_env, Environment)
    ToolboxRegistry.register_algorithm(
        "MAPF-GPT", MAPFGPTInference, MAPFGPTInferenceConfig
    )
    ToolboxRegistry.register_algorithm(
        "MAPF-GPT-Safe", DecentralizedWrapperAlgo, DecentralizedWrapperConfig
    )

    print(f"Running eval suites: {args.folders}")
    print(
        f"Wrapper config: horizon={args.horizon} epsilon={args.epsilon} "
        f"alpha={args.alpha} lambda_1={args.lambda_1} lambda_2={args.lambda_2} "
        f"priority={args.priority_scheme}"
    )
    print(f"Baseline included: {not args.no_baseline}")
    print()

    for folder in args.folders:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path) as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        evaluation_config = inject_safe_algorithms(evaluation_config, args)

        ensure_weights(evaluation_config)

        eval_dir = BASE_PATH / folder
        initialize_wandb(evaluation_config, eval_dir, args.disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)

    print_collision_summary(args.folders)


def print_collision_summary(folders):
    """
    Read per-algo JSON result files and print a summary table of collision stats.
    Only shows algorithms that have collision metrics (i.e. Safe variants).
    """
    COLLISION_KEYS = ("collision_vertex", "collision_edge", "collision_total")

    # Accumulate stats: algo -> num_agents -> list of per-episode values
    stats: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    for folder in folders:
        eval_dir = BASE_PATH / folder
        for json_file in sorted(eval_dir.glob("*.json")):
            with open(json_file) as f:
                results = json.load(f)
            for entry in results:
                metrics = entry.get("metrics", {})
                if not any(k in metrics for k in COLLISION_KEYS):
                    continue
                algo = entry.get("algorithm", json_file.stem)
                num_agents = entry.get("env_grid_search", {}).get("num_agents", "?")
                for k in COLLISION_KEYS:
                    if k in metrics:
                        stats[algo][num_agents][k].append(metrics[k])

    if not stats:
        print("\n[Collision Summary] No collision metrics found (only Safe variants track collisions).")
        return

    print("\n" + "=" * 72)
    print("Collision Summary (mean per episode)")
    print("=" * 72)
    header = f"{'Algorithm':<20} {'agents':>6}  {'vertex':>8}  {'edge':>8}  {'total':>8}"
    print(header)
    print("-" * 72)
    for algo in sorted(stats):
        for num_agents in sorted(stats[algo]):
            row = stats[algo][num_agents]
            def mean(vals):
                return sum(vals) / len(vals) if vals else float("nan")
            v = mean(row.get("collision_vertex", []))
            e = mean(row.get("collision_edge", []))
            t = mean(row.get("collision_total", []))
            print(f"{algo:<20} {num_agents:>6}  {v:>8.2f}  {e:>8.2f}  {t:>8.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
