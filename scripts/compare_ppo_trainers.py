"""Compare matched custom-PPO and SB3-PPO artifact directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


MATCHED_PPO_FIELDS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "n_epochs",
    "n_steps",
    "batch_size",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy(distribution: Mapping[str, Any]) -> float | None:
    probabilities = [float(value) for value in distribution.values() if float(value) > 0]
    if not probabilities:
        return None
    return -sum(value * math.log(value) for value in probabilities)


def _run_summary(path: Path) -> dict[str, Any]:
    config = _read_json(path / "config.json")
    metrics = _read_json(path / "metrics.json")
    evaluation = metrics["final_evaluation"]
    overall = evaluation["overall"]
    progress = metrics.get("curriculum_progress") or {}
    session = config.get("training_session", {})
    elapsed = float(session.get("elapsed_seconds", 0.0))
    timesteps = int(config.get("actual_timesteps", 0))
    return {
        "path": str(path),
        "trainer": config.get("trainer", "custom"),
        "seed": config.get("seed"),
        "requested_timesteps": config.get("total_timesteps"),
        "actual_timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "timesteps_per_second": timesteps / elapsed if elapsed > 0 else None,
        "curriculum_depth": progress.get("current_depth"),
        "solve_rate": overall.get("solve_rate"),
        "timeout_rate": overall.get("timeout_rate"),
        "average_solution_length": overall.get("average_solution_length"),
        "inverse_move_rate": overall.get("inverse_move_rate"),
        "action_entropy": _entropy(overall.get("action_distribution", {})),
        "action_distribution": overall.get("action_distribution", {}),
        "config": config,
    }


def compare_runs(custom_dir: Path, sb3_dir: Path) -> dict[str, Any]:
    """Return parity checks and outcome deltas for two completed runs."""

    custom = _run_summary(custom_dir)
    sb3 = _run_summary(sb3_dir)
    custom_config = custom.pop("config")
    sb3_config = sb3.pop("config")
    checks: dict[str, bool] = {
        "seed": custom["seed"] == sb3["seed"],
        "requested_timesteps": (
            custom["requested_timesteps"] == sb3["requested_timesteps"]
        ),
        "curriculum": custom_config.get("curriculum") == sb3_config.get("curriculum"),
    }
    for field in MATCHED_PPO_FIELDS:
        checks[f"ppo.{field}"] = (
            custom_config.get("ppo", {}).get(field)
            == sb3_config.get("ppo", {}).get(field)
        )
    deltas = {
        field: (
            None
            if custom.get(field) is None or sb3.get(field) is None
            else float(sb3[field]) - float(custom[field])
        )
        for field in (
            "curriculum_depth",
            "solve_rate",
            "timeout_rate",
            "average_solution_length",
            "inverse_move_rate",
            "action_entropy",
            "timesteps_per_second",
        )
    }
    return {
        "comparison_valid": all(checks.values()),
        "parity_checks": checks,
        "custom": custom,
        "sb3": sb3,
        "delta_sb3_minus_custom": deltas,
        "interpretation": (
            "Outcome deltas are comparable under the declared controls."
            if all(checks.values())
            else "Do not interpret outcome deltas until failed parity checks are resolved."
        ),
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    """Render a compact, reviewable experiment report."""

    custom = comparison["custom"]
    sb3 = comparison["sb3"]
    rows = []
    for field in (
        "actual_timesteps",
        "elapsed_seconds",
        "timesteps_per_second",
        "curriculum_depth",
        "solve_rate",
        "timeout_rate",
        "average_solution_length",
        "inverse_move_rate",
        "action_entropy",
    ):
        rows.append(f"| {field} | {custom.get(field)} | {sb3.get(field)} |")
    failed = [
        name for name, passed in comparison["parity_checks"].items() if not passed
    ]
    parity = "passed" if not failed else f"failed: {', '.join(failed)}"
    return "\n".join(
        [
            "# Custom PPO vs Stable-Baselines3 PPO",
            "",
            f"Configuration parity: **{parity}**",
            "",
            "| Metric | Custom PPO | SB3 PPO |",
            "|---|---:|---:|",
            *rows,
            "",
            comparison["interpretation"],
            "This report validates the comparison pipeline. Learning conclusions require a sufficiently large, predeclared training budget and multiple seeds.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("custom_dir", type=Path)
    parser.add_argument("sb3_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_runs(args.custom_dir, args.sb3_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".md":
        args.output.write_text(render_markdown(comparison), encoding="utf-8")
    else:
        args.output.write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"Wrote comparison to {args.output}")
    if not comparison["comparison_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
