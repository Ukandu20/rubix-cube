"""Interactive local showcase for curriculum-trained PPO cube policies."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

from showcase.service import (  # noqa: E402
    BenchmarkResult,
    CheckpointInfo,
    RunConfig,
    SolveResult,
    benchmark_to_csv,
    discover_checkpoints,
    load_curriculum_weights,
    load_policy,
    records_to_csv,
    run_benchmark,
    run_solver,
    sample_scramble,
    solve_result_record,
)
from showcase.visualization import cube_net_html  # noqa: E402

ARTIFACT_ROOT = PROJECT_ROOT / "models" / "artifacts"
CURRICULUM_PATH = PROJECT_ROOT / "config" / "curriculum_config_depth_1_10.yaml"


st.set_page_config(
    page_title="PPO Rubik's Cube Showcase",
    page_icon="🧊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_checkpoint_discovery(
    root: str,
    fingerprint: tuple[tuple[str, int], ...],
) -> list[CheckpointInfo]:
    del fingerprint
    return discover_checkpoints(root)


@st.cache_data(show_spinner=False)
def cached_curriculum_weights(path: str) -> dict[int, dict[int, float]]:
    return load_curriculum_weights(path)


@st.cache_resource(show_spinner="Loading PPO checkpoint...")
def cached_policy(path: str, modified_ns: int):
    del modified_ns
    return load_policy(path)


def main() -> None:
    st.title("PPO Rubik's Cube Showcase")
    st.caption(
        "Interactive inference and benchmarking for curriculum-trained policies. "
        "Difficulty is reported as scramble sequence length, not proven optimal depth."
    )
    _initialize_state()

    fingerprint = (
        tuple(
            (path.as_posix(), path.stat().st_mtime_ns)
            for path in ARTIFACT_ROOT.rglob("*")
            if path.name
            in {
                "best_model.pt",
                "final_model.pt",
                "best_model.zip",
                "final_model.zip",
                "curriculum_progress.json",
            }
        )
        if ARTIFACT_ROOT.exists()
        else ()
    )
    checkpoints = cached_checkpoint_discovery(str(ARTIFACT_ROOT), fingerprint)
    compatible = [checkpoint for checkpoint in checkpoints if checkpoint.compatible]
    if not compatible:
        st.error(
            "No compatible PPO checkpoint was found under "
            f"`{ARTIFACT_ROOT}`. Train or copy a trusted local checkpoint first."
        )
        if checkpoints:
            with st.expander("Incompatible artifacts"):
                for checkpoint in checkpoints:
                    st.code(f"{checkpoint.label}: {checkpoint.error}")
        return

    checkpoint = _checkpoint_selector(compatible)
    _checkpoint_status(checkpoint)
    try:
        policy = cached_policy(
            str(checkpoint.path),
            checkpoint.path.stat().st_mtime_ns,
        )
        curriculum_weights = cached_curriculum_weights(str(CURRICULUM_PATH))
    except Exception as exc:
        st.error(f"Could not load the selected checkpoint: {exc}")
        return

    demo_tab, benchmark_tab, history_tab = st.tabs(
        ["Solve demo", "Benchmark", "Session history"]
    )
    with demo_tab:
        _render_demo(policy, checkpoint, curriculum_weights)
    with benchmark_tab:
        _render_benchmark(policy, checkpoint)
    with history_tab:
        _render_history()


def _initialize_state() -> None:
    st.session_state.setdefault("demo_result", None)
    st.session_state.setdefault("demo_history", [])
    st.session_state.setdefault("benchmark_result", None)


def _checkpoint_selector(checkpoints: list[CheckpointInfo]) -> CheckpointInfo:
    st.sidebar.header("Model")
    labels = [checkpoint.label for checkpoint in checkpoints]
    selected_label = st.sidebar.selectbox(
        "Checkpoint",
        labels,
        index=0,
        help="Only trusted local PyTorch checkpoints are listed.",
    )
    return checkpoints[labels.index(selected_label)]


def _checkpoint_status(checkpoint: CheckpointInfo) -> None:
    with st.sidebar:
        st.metric("Validated through length", checkpoint.validated_depth or "None")
        st.metric("Checkpoint curriculum stage", checkpoint.checkpoint_curriculum_depth)
        st.metric("Latest run curriculum stage", checkpoint.run_curriculum_depth)
        st.caption(
            f"{checkpoint.observation_encoding} observations · "
            f"{checkpoint.timesteps:,} checkpoint timesteps"
        )
    if checkpoint.validated_depth < 10:
        st.warning(
            f"This checkpoint contains evaluation evidence only through scramble "
            f"length {checkpoint.validated_depth or 0}. Lengths "
            f"{checkpoint.validated_depth + 1}–10 remain available for exploratory "
            "testing and are not presented as validated capability."
        )


def _render_demo(policy, checkpoint: CheckpointInfo, curriculum_weights) -> None:
    controls, output = st.columns([1, 2], gap="large")
    with controls:
        st.subheader("Run configuration")
        mode_label = st.radio(
            "Scramble sampling",
            ["Exact length", "Uniform range", "Curriculum weighted"],
        )
        mode = {
            "Exact length": "exact",
            "Uniform range": "range",
            "Curriculum weighted": "curriculum",
        }[mode_label]
        if mode == "exact":
            minimum = maximum = st.slider("Scramble length", 1, 10, 3)
            stage = maximum
        else:
            minimum, maximum = st.slider(
                "Scramble length range",
                1,
                10,
                (1, 10),
            )
            stage = (
                st.slider("Curriculum stage", maximum, 10, maximum)
                if mode == "curriculum"
                else maximum
            )

        seed = int(st.number_input("Seed", min_value=0, value=42, step=1))
        auto_budget = st.checkbox(
            "Automatic move limit (2 × sampled length + 1)",
            value=True,
        )
        max_moves = None
        if not auto_budget:
            max_moves = int(
                st.number_input("Moves per attempt", 1, 100, max(3, 2 * maximum + 1))
            )
        max_attempts = st.slider("Attempt limit", 1, 10, 3)
        timeout = st.number_input(
            "Total solver time limit (seconds)",
            min_value=0.1,
            max_value=60.0,
            value=5.0,
            step=0.5,
        )
        temperature = st.slider(
            "Retry sampling temperature",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
        )
        animation_delay = st.slider(
            "Animation delay (seconds)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        run_clicked = st.button("Generate and solve", type="primary")

    if run_clicked:
        try:
            length, scramble, _ = sample_scramble(
                mode=mode,
                minimum=minimum,
                maximum=maximum,
                seed=seed,
                curriculum_weights=curriculum_weights,
                curriculum_stage=stage,
            )
            config = RunConfig(
                max_moves=max_moves,
                max_attempts=max_attempts,
                timeout_seconds=float(timeout),
                temperature=temperature,
                seed=seed,
            )
            with st.spinner("Running policy..."):
                result = run_solver(
                    policy,
                    scramble_length=length,
                    scramble=scramble,
                    config=config,
                )
            st.session_state.demo_result = result
            st.session_state.demo_history.append(
                solve_result_record(result, checkpoint.label)
            )
        except Exception as exc:
            st.error(f"Could not run the solver: {exc}")

    with output:
        result: SolveResult | None = st.session_state.demo_result
        if result is None:
            st.info("Configure a scramble and select **Generate and solve**.")
            return
        _demo_result(result, animation_delay, checkpoint)


def _demo_result(
    result: SolveResult,
    animation_delay: float,
    checkpoint: CheckpointInfo,
) -> None:
    status = st.success if result.solved else st.error
    status("Solved" if result.solved else f"Not solved: {result.termination_reason}")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Scramble length", result.scramble_length)
    metric_columns[1].metric("Attempts", result.attempts_used)
    metric_columns[2].metric("Moves", result.total_moves)
    metric_columns[3].metric(
        "Inference",
        f"{result.inference_seconds * 1000:.2f} ms",
    )
    metric_columns[4].metric(
        "Solver",
        f"{result.solver_seconds * 1000:.2f} ms",
    )
    if result.scramble_length > checkpoint.validated_depth:
        st.warning(
            "This result is outside the checkpoint's validated curriculum depth."
        )

    st.markdown(f"**Scramble:** `{result.scramble}`")
    attempt = result.display_attempt
    st.markdown(
        f"**Displayed attempt {attempt.attempt} ({attempt.mode}):** "
        f"`{' '.join(attempt.moves) or 'No moves'}`"
    )

    frame_index = st.slider(
        "Playback step",
        0,
        len(attempt.frames) - 1,
        len(attempt.frames) - 1,
        key=f"frame_{result.seed}_{result.scramble}_{attempt.attempt}",
    )
    display = st.empty()
    _render_frame(display, attempt.frames[frame_index])
    if st.button("Play animation", disabled=len(attempt.frames) <= 1):
        for frame in attempt.frames:
            _render_frame(display, frame)
            if animation_delay:
                time.sleep(animation_delay)

    with st.expander("Attempt details", expanded=False):
        rows = [
            {
                "attempt": item.attempt,
                "mode": item.mode,
                "solved": item.solved,
                "termination": item.termination_reason,
                "moves": " ".join(item.moves),
                "inference_ms": item.inference_seconds * 1000,
                "solver_ms": item.solver_seconds * 1000,
            }
            for item in result.attempts
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_frame(container, frame) -> None:
    with container.container():
        st.markdown(cube_net_html(frame.state), unsafe_allow_html=True)
        if frame.decision is None:
            st.caption("Initial scrambled state")
        else:
            probabilities = " · ".join(
                f"{move}: {probability:.1%}"
                for move, probability in frame.decision.top_actions()
            )
            st.caption(
                f"Step {frame.step}: {frame.move} · top policy actions: {probabilities}"
            )


def _render_benchmark(policy, checkpoint: CheckpointInfo) -> None:
    st.subheader("Seeded per-length benchmark")
    st.caption(
        "The inverse-scramble oracle supplies a correctness and reference-length "
        "baseline. Animation is disabled during benchmarking."
    )
    first, second, third = st.columns(3)
    with first:
        selected_range = st.slider(
            "Lengths",
            1,
            10,
            (1, 10),
            key="benchmark_lengths",
        )
        trials = st.number_input(
            "Trials per length",
            min_value=1,
            max_value=1_000,
            value=20,
        )
    with second:
        attempts = st.slider(
            "Attempt limit",
            1,
            10,
            3,
            key="benchmark_attempts",
        )
        timeout = st.number_input(
            "Time limit per scramble (seconds)",
            min_value=0.1,
            max_value=60.0,
            value=5.0,
            step=0.5,
            key="benchmark_timeout",
        )
    with third:
        seed = int(
            st.number_input(
                "Base seed",
                min_value=0,
                value=42,
                key="benchmark_seed",
            )
        )
        temperature = st.slider(
            "Retry temperature",
            0.1,
            3.0,
            1.0,
            0.1,
            key="benchmark_temperature",
        )

    if selected_range[1] > checkpoint.validated_depth:
        st.warning(
            "The selected benchmark includes exploratory lengths beyond this "
            "checkpoint's validated curriculum depth."
        )
    if st.button("Run benchmark", type="primary"):
        progress_bar = st.progress(0.0, text="Preparing benchmark...")

        def update_progress(completed: int, total: int) -> None:
            progress_bar.progress(
                completed / total,
                text=f"Completed {completed} of {total} trials",
            )

        try:
            with st.spinner("Benchmarking policy..."):
                benchmark = run_benchmark(
                    policy,
                    lengths=range(selected_range[0], selected_range[1] + 1),
                    trials_per_length=int(trials),
                    config=RunConfig(
                        max_moves=None,
                        max_attempts=attempts,
                        timeout_seconds=float(timeout),
                        temperature=temperature,
                        seed=seed,
                    ),
                    progress=update_progress,
                )
            st.session_state.benchmark_result = benchmark
            progress_bar.empty()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Benchmark failed: {exc}")

    benchmark: BenchmarkResult | None = st.session_state.benchmark_result
    if benchmark is None:
        return
    summary_frame = pd.DataFrame(benchmark.summary)
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)
    st.line_chart(
        summary_frame.set_index("scramble_length")[
            ["greedy_solve_rate", "retry_solve_rate"]
        ]
    )
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download summary CSV",
        benchmark_to_csv(benchmark, summary=True),
        "ppo_cube_benchmark_summary.csv",
        "text/csv",
    )
    download_right.download_button(
        "Download detailed CSV",
        benchmark_to_csv(benchmark),
        "ppo_cube_benchmark_details.csv",
        "text/csv",
    )


def _render_history() -> None:
    st.subheader("Demo runs in this session")
    history = st.session_state.demo_history
    if not history:
        st.info("No demo runs have been completed in this session.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    st.download_button(
        "Download demo history CSV",
        records_to_csv(history),
        "ppo_cube_demo_history.csv",
        "text/csv",
    )
    if st.button("Clear demo history"):
        st.session_state.demo_history = []
        st.rerun()


if __name__ == "__main__":
    main()
