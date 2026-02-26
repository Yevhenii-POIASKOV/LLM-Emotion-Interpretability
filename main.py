"""
main.py
=======
LLM-Emotion-Interpretability — Pipeline Entrypoint

Orchestrates the full research pipeline:
  1. model_inspector  — extract MLP activations from GPT-2
  2. analyzer         — compute statistical metrics (delta, Cohen's d, eta2, snr)
  3. visualizer       — generate heatmaps and reports

Usage:
    python main.py                         # full pipeline
    python main.py --skip_inspect          # skip step 1 (use existing activations)
    python main.py --run_dir data/activations/run_20260225_2122  # use specific run
    python main.py --bootstrap             # enable bootstrap validation
    python main.py --top_n 50             # top-N neurons in visualizations
"""

from __future__ import annotations
import argparse
import glob
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC_DIR)

_LOG_DIR  = os.path.join(_REPO_ROOT, "outputs", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")

try:
    from colorama import init as _colorama_init, Fore, Style
    _colorama_init(autoreset=True)
    def _c(text: str, color: str) -> str:
        return color + text + Style.RESET_ALL
except ImportError:
    def _c(text: str, color: str) -> str:
        return text
    class Fore:  # type: ignore[no-redef]
        GREEN = RED = YELLOW = CYAN = WHITE = MAGENTA = ""

_BANNER = r"""
+----------------------------------------------------------+
|                                                          |
|    ██╗     ██╗     ███╗   ███╗       ███████╗██╗        |
|    ██║     ██║     ████╗ ████║       ██╔════╝██║        |
|    ██║     ██║     ██╔████╔██║ ████╗ █████╗  ██║        |
|    ██║     ██║     ██║╚██╔╝██║ ╚═══╝ ██╔══╝  ██║        |
|    ███████╗███████╗██║ ╚═╝ ██║       ███████╗██║        |
|    ╚══════╝╚══════╝╚═╝     ╚═╝       ╚══════╝╚═╝        |
|                                                          |
|    GPT-2 MLP Neuron Emotion Interpretability             |
|    Pipeline v1.0  |  pos/neg sentiment analysis          |
|                                                          |
+----------------------------------------------------------+
"""

def _run_step(
    step_num: int,
    total_steps: int,
    label: str,
    func,
    *args,
    **kwargs,
):

    header = f"[{step_num}/{total_steps}]  {label}"
    log.info("-" * 60)
    log.info(header)
    log.info("-" * 60)

    bar_fmt = (
        "  {l_bar}{bar}  {n_fmt}/{total_fmt}  elapsed: {elapsed}  eta: {remaining}"
    )
    t_start = time.perf_counter()

    with tqdm(
        total=100,
        desc=_c(f"  {label}", Fore.CYAN),
        bar_format=bar_fmt,
        colour="cyan",
        leave=True,
    ) as pbar:
        pbar.update(10)
        try:
            result = func(*args, **kwargs)
            pbar.update(90)
        except Exception as exc:
            pbar.colour = "red"
            pbar.update(90)
            raise exc

    elapsed = time.perf_counter() - t_start
    log.info("Step completed in %.1fs", elapsed)
    return result, elapsed

def step_inspect() -> str:
    """Run model_inspector and return path to the newly created run directory."""
    from model_inspector import run_model_inspection
    run_model_inspection()

    runs = sorted(glob.glob(os.path.join(_REPO_ROOT, "data", "activations", "run_*")))
    if not runs:
        raise RuntimeError("model_inspector finished but no run_* folder was found.")
    return runs[-1]


def step_analyze(run_dir: str, bootstrap: bool, bootstrap_candidates: int) -> None:
    """Run analyzer on the given run directory."""
    from analyzer import main as _analyzer_main

    argv = ["analyzer.py", "--run_dir", run_dir]
    if bootstrap:
        argv += [
            "--bootstrap_enable",
            "--bootstrap_candidates", str(bootstrap_candidates),
            "--bootstrap_iters", "1000",
        ]
    sys.argv = argv
    _analyzer_main()


def step_visualize(run_dir: str, top_n: int, no_html: bool) -> str:
    """Run visualizer and return output directory path."""
    from visualizer import Visualizer

    viz = Visualizer(
        run_dir=run_dir,
        top_n=top_n,
        generate_html=not no_html,
    )
    return viz.run_all()

def _print_summary(
    run_dir: str,
    timings: dict,
    bootstrap: bool,
    out_heatmap_dir: str,
) -> None:
    run_name = os.path.basename(os.path.normpath(run_dir))
    total    = sum(timings.values())

    print()
    print(_c("=" * 60, Fore.GREEN))
    print(_c("  PIPELINE COMPLETED SUCCESSFULLY", Fore.GREEN))
    print(_c("=" * 60, Fore.GREEN))
    print()
    print(f"  Run name    : {_c(run_name, Fore.CYAN)}")
    print(f"  Log file    : {_c(_LOG_FILE, Fore.WHITE)}")
    print()
    print(_c("  Timings:", Fore.YELLOW))
    for step, elapsed in timings.items():
        print(f"    {step:<30} {elapsed:>6.1f}s")
    print(f"    {'TOTAL':<30} {total:>6.1f}s")
    print()
    print(_c("  Outputs:", Fore.YELLOW))
    print(f"    Heatmaps / PNG  : {out_heatmap_dir}")
    reports_dir = os.path.join(_REPO_ROOT, "outputs", "reports")
    print(f"    Reports / JSON  : {reports_dir}")
    if bootstrap:
        boot_path = os.path.join(reports_dir, f"bootstrap_{run_name}.json")
        exists = os.path.isfile(boot_path)
        status = _c("found", Fore.GREEN) if exists else _c("not found", Fore.RED)
        print(f"    Bootstrap report: {boot_path}  [{status}]")
    print()
    print(_c("=" * 60, Fore.GREEN))
    print()

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="LLM-Emotion-Interpretability — full research pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --skip_inspect --run_dir data/activations/run_20260225_2122
  python main.py --bootstrap --bootstrap_candidates 200
  python main.py --top_n 50 --no_html
        """,
    )

    p.add_argument(
        "--skip_inspect",
        action="store_true",
        help="Skip model_inspector step (reuse existing activations).",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Explicit path to an existing run directory (implies --skip_inspect).",
    )
    p.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable bootstrap CI validation in analyzer step.",
    )
    p.add_argument(
        "--bootstrap_candidates",
        type=int,
        default=100,
        help="Number of top neurons to bootstrap-check (default: 100).",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Top-N neurons shown in bar charts (default: 30).",
    )
    p.add_argument(
        "--no_html",
        action="store_true",
        help="Skip interactive Plotly HTML report generation.",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=["inspect", "analyze", "visualize"],
        default=["inspect", "analyze", "visualize"],
        help="Which steps to run (default: all three).",
    )
    return p

def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Resolve step list
    steps = set(args.steps)
    if args.run_dir or args.skip_inspect:
        steps.discard("inspect")

    total_steps = len(steps)
    step_num    = 0
    timings     = {}
    run_dir: Optional[str] = args.run_dir
    out_heatmap_dir: str   = ""

    print(_c(_BANNER, Fore.MAGENTA))
    log.info("Pipeline started  |  steps=%s  |  log=%s", sorted(steps), _LOG_FILE)

    if "inspect" in steps:
        step_num += 1
        try:
            run_dir, elapsed = _run_step(
                step_num, total_steps,
                "model_inspector  --  extracting MLP activations from GPT-2",
                step_inspect,
            )
            timings["model_inspector"] = elapsed
            log.info("Activations saved to: %s", run_dir)
        except Exception:
            log.error("model_inspector FAILED:\n%s", traceback.format_exc())
            return 1
    else:
        if run_dir is None:
            runs = sorted(glob.glob(
                os.path.join(_REPO_ROOT, "data", "activations", "run_*")
            ))
            if not runs:
                log.error(
                    "No run_* directories found in data/activations/. "
                    "Run without --skip_inspect first."
                )
                return 1
            run_dir = runs[-1]
            log.info("Using latest run: %s", run_dir)
        else:
            run_dir = os.path.abspath(run_dir)
            log.info("Using specified run: %s", run_dir)

    if "analyze" in steps:
        step_num += 1
        try:
            _, elapsed = _run_step(
                step_num, total_steps,
                "analyzer  --  delta / Cohen's d / eta2 / snr per neuron",
                step_analyze,
                run_dir,
                args.bootstrap,
                args.bootstrap_candidates,
            )
            timings["analyzer"] = elapsed
        except Exception:
            log.error("analyzer FAILED:\n%s", traceback.format_exc())
            return 1

    if "visualize" in steps:
        step_num += 1
        try:
            out_heatmap_dir, elapsed = _run_step(
                step_num, total_steps,
                "visualizer  --  heatmaps / bar charts / interactive HTML",
                step_visualize,
                run_dir,
                args.top_n,
                args.no_html,
            )
            timings["visualizer"] = elapsed
        except Exception:
            log.error("visualizer FAILED:\n%s", traceback.format_exc())
            return 1

    _print_summary(
        run_dir=run_dir,
        timings=timings,
        bootstrap=args.bootstrap,
        out_heatmap_dir=out_heatmap_dir,
    )
    log.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())