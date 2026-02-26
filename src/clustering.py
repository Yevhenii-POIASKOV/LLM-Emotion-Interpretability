"""Binary clustering of GPT-2 neurons into emotional vs neutral groups.

This script is the final stage of the LLM Emotion Interpretability project.
It reads analyzer outputs, builds a feature matrix across all neurons, runs
binary K-Means multiple times, identifies the emotional cluster, and saves
neurons that are *consistently* emotional (stability = 1.0).

Outputs (under outputs/reports/):
  - emotional_clusters_<run_name>.json   : full stable emotional neurons
  - emotional_clusters_<run_name>.summary.json : compact machine-readable summary
  - emotional_clusters_<run_name>.summary.md   : human-readable summary
  - emotional_clusters_<run_name>.report.md    : detailed report (counts, top list)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for clustering and reporting."""

    run_name: str
    features: Tuple[str, ...]
    iterations: int = 10
    random_seed: int = 42
    n_init: int = 20
    report_top_k: int = 50
    report_make_plots: bool = False


@dataclass(frozen=True)
class LoadedMetrics:
    """Container for stacked analyzer metrics."""

    layer_ids: np.ndarray  # shape (N,)
    neuron_ids: np.ndarray  # shape (N,)
    feature_matrix: np.ndarray  # shape (N, n_features)
    effect_size: np.ndarray  # shape (N,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LAYER_FILE_RE = re.compile(r"analyzer_(?P<run>.+)_layer_(?P<layer>\d+)\.npz$")


def repo_root_from_here() -> Path:
    """Return repository root inferred from this file location."""
    return Path(__file__).resolve().parent.parent


def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class NeuronClusterizer:
    """Cluster GPT-2 neurons into emotional vs neutral using analyzer metrics."""

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        self.repo_root = repo_root_from_here()
        self.reports_dir = self.repo_root / "outputs" / "reports"

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self) -> str:
        """Execute full clustering pipeline and return path to main JSON output."""
        loaded = self._load_metrics()
        x_scaled = self._standardize_features(loaded.feature_matrix)

        emotional_votes, silhouette_values = self._run_repeated_kmeans(
            x_scaled=x_scaled,
            effect_size=loaded.effect_size,
        )

        stability = emotional_votes.astype(np.float64) / float(self.config.iterations)
        records, stable_mask = self._build_stable_records(
            layer_ids=loaded.layer_ids,
            neuron_ids=loaded.neuron_ids,
            effect_size=loaded.effect_size,
            stability=stability,
        )

        safe_makedirs(self.reports_dir)
        json_path = self._write_emotional_clusters_json(records)
        summary_json, summary_md = self._write_summary(
            records=records,
            layer_ids=loaded.layer_ids,
            stable_mask=stable_mask,
            silhouette_values=silhouette_values,
            total_neurons=int(loaded.feature_matrix.shape[0]),
        )
        report_path = self._write_detailed_report(
            layer_ids=loaded.layer_ids,
            neuron_ids=loaded.neuron_ids,
            effect_size=loaded.effect_size,
            stability=stability,
            silhouette_values=silhouette_values,
            top_k=self.config.report_top_k,
            make_plots=self.config.report_make_plots,
        )

        LOGGER.info(
            "Done. total_neurons=%d stable_emotional=%d silhouette_mean=%.4f silhouette_std=%.4f",
            loaded.feature_matrix.shape[0],
            int(np.sum(stable_mask)),
            float(np.mean(silhouette_values)),
            float(np.std(silhouette_values)),
        )
        LOGGER.info("Saved: %s", json_path)
        LOGGER.info("Summary: %s | %s", summary_json, summary_md)
        LOGGER.info("Report : %s", report_path)
        return json_path

    # -----------------------------
    # Loading
    # -----------------------------
    def _find_layer_files(self) -> List[Tuple[int, Path]]:
        pattern = str(self.reports_dir / f"analyzer_{self.config.run_name}_layer_*.npz")
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(
                f"No analyzer files for run_name='{self.config.run_name}' in {self.reports_dir}."
            )

        parsed: List[Tuple[int, Path]] = []
        for path_str in paths:
            name = os.path.basename(path_str)
            match = LAYER_FILE_RE.match(name)
            if match is None:
                continue
            parsed.append((int(match.group("layer")), Path(path_str)))

        if not parsed:
            raise FileNotFoundError(
                f"Matched files do not contain valid layer suffixes for run_name='{self.config.run_name}'."
            )

        parsed.sort(key=lambda pair: pair[0])
        LOGGER.info("Found %d layer files for run '%s'.", len(parsed), self.config.run_name)
        return parsed

    def _load_metrics(self) -> LoadedMetrics:
        layer_files = self._find_layer_files()

        layer_ids_chunks: List[np.ndarray] = []
        neuron_ids_chunks: List[np.ndarray] = []
        feature_chunks: List[np.ndarray] = []
        effect_size_chunks: List[np.ndarray] = []

        expected_neurons: int | None = None
        for layer, path in layer_files:
            with np.load(path) as data:
                missing = [feat for feat in self.config.features if feat not in data]
                if missing:
                    raise KeyError(f"Missing features in {path}: {missing}")
                if "effect_size" not in data:
                    raise KeyError(f"Missing required effect_size in {path}")

                columns = [np.asarray(data[feat], dtype=np.float64).reshape(-1) for feat in self.config.features]
                current_effect = np.asarray(data["effect_size"], dtype=np.float64).reshape(-1)

            n_neurons = columns[0].shape[0]
            if expected_neurons is None:
                expected_neurons = n_neurons
            elif n_neurons != expected_neurons:
                raise ValueError(
                    f"Inconsistent neuron count across layers: expected {expected_neurons}, got {n_neurons} in layer {layer}."
                )

            for feat_arr, feat_name in zip(columns, self.config.features):
                if feat_arr.shape[0] != n_neurons:
                    raise ValueError(f"Feature '{feat_name}' in layer {layer} has incompatible shape.")
                if not np.all(np.isfinite(feat_arr)):
                    raise ValueError(f"Feature '{feat_name}' in layer {layer} contains non-finite values.")

            if current_effect.shape[0] != n_neurons or not np.all(np.isfinite(current_effect)):
                raise ValueError(f"effect_size in layer {layer} has invalid shape or non-finite values.")

            layer_ids_chunks.append(np.full(n_neurons, layer, dtype=np.int32))
            neuron_ids_chunks.append(np.arange(n_neurons, dtype=np.int32))
            feature_chunks.append(np.column_stack(columns))
            effect_size_chunks.append(current_effect)

        layer_ids = np.concatenate(layer_ids_chunks, axis=0)
        neuron_ids = np.concatenate(neuron_ids_chunks, axis=0)
        feature_matrix = np.vstack(feature_chunks)
        effect_size = np.concatenate(effect_size_chunks, axis=0)

        LOGGER.info("Loaded feature matrix %s using features=%s", feature_matrix.shape, ",".join(self.config.features))
        return LoadedMetrics(
            layer_ids=layer_ids,
            neuron_ids=neuron_ids,
            feature_matrix=feature_matrix,
            effect_size=effect_size,
        )

    # -----------------------------
    # Clustering
    # -----------------------------
    @staticmethod
    def _standardize_features(x: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        if not np.all(np.isfinite(x_scaled)):
            raise ValueError("Standardized feature matrix contains non-finite values.")
        return x_scaled

    def _run_repeated_kmeans(
        self,
        *,
        x_scaled: np.ndarray,
        effect_size: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        emotional_votes = np.zeros(x_scaled.shape[0], dtype=np.int32)
        silhouette_values: List[float] = []

        for run_idx in range(self.config.iterations):
            random_state = self.config.random_seed + run_idx
            model = KMeans(
                n_clusters=2,
                random_state=random_state,
                n_init=self.config.n_init,
            )
            labels = model.fit_predict(x_scaled)

            emotional_cluster_id = self._identify_emotional_cluster(labels=labels, effect_size=effect_size)
            emotional_mask = labels == emotional_cluster_id
            emotional_votes += emotional_mask.astype(np.int32)

            sil = silhouette_score(x_scaled, labels)
            silhouette_values.append(float(sil))
            LOGGER.info(
                "Run %d/%d | random_state=%d | silhouette=%.4f | emotional_count=%d",
                run_idx + 1,
                self.config.iterations,
                random_state,
                float(sil),
                int(np.sum(emotional_mask)),
            )

        return emotional_votes, silhouette_values

    @staticmethod
    def _identify_emotional_cluster(labels: np.ndarray, effect_size: np.ndarray) -> int:
        cluster_scores: Dict[int, float] = {}
        for cluster_id in (0, 1):
            mask = labels == cluster_id
            if not np.any(mask):
                cluster_scores[cluster_id] = -np.inf
                continue
            cluster_scores[cluster_id] = float(np.mean(np.abs(effect_size[mask])))
        return int(max(cluster_scores, key=cluster_scores.get))

    # -----------------------------
    # Outputs
    # -----------------------------
    @staticmethod
    def _build_stable_records(
        *,
        layer_ids: np.ndarray,
        neuron_ids: np.ndarray,
        effect_size: np.ndarray,
        stability: np.ndarray,
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        stable_mask = np.isclose(stability, 1.0)
        idx = np.where(stable_mask)[0]
        if idx.size == 0:
            return [], stable_mask

        order = np.argsort(np.abs(effect_size[idx]))[::-1]
        ordered_idx = idx[order]
        records = [
            {
                "layer": int(layer_ids[i]),
                "neuron": int(neuron_ids[i]),
                "effect_size": float(effect_size[i]),
                "stability": float(stability[i]),
            }
            for i in ordered_idx
        ]
        return records, stable_mask

    def _write_emotional_clusters_json(self, records: List[Dict[str, float]]) -> str:
        output_path = self.reports_dir / f"emotional_clusters_{self.config.run_name}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return str(output_path)

    def _write_summary(
        self,
        *,
        records: List[Dict[str, float]],
        layer_ids: np.ndarray,
        stable_mask: np.ndarray,
        silhouette_values: Sequence[float],
        total_neurons: int,
    ) -> Tuple[str, str]:
        stable_count = int(np.sum(stable_mask))
        per_layer_counts: Dict[int, int] = {}
        if stable_mask.any():
            unique_layers, counts = np.unique(layer_ids[stable_mask], return_counts=True)
            per_layer_counts = {int(l): int(c) for l, c in zip(unique_layers, counts)}

        top_records = records[: min(self.config.report_top_k, len(records))]

        sil = np.asarray(list(silhouette_values), dtype=np.float64)
        sil_mean = float(np.mean(sil)) if sil.size else float("nan")
        sil_std = float(np.std(sil)) if sil.size else float("nan")

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": self.config.run_name,
            "features": list(self.config.features),
            "iterations": int(self.config.iterations),
            "random_seed": int(self.config.random_seed),
            "n_init": int(self.config.n_init),
            "total_neurons": int(total_neurons),
            "stable_emotional_neurons": int(stable_count),
            "stable_rate": float(stable_count / float(total_neurons)) if total_neurons else 0.0,
            "silhouette": {
                "mean": sil_mean,
                "std": sil_std,
                "values": [float(v) for v in sil.tolist()],
            },
            "stable_counts_per_layer": {str(k): v for k, v in sorted(per_layer_counts.items())},
            "top_stable_neurons": top_records,
            "full_output": f"emotional_clusters_{self.config.run_name}.json",
        }

        summary_json = self.reports_dir / f"emotional_clusters_{self.config.run_name}.summary.json"
        summary_md = self.reports_dir / f"emotional_clusters_{self.config.run_name}.summary.md"

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        lines: List[str] = []
        lines.append(f"# Emotional clusters summary — {self.config.run_name}")
        lines.append("")
        lines.append("## Коротко")
        lines.append(f"- Total neurons: {total_neurons}")
        lines.append(f"- Stable emotional neurons: {stable_count} ({payload['stable_rate']:.2%})")
        lines.append(f"- Features: {', '.join(self.config.features)}")
        lines.append(f"- K-Means runs: {self.config.iterations} (n_init={self.config.n_init})")
        lines.append(f"- Silhouette mean/std: {sil_mean:.4f} / {sil_std:.4f}")
        lines.append("")
        lines.append("## Stable counts per layer")
        if per_layer_counts:
            for layer in sorted(per_layer_counts):
                lines.append(f"- Layer {layer}: {per_layer_counts[layer]}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append(f"## Top-{len(top_records)} stable neurons (by |effect_size|)")
        if top_records:
            lines.append("| rank | layer | neuron | effect_size | stability |")
            lines.append("|---:|---:|---:|---:|---:|")
            for rank, rec in enumerate(top_records, start=1):
                lines.append(
                    f"| {rank} | {rec['layer']} | {rec['neuron']} | {rec['effect_size']:.6g} | {rec['stability']:.3g} |"
                )
        else:
            lines.append("No stable emotional neurons found (stability=1.0).")
        lines.append("")
        lines.append("## Full output")
        lines.append(f"- emotional_clusters_{self.config.run_name}.json (може бути дуже великим)")
        lines.append(f"- emotional_clusters_{self.config.run_name}.summary.json (машиночитний summary)")

        summary_md.write_text("\n".join(lines), encoding="utf-8")
        return str(summary_json), str(summary_md)

    def _write_detailed_report(
        self,
        *,
        layer_ids: np.ndarray,
        neuron_ids: np.ndarray,
        effect_size: np.ndarray,
        stability: np.ndarray,
        silhouette_values: Sequence[float],
        top_k: int,
        make_plots: bool,
    ) -> str:
        report_path = self.reports_dir / f"emotional_clusters_{self.config.run_name}.report.md"
        stable_mask = np.isclose(stability, 1.0)
        stable_idx = np.where(stable_mask)[0]
        n_stable = stable_idx.size

        unique_layers, counts = np.unique(layer_ids, return_counts=True)
        per_layer_total = {int(l): int(c) for l, c in zip(unique_layers, counts)}
        per_layer_stable: Dict[int, int] = {}
        if stable_mask.any():
            s_l, s_c = np.unique(layer_ids[stable_mask], return_counts=True)
            per_layer_stable = {int(l): int(c) for l, c in zip(s_l, s_c)}

        abs_effect_stable = np.abs(effect_size[stable_idx]) if n_stable else np.array([])
        if n_stable:
            order = np.argsort(abs_effect_stable)[::-1]
            top_indices = stable_idx[order[: min(top_k, n_stable)]]
        else:
            top_indices = np.array([], dtype=int)

        sil = np.asarray(list(silhouette_values), dtype=np.float64)
        sil_mean = float(np.mean(sil)) if sil.size else float("nan")
        sil_std = float(np.std(sil)) if sil.size else float("nan")
        sil_min = float(np.min(sil)) if sil.size else float("nan")
        sil_max = float(np.max(sil)) if sil.size else float("nan")

        def fmt_per_layer() -> List[str]:
            lines: List[str] = []
            lines.append("| layer | stable | total | stable share |")
            lines.append("|---:|---:|---:|---:|")
            for l in sorted(per_layer_total):
                stable_cnt = per_layer_stable.get(l, 0)
                total_cnt = per_layer_total.get(l, 0)
                share = (stable_cnt / total_cnt) if total_cnt else 0.0
                lines.append(f"| {l} | {stable_cnt} | {total_cnt} | {share:.2%} |")
            return lines

        def fmt_top() -> List[str]:
            if top_indices.size == 0:
                return ["No stable emotional neurons under current criterion."]
            lines: List[str] = []
            lines.append("| rank | layer | neuron | effect_size | |effect_size| | stability |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for rank, idx in enumerate(top_indices, start=1):
                l = int(layer_ids[idx])
                n = int(neuron_ids[idx])
                es = float(effect_size[idx])
                st = float(stability[idx])
                lines.append(f"| {rank} | {l} | {n} | {es:.6g} | {abs(es):.6g} | {st:.3g} |")
            return lines

        lines: List[str] = []
        lines.append(f"# Clustering report — {self.config.run_name}")
        lines.append("")
        lines.append("## Inputs")
        lines.append(f"- run_name: {self.config.run_name}")
        lines.append(f"- features: {', '.join(self.config.features)}")
        lines.append(
            f"- iterations: {self.config.iterations} (random_seed base={self.config.random_seed}, n_init={self.config.n_init})"
        )
        lines.append(f"- total neurons: {effect_size.shape[0]}")
        lines.append("")
        lines.append("## Result size")
        lines.append(f"- stable emotional: {n_stable} ({(n_stable / effect_size.shape[0]):.2%})")
        lines.append("")
        lines.append("## Clustering quality (silhouette)")
        lines.append(f"- mean={sil_mean:.4f}, std={sil_std:.4f}, min={sil_min:.4f}, max={sil_max:.4f}")
        lines.append("")
        lines.append("## Stable counts per layer")
        lines.extend(fmt_per_layer())
        lines.append("")
        lines.append(f"## Top {min(top_k, n_stable)} stable emotional neurons (by |effect_size|)")
        lines.extend(fmt_top())
        lines.append("")
        lines.append("## Files")
        lines.append(f"- emotional_clusters_{self.config.run_name}.json (full)")
        lines.append(f"- emotional_clusters_{self.config.run_name}.summary.json (short)")
        lines.append(f"- emotional_clusters_{self.config.run_name}.summary.md (short)")

        if make_plots and n_stable > 0:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.bar(sorted(per_layer_total), [per_layer_stable.get(l, 0) for l in sorted(per_layer_total)])
                ax.set_xlabel("Layer")
                ax.set_ylabel("Stable emotional count")
                ax.set_title("Stable emotional neurons per layer")
                bar_path = self.reports_dir / f"emotional_clusters_{self.config.run_name}.per_layer.png"
                fig.tight_layout()
                fig.savefig(bar_path, dpi=160)
                plt.close(fig)
                lines.append(f"- Plot: {bar_path.name}")

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(abs_effect_stable, bins=50)
                ax.set_xlabel("|effect_size|")
                ax.set_ylabel("Count")
                ax.set_title("|effect_size| for stable emotional neurons")
                hist_path = self.reports_dir / f"emotional_clusters_{self.config.run_name}.abs_effect_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=160)
                plt.close(fig)
                lines.append(f"- Plot: {hist_path.name}")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Plot generation skipped: %s", exc)
                lines.append("- Plot generation skipped (matplotlib missing or error).")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster GPT-2 neurons into emotional/neutral groups using analyzer metrics.",
    )
    parser.add_argument("--run_name", required=True, help="Run identifier, e.g. run_20260225_0035")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["effect_size", "eta2"],
        help="Feature list for clustering (must exist in analyzer .npz)",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Number of K-Means runs for stability.")
    parser.add_argument("--random_seed", type=int, default=42, help="Base random seed; each run adds +i.")
    parser.add_argument("--n_init", type=int, default=20, help="K-Means n_init per run.")
    parser.add_argument(
        "--report_top_k",
        type=int,
        default=50,
        help="How many stable neurons to show in reports (by |effect_size|).",
    )
    parser.add_argument(
        "--report_make_plots",
        action="store_true",
        help="Generate PNG plots in reports (requires matplotlib).",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.n_init <= 0:
        raise ValueError("--n_init must be > 0")
    if not args.features:
        raise ValueError("--features must contain at least one feature")


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    validate_args(args)

    config = ClusterConfig(
        run_name=str(args.run_name),
        features=tuple(str(f) for f in args.features),
        iterations=int(args.iterations),
        random_seed=int(args.random_seed),
        n_init=int(args.n_init),
        report_top_k=int(args.report_top_k),
        report_make_plots=bool(args.report_make_plots),
    )

    clusterizer = NeuronClusterizer(config)
    clusterizer.run()


if __name__ == "__main__":
    main()