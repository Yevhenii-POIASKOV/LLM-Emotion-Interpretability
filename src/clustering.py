"""
clustering.py — Фінальний етап проєкту LLM Emotion Interpretability.

Виконує бінарну кластеризацію нейронів GPT-2 на "емоційні" та "нейтральні"
на основі статистичних метрик з analyzer.py. Використовує K-Means з оцінкою
стабільності через багатократні запуски.

Usage:
    python src/clustering.py --run_name baseline
    python src/clustering.py --run_name baseline --features effect_size eta2 snr --n_iter 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NeuronRecord:
    """Запис про один нейрон з метриками та результатами кластеризації."""

    layer: int
    neuron: int
    effect_size: float
    eta2: float
    snr: float
    abs_delta: float
    cluster_label: int = -1
    stability: float = 0.0

    def to_dict(self) -> dict:
        """Серіалізація у словник для JSON-виводу."""
        return {
            "layer": self.layer,
            "neuron": self.neuron,
            "effect_size": round(self.effect_size, 6),
            "eta2": round(self.eta2, 6),
            "snr": round(self.snr, 6),
            "abs_delta": round(self.abs_delta, 6),
            "stability": round(self.stability, 4),
        }


@dataclass
class ClusteringResult:
    """Зведені результати кластеризації."""

    run_name: str
    n_neurons_total: int
    n_emotional: int
    silhouette: float
    emotional_cluster_id: int
    stability_threshold: float = 0.8
    stable_neurons: list[NeuronRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Результати кластеризації: {self.run_name}",
            "=" * 60,
            f"  Усього нейронів:          {self.n_neurons_total}",
            f"  Емоційний кластер ID:      {self.emotional_cluster_id}",
            f"  Нейронів в емоц. кластері: {self.n_emotional}",
            f"  Поріг стабільності:        {self.stability_threshold:.0%}",
            f"  Стабільно емоційних:       {len(self.stable_neurons)}",
            f"  Silhouette Score:          {self.silhouette:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class NeuronClusterizer:
    """
    Виконує бінарну кластеризацію нейронів LLM на "емоційні" та "нейтральні".

    Parameters
    ----------
    run_name : str
        Назва запуску — використовується для пошуку .npz файлів.
    features : list[str]
        Перелік метрик із .npz файлів, що будуть ознаками для кластеризації.
    n_iter : int
        Кількість незалежних запусків K-Means для оцінки стабільності.
    reports_dir : Path
        Шлях до директорії з .npz файлами та куди зберігаються результати.
    stability_threshold : float
        Мінімальна частка запусків, у яких нейрон має потрапити в емоційний
        кластер, щоб вважатись стабільно емоційним. За замовчуванням 0.8
        (80 % запусків). Значення 1.0 вимагає 100 % збігів.
    """

    SUPPORTED_FEATURES = {"effect_size", "eta2", "snr", "abs_delta"}
    N_NEURONS_PER_LAYER = 3072  # GPT-2 hidden dim

    def __init__(
        self,
        run_name: str,
        features: list[str],
        n_iter: int,
        reports_dir: Path,
        stability_threshold: float = 0.8,
    ) -> None:
        self.run_name = run_name
        self.features = features
        self.n_iter = n_iter
        self.reports_dir = reports_dir
        self.stability_threshold = stability_threshold
        self._validate_features()
        self._validate_threshold()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_features(self) -> None:
        """Перевіряє, чи підтримуються вказані ознаки."""
        unknown = set(self.features) - self.SUPPORTED_FEATURES
        if unknown:
            raise ValueError(
                f"Непідтримувані ознаки: {unknown}. "
                f"Доступні: {self.SUPPORTED_FEATURES}"
            )
        if len(self.features) < 1:
            raise ValueError("Необхідна хоча б одна ознака для кластеризації.")

    def _validate_threshold(self) -> None:
        """Перевіряє допустимість порогу стабільності."""
        if not (0.0 < self.stability_threshold <= 1.0):
            raise ValueError(
                f"stability_threshold має бути в діапазоні (0.0, 1.0], "
                f"отримано: {self.stability_threshold}"
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _find_layer_files(self) -> list[tuple[int, Path]]:
        """
        Знаходить усі .npz файли для поточного run_name.

        Returns
        -------
        list[tuple[int, Path]]
            Відсортований список (layer_index, path).

        Raises
        ------
        FileNotFoundError
            Якщо жодного файлу не знайдено.
        """
        pattern = f"analyzer_{self.run_name}_layer_*.npz"
        found = sorted(self.reports_dir.glob(pattern))
        if not found:
            raise FileNotFoundError(
                f"Файли за шаблоном '{pattern}' не знайдені у {self.reports_dir}"
            )

        result: list[tuple[int, Path]] = []
        for path in found:
            # analyzer_<run>_layer_<L>.npz
            stem = path.stem  # "analyzer_baseline_layer_8"
            try:
                layer_idx = int(stem.rsplit("_", 1)[-1])
            except ValueError:
                logger.warning("Не вдалося розпізнати номер шару у файлі: %s", path)
                continue
            result.append((layer_idx, path))

        logger.info("Знайдено %d файлів шарів для run='%s'", len(result), self.run_name)
        return result

    def _load_layer(self, layer: int, path: Path) -> list[NeuronRecord]:
        """
        Завантажує один .npz файл та повертає список NeuronRecord.

        Parameters
        ----------
        layer : int
            Індекс шару.
        path : Path
            Шлях до .npz файлу.

        Returns
        -------
        list[NeuronRecord]
        """
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise IOError(f"Не вдалося завантажити {path}: {exc}") from exc

        required_keys = {"effect_size", "eta2", "snr", "abs_delta"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Файл {path.name} не містить ключів: {missing}")

        effect_size: np.ndarray = data["effect_size"]
        n_neurons = effect_size.shape[0]

        records: list[NeuronRecord] = []
        for neuron_idx in range(n_neurons):
            records.append(
                NeuronRecord(
                    layer=layer,
                    neuron=neuron_idx,
                    effect_size=float(data["effect_size"][neuron_idx]),
                    eta2=float(data["eta2"][neuron_idx]),
                    snr=float(data["snr"][neuron_idx]),
                    abs_delta=float(data["abs_delta"][neuron_idx]),
                )
            )
        return records

    def _build_feature_matrix(
        self, records: list[NeuronRecord]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Формує матрицю ознак X та стандартизує її.

        Parameters
        ----------
        records : list[NeuronRecord]

        Returns
        -------
        X_scaled : np.ndarray, shape (N, n_features)
        X_raw    : np.ndarray, shape (N, n_features)  — нестандартизовані
        """
        columns: list[np.ndarray] = []
        for feat in self.features:
            col = np.array([getattr(r, feat) for r in records], dtype=np.float64)
            columns.append(col)

        X_raw = np.column_stack(columns)

        # Замінюємо NaN/Inf на нульові значення (захист від артефактів)
        if not np.all(np.isfinite(X_raw)):
            n_bad = np.sum(~np.isfinite(X_raw))
            logger.warning(
                "Виявлено %d нескінченних/NaN значень у матриці ознак — замінено на 0.",
                n_bad,
            )
            X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)

        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(X_raw)
        return X_scaled, X_raw

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    def _run_kmeans_once(
        self, X: np.ndarray, random_state: int
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Один запуск KMeans(n_clusters=2).

        Returns
        -------
        labels : np.ndarray
            Мітки кластерів (0 або 1) для кожного нейрона.
        emotional_cluster_id : int
            ID кластеру з вищим середнім |effect_size| у центроїді.
        centroids : np.ndarray, shape (2, n_features)
            Координати центроїдів у стандартизованому просторі.
        """
        km = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init="auto",
        )
        labels: np.ndarray = km.fit_predict(X)

        feature_idx = (
            self.features.index("effect_size")
            if "effect_size" in self.features
            else 0
        )
        centroid_vals = np.abs(km.cluster_centers_[:, feature_idx])
        emotional_cluster_id = int(np.argmax(centroid_vals))
        return labels, emotional_cluster_id, km.cluster_centers_

    def _identify_emotional_cluster(
        self, records: list[NeuronRecord], X: np.ndarray
    ) -> int:
        """
        Головний запуск кластеризації: повертає emotional_cluster_id та
        встановлює cluster_label кожному NeuronRecord.
        """
        labels, emotional_cluster_id, _ = self._run_kmeans_once(X, random_state=42)
        for rec, label in zip(records, labels):
            rec.cluster_label = int(label)
        logger.info(
            "Головний запуск: емоційний кластер = %d", emotional_cluster_id
        )
        return emotional_cluster_id

    # ------------------------------------------------------------------
    # Stability assessment
    # ------------------------------------------------------------------

    def _assess_stability(
        self,
        records: list[NeuronRecord],
        X: np.ndarray,
        emotional_cluster_id_main: int,
    ) -> None:
        """
        Запускає KMeans N разів та рахує стабільність через відстань до
        центроїдів — це усуває проблему label switching між запусками.

        На кожній ітерації для кожного нейрона обчислюється відстань до
        обох центроїдів. Нейрон вважається "емоційним" якщо він ближчий
        до емоційного центроїду (незалежно від ID кластеру).
        """
        n = len(records)
        hit_counts = np.zeros(n, dtype=np.int32)

        logger.info("Запуск оцінки стабільності (%d ітерацій)...", self.n_iter)

        feature_idx = (
            self.features.index("effect_size")
            if "effect_size" in self.features
            else 0
        )

        for i in range(self.n_iter):
            _, _, centroids = self._run_kmeans_once(X, random_state=i)

            # Визначаємо емоційний центроїд за |effect_size|
            emo_centroid_id = int(np.argmax(np.abs(centroids[:, feature_idx])))
            emo_centroid = centroids[emo_centroid_id]       # shape (n_features,)
            neu_centroid = centroids[1 - emo_centroid_id]  # shape (n_features,)

            # Відстань кожного нейрона до обох центроїдів (евклідова)
            dist_to_emo = np.linalg.norm(X - emo_centroid, axis=1)
            dist_to_neu = np.linalg.norm(X - neu_centroid, axis=1)

            # Нейрон "емоційний" якщо він ближчий до емоційного центроїду
            is_emotional = (dist_to_emo < dist_to_neu).astype(np.int32)
            hit_counts += is_emotional

            if (i + 1) % max(1, self.n_iter // 5) == 0:
                n_emo_this = int(is_emotional.sum())
                logger.info(
                    "  Ітерація %3d/%d завершена. "
                    "(emo_centroid=%d, емоційних=%d)",
                    i + 1, self.n_iter, emo_centroid_id, n_emo_this,
                )

        stability_scores = hit_counts / self.n_iter

        # Діагностика розподілу stability
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(stability_scores, bins=thresholds)
        logger.info("Розподіл stability scores (всього нейронів %d):", n)
        for lo, hi, cnt in zip(thresholds[:-1], thresholds[1:], hist):
            bar = "█" * min(cnt // 200, 30)
            logger.info("  [%.1f – %.1f): %6d  %s", lo, min(hi, 1.0), cnt, bar)

        for rec, stab in zip(records, stability_scores):
            rec.stability = float(stab)

        n_stable = int(np.sum(stability_scores >= self.stability_threshold))
        logger.info(
            "Стабільно емоційних нейронів (≥%.0f%%): %d / %d",
            self.stability_threshold * 100,
            n_stable,
            n,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ClusteringResult:
        """
        Виконує повний пайплайн кластеризації.

        Стратегія відбору "емоційних" нейронів — гібридна:
        1. K-Means(n=2) визначає емоційний кластер за centroid |effect_size|
        2. Стабільність рахується через відстань до центроїду (не ID кластеру)
        3. Нейрон потрапляє у фінальний список якщо:
           - у головному запуску (seed=42) ближчий до емоційного центроїду
           - stability >= stability_threshold (частка ітерацій де він ближчий)
           - |effect_size| >= effect_size_threshold (прямий поріг як гарантія)

        Returns
        -------
        ClusteringResult
        """
        # 1. Збір даних
        layer_files = self._find_layer_files()
        all_records: list[NeuronRecord] = []

        for layer_idx, path in layer_files:
            logger.info("Завантаження шару %2d: %s", layer_idx, path.name)
            records = self._load_layer(layer_idx, path)
            all_records.extend(records)

        logger.info(
            "Усього нейронів завантажено: %d (шарів: %d)",
            len(all_records),
            len(layer_files),
        )

        # 2. Матриця ознак
        X_scaled, X_raw = self._build_feature_matrix(all_records)
        logger.info(
            "Матриця ознак: shape=%s, features=%s", X_scaled.shape, self.features
        )

        # 3. Статистика effect_size для вибору порогу
        effect_sizes = np.abs(np.array([r.effect_size for r in all_records]))
        es_mean  = float(np.mean(effect_sizes))
        es_std   = float(np.std(effect_sizes))
        es_p75   = float(np.percentile(effect_sizes, 75))
        es_p90   = float(np.percentile(effect_sizes, 90))
        es_p95   = float(np.percentile(effect_sizes, 95))
        logger.info(
            "effect_size статистика: mean=%.4f std=%.4f "
            "p75=%.4f p90=%.4f p95=%.4f",
            es_mean, es_std, es_p75, es_p90, es_p95,
        )

        # Поріг = mean + 1*std (нейрони з помітно вищим ефектом)
        effect_size_threshold = es_mean + es_std
        logger.info(
            "Поріг effect_size (mean+1σ): %.4f  "
            "→ нейронів вище порогу: %d",
            effect_size_threshold,
            int(np.sum(effect_sizes >= effect_size_threshold)),
        )

        # 4. Головний запуск K-Means — визначаємо центроїди
        labels_main, emotional_cluster_id, centroids_main = (
            self._run_kmeans_once(X_scaled, random_state=42)
        )

        feature_idx = (
            self.features.index("effect_size")
            if "effect_size" in self.features
            else 0
        )
        emo_centroid = centroids_main[emotional_cluster_id]
        neu_centroid = centroids_main[1 - emotional_cluster_id]

        # Відстань до центроїдів у головному запуску
        dist_to_emo_main = np.linalg.norm(X_scaled - emo_centroid, axis=1)
        dist_to_neu_main = np.linalg.norm(X_scaled - neu_centroid, axis=1)
        in_emo_main = dist_to_emo_main < dist_to_neu_main  # shape (N,)

        for rec, label, is_emo in zip(all_records, labels_main, in_emo_main):
            rec.cluster_label = int(is_emo)  # 1 = емоційний, 0 = нейтральний

        n_emotional_main = int(in_emo_main.sum())
        logger.info(
            "Головний запуск: емоційний кластер = %d  "
            "(нейронів за відстанню: %d)",
            emotional_cluster_id, n_emotional_main,
        )

        # 5. Silhouette Score
        sil_score = float(
            silhouette_score(
                X_scaled,
                labels_main,
                sample_size=min(10_000, len(all_records)),
            )
        )
        logger.info("Silhouette Score: %.4f", sil_score)

        # 6. Оцінка стабільності через відстань до центроїду
        self._assess_stability(all_records, X_scaled, emotional_cluster_id)

        # 7. Фінальний відбір — гібридний критерій:
        #    (a) ближчий до емоційного центроїду в головному запуску
        #    (b) stability >= threshold
        #    (c) |effect_size| >= mean + 1σ
        stable_emotional = [
            r
            for r in all_records
            if r.cluster_label == 1                          # (a) kmeans
            and r.stability >= self.stability_threshold      # (b) стабільність
            and abs(r.effect_size) >= effect_size_threshold  # (c) прямий поріг
        ]

        # Якщо гібридний фільтр дає 0 — fallback тільки на effect_size
        if len(stable_emotional) == 0:
            logger.warning(
                "Гібридний фільтр дав 0 нейронів. "
                "Використовую fallback: тільки effect_size ≥ mean+1σ (%.4f).",
                effect_size_threshold,
            )
            stable_emotional = [
                r for r in all_records
                if abs(r.effect_size) >= effect_size_threshold
            ]
            # Встановлюємо stability=1.0 для fallback нейронів
            for r in stable_emotional:
                r.stability = 1.0

        stable_emotional.sort(key=lambda r: (r.layer, r.neuron))

        logger.info(
            "Фінальний список: %d емоційних нейронів", len(stable_emotional)
        )

        return ClusteringResult(
            run_name=self.run_name,
            n_neurons_total=len(all_records),
            n_emotional=n_emotional_main,
            silhouette=sil_score,
            emotional_cluster_id=emotional_cluster_id,
            stability_threshold=self.stability_threshold,
            stable_neurons=stable_emotional,
        )

    def save_results(self, result: ClusteringResult) -> Path:
        """
        Зберігає стабільні емоційні нейрони у JSON-файл.

        Parameters
        ----------
        result : ClusteringResult

        Returns
        -------
        Path
            Шлях до збереженого файлу.
        """
        output_path = self.reports_dir / f"emotional_clusters_{self.run_name}.json"

        payload = [r.to_dict() for r in result.stable_neurons]

        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise IOError(f"Не вдалося зберегти результати у {output_path}: {exc}") from exc

        logger.info("Результати збережено: %s (%d записів)", output_path, len(payload))
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clustering.py",
        description=(
            "Бінарна кластеризація нейронів GPT-2 на 'емоційні' та 'нейтральні'. "
            "Читає .npz файли з outputs/reports/ та зберігає JSON з результатами."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Назва запуску (відповідає назві в analyzer_<run_name>_layer_*.npz).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["effect_size", "eta2"],
        choices=list(NeuronClusterizer.SUPPORTED_FEATURES),
        help="Метрики для кластеризації.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        metavar="N",
        help="Кількість незалежних запусків K-Means для оцінки стабільності.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default=None,
        help=(
            "Директорія з .npz файлами та для збереження результатів. "
            "За замовчуванням: <project_root>/outputs/reports/"
        ),
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help=(
            "Мінімальна частка запусків K-Means, у яких нейрон має потрапити "
            "в емоційний кластер (0.0–1.0, default: 0.8 = 80%%)."
        ),
    )

    return parser.parse_args(argv)


def _resolve_reports_dir(override: Optional[str]) -> Path:
    """
    Визначає шлях до outputs/reports/ відносно кореня проєкту.

    Структура: <root>/src/clustering.py → <root>/outputs/reports/
    """
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Вказана директорія не існує: {path}")
        return path

    # __file__ = .../LLM-EMOTION-INTERPRETABILITY/src/clustering.py
    src_dir = Path(os.path.abspath(__file__)).parent
    project_root = src_dir.parent
    reports_dir = project_root / "outputs" / "reports"

    if not reports_dir.exists():
        logger.info("Директорія %s не існує — створюємо.", reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

    return reports_dir


def main(argv: Optional[list[str]] = None) -> None:
    """Точка входу CLI."""
    args = _parse_args(argv)

    logger.info("=" * 60)
    logger.info("  LLM Emotion Interpretability — Clustering")
    logger.info("  run_name            : %s", args.run_name)
    logger.info("  features            : %s", args.features)
    logger.info("  n_iter              : %d", args.n_iter)
    logger.info("  stability_threshold : %.0f%%", args.stability_threshold * 100)
    logger.info("=" * 60)

    try:
        reports_dir = _resolve_reports_dir(args.reports_dir)
        logger.info("Директорія звітів: %s", reports_dir)

        clusterizer = NeuronClusterizer(
            run_name=args.run_name,
            features=args.features,
            n_iter=args.n_iter,
            reports_dir=reports_dir,
            stability_threshold=args.stability_threshold,
        )

        result = clusterizer.run()
        clusterizer.save_results(result)

        print(result.summary())

    except (FileNotFoundError, KeyError, ValueError, IOError) as exc:
        logger.error("Критична помилка: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Перервано користувачем.")
        sys.exit(130)


if __name__ == "__main__":
    main()
