"""Benchmark comparison results with formatting and export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkComparison:
    """Holds results from :func:`compare_benchmarks` and provides
    formatting, comparison, and export utilities.

    Parameters
    ----------
    model : str
        The model used across all strategies.
    tasks : list[str]
        The benchmark tasks that were run.
    results : dict[str, dict]
        Map of strategy name → raw lm-eval results dict.
    """

    model: str
    tasks: list[str]
    results: dict[str, dict[str, Any]]

    # -- Extraction helpers ------------------------------------------------

    def _extract_scores(self) -> dict[str, dict[str, float | None]]:
        """Extract metric scores per strategy per task.

        Returns ``{strategy: {task_metric: score}}``.
        """
        scores: dict[str, dict[str, float | None]] = {}
        for strategy_name, raw in self.results.items():
            strategy_scores: dict[str, float | None] = {}
            task_results = raw.get("results", {})
            for task_name, metrics in task_results.items():
                for metric_key, value in metrics.items():
                    if metric_key.endswith(",none"):
                        metric_key = metric_key[: -len(",none")]
                    if isinstance(value, (int, float)):
                        key = f"{task_name}/{metric_key}"
                        strategy_scores[key] = round(float(value), 4)
            scores[strategy_name] = strategy_scores
        return scores

    # -- Formatting --------------------------------------------------------

    def print_table(self) -> str:
        """Print a formatted comparison table and return it as a string.

        Rows = metric (task/metric_name), Columns = strategies.
        """
        scores = self._extract_scores()
        strategies = list(scores.keys())

        # Collect all unique metric keys across strategies
        all_metrics: list[str] = []
        seen: set[str] = set()
        for strat_scores in scores.values():
            for k in strat_scores:
                if k not in seen:
                    all_metrics.append(k)
                    seen.add(k)

        if not all_metrics:
            msg = "(No benchmark results to display)"
            print(msg)
            return msg

        # Column widths
        metric_col_width = max(len(m) for m in all_metrics)
        metric_col_width = max(metric_col_width, len("Metric"))
        strat_col_width = max(max((len(s) for s in strategies), default=8), 10)

        # Header
        header = f"{'Metric':<{metric_col_width}}"
        for s in strategies:
            header += f"  {s:>{strat_col_width}}"
        separator = "-" * len(header)

        lines = [
            f"Model: {self.model}",
            f"Tasks: {', '.join(self.tasks)}",
            "",
            header,
            separator,
        ]

        # Find best score per metric for highlighting
        for metric in all_metrics:
            vals = {s: scores[s].get(metric) for s in strategies}
            best_val = max(
                (v for v in vals.values() if v is not None), default=None
            )

            row = f"{metric:<{metric_col_width}}"
            for s in strategies:
                v = vals[s]
                if v is None:
                    cell = "—"
                elif v == best_val and len(strategies) > 1:
                    cell = f"{v:.4f} ★"
                else:
                    cell = f"{v:.4f}"
                row += f"  {cell:>{strat_col_width}}"
            lines.append(row)

        lines.append(separator)
        table = "\n".join(lines)
        print(table)
        return table

    def best_strategy(self, metric: str) -> str | None:
        """Return the strategy name with the highest score for *metric*.

        Parameters
        ----------
        metric : str
            Metric key like ``"gsm8k/acc"`` or ``"mmlu/acc"``.
        """
        scores = self._extract_scores()
        best_name: str | None = None
        best_val: float | None = None
        for strat, strat_scores in scores.items():
            val = strat_scores.get(metric)
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_name = strat
        return best_name

    # -- Export ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "model": self.model,
            "tasks": self.tasks,
            "scores": self._extract_scores(),
        }

    def to_json(self, path: str) -> None:
        """Write comparison results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path: str) -> None:
        """Write comparison results to a CSV file."""
        import csv

        scores = self._extract_scores()
        strategies = list(scores.keys())

        all_metrics: list[str] = []
        seen: set[str] = set()
        for strat_scores in scores.values():
            for k in strat_scores:
                if k not in seen:
                    all_metrics.append(k)
                    seen.add(k)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + strategies)
            for metric in all_metrics:
                row = [metric]
                for s in strategies:
                    val = scores[s].get(metric)
                    row.append(f"{val:.4f}" if val is not None else "")
                writer.writerow(row)
