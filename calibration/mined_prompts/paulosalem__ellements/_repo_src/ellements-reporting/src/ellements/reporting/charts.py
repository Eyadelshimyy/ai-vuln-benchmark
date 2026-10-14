"""Structured chart generation helpers for Canvas/chat/report surfaces."""

from __future__ import annotations

import base64
import html
import json
import math
from io import BytesIO
from typing import Any

_SUPPORTED_CHART_TYPES = {"line", "bar", "pie", "histogram", "scatter", "area"}


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        raise ValueError(
            "Chart rendering requires matplotlib. Install with: pip install matplotlib"
        ) from exc
    return plt


def _coerce_float(value: Any, field_name: str, *, allow_nan: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} values must be numeric.")
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} values must be numeric.") from exc
    if math.isnan(parsed):
        if allow_nan:
            return parsed
        raise ValueError(f"{field_name} values must be finite.")
    if math.isinf(parsed):
        raise ValueError(f"{field_name} values must be finite.")
    return parsed


def _coerce_numeric_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array.")
    return [_coerce_float(item, field_name, allow_nan=True) for item in value]


def _coerce_x_axis(value: Any, count: int, field_name: str) -> tuple[list[float], list[str] | None]:
    if value is None:
        return [float(index + 1) for index in range(count)], None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided.")
    if len(value) != count:
        raise ValueError(f"{field_name} length must match y/values length.")

    numeric_values: list[float] = []
    all_numeric = True
    for item in value:
        try:
            numeric_values.append(_coerce_float(item, field_name))
        except ValueError:
            all_numeric = False
            break
    if all_numeric:
        return numeric_values, None
    labels = [str(item).strip() for item in value]
    return [float(index) for index in range(count)], labels


def _normalize_chart_type(spec: dict[str, Any]) -> str:
    raw = str(spec.get("type") or spec.get("chart_type") or "").strip().lower()
    if raw == "hist":
        raw = "histogram"
    if raw not in _SUPPORTED_CHART_TYPES:
        raise ValueError(
            "Unsupported chart type. Supported types: "
            + ", ".join(sorted(_SUPPORTED_CHART_TYPES))
            + "."
        )
    return raw


def _normalize_series(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw_series = spec.get("series")
    if isinstance(raw_series, list) and raw_series:
        normalized: list[dict[str, Any]] = []
        for index, entry in enumerate(raw_series):
            if not isinstance(entry, dict):
                continue
            raw_y = entry.get("y", entry.get("values"))
            if raw_y is None:
                continue
            y_values = _coerce_numeric_list(raw_y, f"series[{index}].y")
            x_raw = entry.get("x", entry.get("labels", spec.get("x", spec.get("labels"))))
            x_values, x_labels = _coerce_x_axis(x_raw, len(y_values), f"series[{index}].x")
            name = str(entry.get("name", entry.get("label", f"Series {index + 1}"))).strip()
            color = str(entry.get("color", "")).strip()
            normalized.append(
                {
                    "name": name or f"Series {index + 1}",
                    "x": x_values,
                    "x_labels": x_labels,
                    "y": y_values,
                    "color": color or None,
                }
            )
        if normalized:
            return normalized

    raw_y = spec.get("y", spec.get("values"))
    if raw_y is None:
        raise ValueError("chart spec must include y/values or series.")
    y_values = _coerce_numeric_list(raw_y, "y")
    x_values, x_labels = _coerce_x_axis(spec.get("x", spec.get("labels")), len(y_values), "x")
    series_name = str(spec.get("series_name", "Series 1")).strip() or "Series 1"
    color = str(spec.get("color", "")).strip()
    return [
        {
            "name": series_name,
            "x": x_values,
            "x_labels": x_labels,
            "y": y_values,
            "color": color or None,
        }
    ]


def _build_figure(spec: dict[str, Any]) -> tuple[Any, str]:
    plt = _load_matplotlib()
    chart_type = _normalize_chart_type(spec)
    width = _coerce_float(spec.get("width", 8), "width")
    height = _coerce_float(spec.get("height", 4.8), "height")
    fig, ax = plt.subplots(figsize=(max(3.5, min(width, 18.0)), max(2.5, min(height, 12.0))))
    series = _normalize_series(spec) if chart_type in {"line", "bar", "scatter", "area"} else []

    if chart_type in {"line", "scatter", "area"}:
        for index, entry in enumerate(series):
            name = str(entry.get("name", f"Series {index + 1}")).strip() or f"Series {index + 1}"
            color = entry.get("color")
            x_values = entry.get("x", [])
            y_values = entry.get("y", [])
            x_labels = entry.get("x_labels")
            if chart_type == "line":
                ax.plot(x_values, y_values, label=name, color=color)
            elif chart_type == "scatter":
                ax.scatter(x_values, y_values, label=name, color=color)
            else:
                ax.plot(x_values, y_values, label=name, color=color)
                ax.fill_between(x_values, y_values, alpha=0.3, color=color)
            if isinstance(x_labels, list) and x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=25, ha="right")

    elif chart_type == "bar":
        stacked = bool(spec.get("stacked", False))
        if len(series) == 1:
            entry = series[0]
            x_values = entry.get("x", [])
            y_values = entry.get("y", [])
            ax.bar(x_values, y_values, color=entry.get("color"), label=str(entry.get("name", "")).strip() or None)
            x_labels = entry.get("x_labels")
            if isinstance(x_labels, list) and x_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=25, ha="right")
        else:
            max_count = max(len(entry.get("y", [])) for entry in series)
            explicit_labels = spec.get("labels", spec.get("x"))
            if isinstance(explicit_labels, list) and len(explicit_labels) == max_count:
                category_labels = [str(item).strip() for item in explicit_labels]
            else:
                first_labels = series[0].get("x_labels")
                if isinstance(first_labels, list) and len(first_labels) == max_count:
                    category_labels = [str(item).strip() for item in first_labels]
                else:
                    category_labels = [str(index + 1) for index in range(max_count)]
            positions = [float(index) for index in range(max_count)]
            if stacked:
                bottoms = [0.0 for _ in range(max_count)]
                for index, entry in enumerate(series):
                    y_raw = list(entry.get("y", []))
                    y_values = y_raw + [0.0 for _ in range(max_count - len(y_raw))]
                    ax.bar(
                        positions,
                        y_values,
                        bottom=bottoms,
                        label=str(entry.get("name", f"Series {index + 1}")),
                        color=entry.get("color"),
                    )
                    bottoms = [bottoms[i] + y_values[i] for i in range(max_count)]
            else:
                width_each = 0.8 / max(1, len(series))
                for index, entry in enumerate(series):
                    y_raw = list(entry.get("y", []))
                    y_values = y_raw + [0.0 for _ in range(max_count - len(y_raw))]
                    offsets = [
                        position - 0.4 + (width_each / 2.0) + (index * width_each)
                        for position in positions
                    ]
                    ax.bar(
                        offsets,
                        y_values,
                        width=width_each,
                        label=str(entry.get("name", f"Series {index + 1}")),
                        color=entry.get("color"),
                    )
            ax.set_xticks(positions)
            ax.set_xticklabels(category_labels, rotation=25, ha="right")

    elif chart_type == "pie":
        raw_values = spec.get("values", spec.get("y"))
        values = _coerce_numeric_list(raw_values, "values")
        raw_labels = spec.get("labels", spec.get("x"))
        pie_labels: list[str] | None = None
        if isinstance(raw_labels, list) and len(raw_labels) == len(values):
            pie_labels = [str(item).strip() for item in raw_labels]
        colors = spec.get("colors")
        if not isinstance(colors, list):
            colors = None
        autopct = str(spec.get("autopct", "%1.1f%%")).strip()
        show_pct = bool(spec.get("show_percentages", True))
        wedgeprops = {"width": 0.45} if bool(spec.get("donut", False)) else None
        ax.pie(
            values,
            labels=pie_labels,
            colors=colors,
            autopct=autopct if show_pct else None,
            startangle=90,
            wedgeprops=wedgeprops,
        )
        ax.axis("equal")

    elif chart_type == "histogram":
        bins = int(_coerce_float(spec.get("bins", 10), "bins"))
        bins = max(2, min(200, bins))
        color = str(spec.get("color", "")).strip() or None
        raw_series = spec.get("series")
        if isinstance(raw_series, list) and raw_series:
            datasets: list[list[float]] = []
            series_labels: list[str] = []
            for index, entry in enumerate(raw_series):
                if not isinstance(entry, dict):
                    continue
                raw_values = entry.get("values", entry.get("y"))
                if raw_values is None:
                    continue
                datasets.append(_coerce_numeric_list(raw_values, f"series[{index}].values"))
                series_labels.append(str(entry.get("name", entry.get("label", f"Series {index + 1}"))).strip() or f"Series {index + 1}")
            if not datasets:
                raise ValueError("histogram series must include values.")
            ax.hist(
                datasets,
                bins=bins,
                stacked=bool(spec.get("stacked", False)),
                alpha=0.65,
                label=series_labels if len(series_labels) > 1 else None,
            )
        else:
            values = _coerce_numeric_list(spec.get("values", spec.get("y")), "values")
            ax.hist(values, bins=bins, alpha=0.75, color=color)

    title = str(spec.get("title", "")).strip()
    x_label = str(spec.get("x_label", spec.get("xlabel", ""))).strip()
    y_label = str(spec.get("y_label", spec.get("ylabel", ""))).strip()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if bool(spec.get("grid", chart_type in {"line", "bar", "scatter", "area", "histogram"})):
        ax.grid(True, alpha=0.25)
    if chart_type != "pie" and bool(spec.get("legend", len(series) > 1)):
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()

    fig.tight_layout()
    return fig, chart_type


def _serialize_figure(fig: Any, image_format: str, dpi: int) -> tuple[str, str]:
    clean_format = str(image_format or "png").strip().lower()
    if clean_format not in {"png", "svg"}:
        raise ValueError("image_format must be 'png' or 'svg'.")

    buffer = BytesIO()
    if clean_format == "png":
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=max(60, min(int(dpi), 240)))
        mime_type = "image/png"
    else:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        mime_type = "image/svg+xml"
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded, mime_type


def create_chart_artifacts(
    chart_spec: dict[str, Any],
    *,
    title: str = "",
    image_format: str = "png",
    dpi: int = 120,
) -> dict[str, Any]:
    """Render a chart from a JSON-style spec and return reusable assets."""
    if not isinstance(chart_spec, dict):
        raise ValueError("chart_spec must be a JSON object.")

    plt = _load_matplotlib()
    fig, chart_type = _build_figure(chart_spec)
    image_base64, mime_type = _serialize_figure(fig, image_format, dpi)
    plt.close(fig)

    resolved_title = str(title or chart_spec.get("title", "")).strip() or "Chart"
    alt = str(chart_spec.get("alt", resolved_title)).strip() or "Chart"
    caption = str(chart_spec.get("caption", "")).strip()
    data_uri = f"data:{mime_type};base64,{image_base64}"
    safe_alt = html.escape(alt)
    safe_caption = html.escape(caption)
    safe_title = html.escape(resolved_title)
    html_img_tag = (
        f'<img src="{data_uri}" alt="{safe_alt}" '
        'style="max-width:100%;height:auto;border-radius:10px;border:1px solid rgba(148,163,184,0.35);" />'
    )
    html_document = (
        "<!doctype html><html><head><meta charset=\"utf-8\" />"
        f"<title>{safe_title}</title>"
        "</head><body style=\"margin:0;padding:20px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\">"
        f"<h2 style=\"margin:0 0 12px;\">{safe_title}</h2>"
        f"<figure style=\"margin:0;\">{html_img_tag}"
        + (f"<figcaption style=\"margin-top:8px;color:#64748b;\">{safe_caption}</figcaption>" if caption else "")
        + "</figure></body></html>"
    )
    canvas_chart_payload = {
        "type": chart_type,
        "title": resolved_title,
        "alt": alt,
        "caption": caption,
        "data_uri": data_uri,
    }
    canvas_chart_block = "```canvas-chart\n" + json.dumps(
        canvas_chart_payload,
        ensure_ascii=False,
    ) + "\n```"
    canvas_card_item = {
        "type": "image",
        "label": resolved_title,
        "url": data_uri,
        "alt": alt,
        "caption": caption,
    }

    return {
        "title": resolved_title,
        "chart_type": chart_type,
        "image_format": "svg" if mime_type == "image/svg+xml" else "png",
        "mime_type": mime_type,
        "image_base64": image_base64,
        "data_uri": data_uri,
        "html_img_tag": html_img_tag,
        "html_document": html_document,
        "canvas_chart_payload": canvas_chart_payload,
        "canvas_chart_block": canvas_chart_block,
        "canvas_card_item": canvas_card_item,
    }
