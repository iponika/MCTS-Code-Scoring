from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path("/data1/xianzhiwei/mcts-code-review")
INPUT = ROOT / "model_training/src/output/review-eval-report_pilot_training_20260420/comparison.json"
OUT_DIR = ROOT / "docs/report_pilot_20260421"

ORDER = [
    ("base_direct_det", "Base Direct", "#5B8FF9"),
    ("report_direct_det", "Trained Direct", "#61DDAA"),
    ("report_value_guided_mcts", "Trained Value-Guided MCTS", "#F6BD16"),
]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def panel_svg(
    title: str,
    metrics: list[tuple[str, str, bool]],
    rows: dict[str, dict[str, float]],
    width: int = 1320,
    height: int = 760,
) -> str:
    bg = "#FFFDF8"
    axis = "#353535"
    grid = "#D9D3C7"
    text = "#1F1F1F"
    small = "#666666"
    bar_w = 70
    group_gap = 70
    chart_left = 90
    chart_right = 40
    chart_top = 140
    chart_bottom = 120
    chart_h = height - chart_top - chart_bottom
    group_w = bar_w * len(ORDER) + group_gap
    max_val = 1.6

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="60" y="60" font-size="30" font-weight="700" fill="{text}">{escape(title)}</text>',
        f'<text x="60" y="92" font-size="16" fill="{small}">纵轴统一为 0-1 或 0-1.6；MAE 越低越好，其余指标越高越好。</text>',
    ]

    legend_x = width - 360
    legend_y = 50
    for idx, (_, label, color) in enumerate(ORDER):
        y = legend_y + idx * 28
        parts.append(f'<rect x="{legend_x}" y="{y-12}" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{y+1}" font-size="15" fill="{text}">{escape(label)}</text>')

    for i in range(5):
        y = chart_top + chart_h * i / 4
        val = max_val * (1 - i / 4)
        parts.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{width-chart_right}" y2="{y:.1f}" stroke="{grid}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{chart_left-12}" y="{y+5:.1f}" font-size="14" text-anchor="end" fill="{small}">{val:.1f}</text>'
        )

    parts.append(
        f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top+chart_h}" stroke="{axis}" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{chart_left}" y1="{chart_top+chart_h}" x2="{width-chart_right}" y2="{chart_top+chart_h}" stroke="{axis}" stroke-width="2"/>'
    )

    for metric_idx, (key, label, lower_better) in enumerate(metrics):
        group_x = chart_left + 40 + metric_idx * group_w
        center_x = group_x + (len(ORDER) * bar_w) / 2
        parts.append(
            f'<text x="{center_x:.1f}" y="{height-55}" font-size="16" text-anchor="middle" fill="{text}">{escape(label)}</text>'
        )
        if lower_better:
            parts.append(
                f'<text x="{center_x:.1f}" y="{height-30}" font-size="12" text-anchor="middle" fill="{small}">越低越好</text>'
            )
        else:
            parts.append(
                f'<text x="{center_x:.1f}" y="{height-30}" font-size="12" text-anchor="middle" fill="{small}">越高越好</text>'
            )
        for order_idx, (row_key, _, color) in enumerate(ORDER):
            value = float(rows[row_key][key])
            bar_h = min(chart_h, chart_h * value / max_val)
            x = group_x + order_idx * bar_w
            y = chart_top + chart_h - bar_h
            parts.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_w-12}" height="{bar_h:.1f}" rx="6" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + (bar_w-12)/2:.1f}" y="{y-8:.1f}" font-size="13" text-anchor="middle" fill="{text}">{fmt(value)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_metrics = [
        ("grade_mae", "MAE", True),
        ("boundary_acc", "Boundary Acc", False),
        ("valid_rate", "Valid Rate", False),
    ]
    risk_metrics = [
        ("low_grade_false_positive_rate", "Low-FP", True),
        ("high_grade_false_negative_rate", "High-FN", True),
        ("unsupported_evidence_rate", "Unsupported", True),
    ]

    (OUT_DIR / "figure_main_metrics.svg").write_text(
        panel_svg("Figure 1. Main Metrics", main_metrics, rows),
        encoding="utf-8",
    )
    (OUT_DIR / "figure_risk_metrics.svg").write_text(
        panel_svg("Figure 2. Risk And Reliability Metrics", risk_metrics, rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
