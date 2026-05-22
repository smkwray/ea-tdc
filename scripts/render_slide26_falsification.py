#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLIDES = Path(os.environ.get("EA_TDC_SLIDES_HTML", REPO_ROOT / "presentation" / "slides.html"))
DEFAULT_PARTIAL = Path(
    os.environ.get("EA_TDC_SLIDE26_PARTIAL", REPO_ROOT / "output" / "reports" / "slide26_falsification.html")
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt_value(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def fmt_plain(value: float) -> str:
    return f"{value:.2f}"


def xscale(value: float, *, min_x: float, max_x: float, left: float, right: float) -> float:
    return left + (value - min_x) / (max_x - min_x) * (right - left)


def yscale(value: float, *, max_y: float, top: float, bottom: float) -> float:
    return bottom - value / max_y * (bottom - top)


def load_window_rows() -> list[dict[str, Any]]:
    full_path = REPO_ROOT / "output/models/baseline_tdc_lp_deposits__robustness_k200_estimates.csv"
    regime_path = REPO_ROOT / "output/reports/baseline_tdc_lp_deposits__regime_sensitivity_estimates.csv"

    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(full_path):
        if row["outcome"] == "matched_total_deposits" and row["horizon"] == "0":
            rows.append(
                {
                    "label": "Full sample",
                    "n": int(row["n"]),
                    "beta": as_float(row, "beta"),
                    "lower95": as_float(row, "lower95"),
                    "upper95": as_float(row, "upper95"),
                }
            )
            break

    label_for_regime = {
        "exclude_2008_2009": "Excl. 2008-09",
        "exclude_2020": "Excl. 2020",
    }
    for regime_id, label in label_for_regime.items():
        for row in read_csv_rows(regime_path):
            if (
                row["regime_id"] == regime_id
                and row["outcome"] == "matched_total_deposits"
                and row["horizon"] == "0"
            ):
                rows.append(
                    {
                        "label": label,
                        "n": int(row["n"]),
                        "beta": as_float(row, "beta"),
                        "lower95": as_float(row, "lower95"),
                        "upper95": as_float(row, "upper95"),
                    }
                )
                break

    if len(rows) != 3:
        raise RuntimeError(f"Expected 3 same-quarter window rows, got {len(rows)}")
    return rows


def load_lead_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "output/reports/baseline_tdc_lp_deposits__negative_control_mining.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload.get("lead_placebos", []), key=lambda row: int(row["lead"]))
    if [int(row["lead"]) for row in rows] != [1, 2, 4]:
        raise RuntimeError("Expected lead placebo rows for leads 1, 2, and 4")
    return rows


def render_combined_svg(window_rows: list[dict[str, Any]], lead_rows: list[dict[str, Any]]) -> str:
    parts = [
        '<svg class="falsviz" viewBox="0 0 940 252" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pass-through falsification screens: same-quarter robustness and lead placebo diagnostics">',
        '<text x="88" y="27" class="bandttl">Same-quarter response across sample windows</text>',
        '<text x="540" y="27" class="bandttl">Lead-placebo timing screen</text>',
        '<text x="88" y="45" class="subttl">Matched total deposits, h=0, 95% CI</text>',
        '<text x="540" y="45" class="subttl">Future-dated TDC, four tested horizons per lead</text>',
        '<line x1="500" y1="22" x2="500" y2="224" class="sep"/>',
    ]

    # Left panel: coefficient forest.
    min_x, max_x = -0.10, 1.00
    left, right = 184.0, 456.0
    top, bottom = 60.0, 205.0
    row_ys = [76.0, 123.0, 170.0]
    zero_x = xscale(0, min_x=min_x, max_x=max_x, left=left, right=right)
    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x = xscale(tick, min_x=min_x, max_x=max_x, left=left, right=right)
        parts.append(f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 4:.1f}" class="tick"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 18:.1f}" class="axlbl" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" class="axis"/>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top:.1f}" x2="{zero_x:.1f}" y2="{bottom:.1f}" class="zero"/>')

    for row, y in zip(window_rows, row_ys):
        lo = xscale(row["lower95"], min_x=min_x, max_x=max_x, left=left, right=right)
        hi = xscale(row["upper95"], min_x=min_x, max_x=max_x, left=left, right=right)
        beta = xscale(row["beta"], min_x=min_x, max_x=max_x, left=left, right=right)
        parts.extend(
            [
                f'<text x="{left - 18:.1f}" y="{y - 3:.1f}" class="rowlbl" text-anchor="end">{html.escape(row["label"])}</text>',
                f'<text x="{left - 18:.1f}" y="{y + 12:.1f}" class="nlabel" text-anchor="end">n={row["n"]}</text>',
                f'<line x1="{lo:.1f}" y1="{y:.1f}" x2="{hi:.1f}" y2="{y:.1f}" class="ci-line"/>',
                f'<line x1="{lo:.1f}" y1="{y - 7:.1f}" x2="{lo:.1f}" y2="{y + 7:.1f}" class="ci-cap"/>',
                f'<line x1="{hi:.1f}" y1="{y - 7:.1f}" x2="{hi:.1f}" y2="{y + 7:.1f}" class="ci-cap"/>',
                f'<circle cx="{beta:.1f}" cy="{y:.1f}" r="4.2" class="point"/>',
                f'<text x="{beta:.1f}" y="{y - 12:.1f}" class="value-label" text-anchor="middle">{fmt_value(row["beta"])}</text>',
            ]
        )

    # Right panel: bars.
    max_y = 4.2
    bar_left, bar_right = 592.0, 900.0
    bar_top, bar_bottom = 58.0, 205.0
    centers = [654.0, 754.0, 854.0]
    bar_w = 23.0
    gap = 6.0
    parts.append(f'<line x1="{bar_left:.1f}" y1="{bar_top:.1f}" x2="{bar_left:.1f}" y2="{bar_bottom:.1f}" class="axis"/>')
    for tick in [0, 1, 2, 3, 4]:
        y = yscale(tick, max_y=max_y, top=bar_top, bottom=bar_bottom)
        cls = "zero-grid" if tick == 0 else "threshold-grid" if tick == 2 else "grid"
        parts.append(f'<line x1="{bar_left:.1f}" y1="{y:.1f}" x2="{bar_right:.1f}" y2="{y:.1f}" class="{cls}"/>')
        parts.append(f'<text x="{bar_left - 8:.1f}" y="{y + 3.2:.1f}" class="axlbl" text-anchor="end">{tick}</text>')
    threshold_y = yscale(2, max_y=max_y, top=bar_top, bottom=bar_bottom)
    parts.append(f'<text x="{bar_right - 2:.1f}" y="{threshold_y + 16:.1f}" class="threshold-label" text-anchor="end">|z|=2</text>')
    parts.append(f'<text x="548" y="{(bar_top + bar_bottom) / 2:.1f}" class="ytitle" text-anchor="middle" transform="rotate(-90 548 {(bar_top + bar_bottom) / 2:.1f})">|z| statistic</text>')

    for row, center in zip(lead_rows, centers):
        lead = int(row["lead"])
        avg = float(row["avg_abs_z"])
        max_abs = float(row["max_abs_z"])
        color_class = "warn" if lead == 1 else "quiet"
        x1 = center - bar_w - gap / 2
        x2 = center + gap / 2
        avg_y = yscale(avg, max_y=max_y, top=bar_top, bottom=bar_bottom)
        max_y_pos = yscale(max_abs, max_y=max_y, top=bar_top, bottom=bar_bottom)
        parts.extend(
            [
                f'<rect x="{x1:.1f}" y="{avg_y:.1f}" width="{bar_w:.1f}" height="{bar_bottom - avg_y:.1f}" class="bar avg {color_class}"/>',
                f'<rect x="{x2:.1f}" y="{max_y_pos:.1f}" width="{bar_w:.1f}" height="{bar_bottom - max_y_pos:.1f}" class="bar max {color_class}"/>',
                f'<text x="{x1 + bar_w / 2:.1f}" y="{avg_y - 7:.1f}" class="bar-label {color_class}" text-anchor="middle">{fmt_plain(avg)}</text>',
                f'<text x="{x2 + bar_w / 2:.1f}" y="{max_y_pos - 7:.1f}" class="bar-label maxlbl {color_class}" text-anchor="middle">{fmt_plain(max_abs)}</text>',
                f'<text x="{center:.1f}" y="{bar_bottom + 19:.1f}" class="xlbl" text-anchor="middle">Lead {lead}</text>',
            ]
        )

    parts.extend(
        [
            '<g class="leg-txt">',
            '<rect x="767" y="63" width="10" height="8" class="legend-swatch avg"/>',
            '<text x="782" y="71">avg |z|</text>',
            '<rect x="767" y="79" width="10" height="8" class="legend-swatch max"/>',
            '<text x="782" y="87">max |z|</text>',
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def render_section() -> str:
    window_rows = load_window_rows()
    lead_rows = load_lead_rows()
    clean_candidates = read_csv_rows(REPO_ROOT / "output/reports/baseline_tdc_lp_deposits__negative_control_mining.csv")
    clean_values = sorted(float(row["avg_abs_z"]) for row in clean_candidates)[:5]
    clean_min = min(clean_values)
    clean_max = max(clean_values)

    return f"""  <!-- GENERATED: slide 26 falsification screens. Source: scripts/render_slide26_falsification.py -->
  <style>
    .falsviz {{
      width: 100%;
      max-width: 1160px;
      height: auto;
      display: block;
      margin: 0 auto 0.02em;
    }}
    .falsviz .grid {{
      stroke: var(--line);
      stroke-width: 0.65;
    }}
    .falsviz .axis,
    .falsviz .tick {{
      stroke: var(--line-strong);
      stroke-width: 0.85;
    }}
    .falsviz .zero,
    .falsviz .threshold-grid {{
      stroke: var(--line-strong);
      stroke-width: 1.1;
      stroke-dasharray: 3 3;
    }}
    .falsviz .zero-grid {{
      stroke: var(--line-strong);
      stroke-width: 0.95;
    }}
    .falsviz .sep {{
      stroke: var(--line);
      stroke-width: 0.8;
    }}
    .falsviz .bandttl {{
      font-size: 12px;
      fill: var(--heading);
      font-family: Inter, sans-serif;
      font-weight: 700;
    }}
    .falsviz .subttl {{
      font-family: Inter, sans-serif;
      font-size: 9.5px;
      fill: var(--muted);
    }}
    .falsviz .axlbl {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 9px;
      fill: var(--muted);
    }}
    .falsviz .rowlbl,
    .falsviz .xlbl,
    .falsviz .ytitle {{
      font-family: Inter, sans-serif;
      font-size: 10.5px;
      fill: var(--heading);
      font-weight: 650;
    }}
    .falsviz .nlabel,
    .falsviz .threshold-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 8.7px;
      fill: var(--muted);
    }}
    .falsviz .ci-line,
    .falsviz .ci-cap,
    .falsviz .point,
    .falsviz .bar.warn,
    .falsviz .bar-label.warn {{
      stroke: var(--accent);
      fill: var(--accent);
    }}
    .falsviz .ci-line,
    .falsviz .ci-cap {{
      stroke-width: 1.55;
    }}
    .falsviz .value-label,
    .falsviz .bar-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 9.5px;
      font-weight: 700;
      fill: var(--accent);
    }}
    .falsviz .bar.quiet,
    .falsviz .bar-label.quiet {{
      stroke: var(--accent-2);
      fill: var(--accent-2);
    }}
    .falsviz .bar.avg {{ opacity: 0.88; }}
    .falsviz .bar.max {{ opacity: 0.40; }}
    .falsviz .legend-swatch.avg {{
      fill: var(--heading);
      opacity: 0.72;
    }}
    .falsviz .legend-swatch.max {{
      fill: var(--heading);
      opacity: 0.28;
    }}
    .falsviz .maxlbl {{ font-weight: 600; }}
    .falsviz .leg-txt text {{
      font-family: Inter, sans-serif;
      font-size: 9.4px;
      fill: var(--heading);
    }}
    .fals-rand {{
      margin-top: 4px;
      padding: 6px 12px;
      background: var(--panel);
      border-left: 3px solid var(--accent-3);
      font-size: 13px;
      color: var(--heading);
      font-family: Inter, sans-serif;
      line-height: 1.35;
    }}
    .fals-rand strong {{ color: var(--accent-3); }}
  </style>

  <section>
    <div class="section-marker">Empirical validation <span class="dot">·</span> Falsification screens</div>
    <h2>What would make the pass-through result fail?</h2>
    <div class="slide-subtitle">K=200 factor-augmented LP · sample-window robustness and lead-placebo timing checks</div>

    {render_combined_svg(window_rows, lead_rows)}

    <div class="fals-rand">
      <strong>Sanity check:</strong> the five cleanest non-response outcomes average |z| {clean_min:.2f}-{clean_max:.2f}, so the screen is not mechanically manufacturing significance.
    </div>

    <ul style="margin-top: 8px;">
      <li><strong>Holds up:</strong> the same-quarter coefficient stays positive and significant when 2008-09 or 2020 is excluded.</li>
      <li><strong>Lead-1 warning.</strong> Deposits co-move with TDC one quarter ahead (avg |z|&asymp;{float(lead_rows[0]["avg_abs_z"]):.1f}, max {float(lead_rows[0]["max_abs_z"]):.1f}), so timing is not fully clean. Leads 2 and 4 are much quieter, keeping the concern local to lead 1.</li>
      <li><strong>Read:</strong> h=0 is reported as a pass-through <em>association</em>, not a strict causal coefficient; h=4 persistence remains factor-screening evidence.</li>
    </ul>
  </section>
"""


def update_slides(slides_path: Path, section_html: str) -> None:
    text = slides_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"  <!-- ═+\n"
        r"       SLIDE 14B · FALSIFICATION SCREENS\n"
        r"       ═+ -->\n"
        r"(?P<body>.*?)"
        r"(?=\n\n  <!-- ═+\n       SLIDE 24 · DOMESTIC DEPOSIT PASS-THROUGH)",
        re.DOTALL,
    )
    replacement = (
        "  <!-- ═══════════════════════════════════════════════════════\n"
        "       SLIDE 14B · FALSIFICATION SCREENS\n"
        "       ═══════════════════════════════════════════════════════ -->\n"
        f"{section_html.rstrip()}"
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the slide 26 falsification section to replace")
    slides_path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", type=Path, default=DEFAULT_SLIDES)
    parser.add_argument("--partial", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--update-slides", action="store_true")
    args = parser.parse_args()

    section_html = render_section()
    args.partial.parent.mkdir(parents=True, exist_ok=True)
    args.partial.write_text(section_html, encoding="utf-8")
    if args.update_slides:
        update_slides(args.slides, section_html)
    print(args.partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
