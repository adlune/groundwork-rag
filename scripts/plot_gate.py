"""Draw docs/figures/gate-scores.svg from eval/results.json.

Every evaluation question is a dot: x = best meaning (cosine) score among its retrieved
chunks, y = best keyword (BM25) score. The shaded L-shaped area is what the gate
accepts: meaning >= threshold OR keywords >= threshold.

    uv run python scripts/plot_gate.py
"""

import json

from groundwork.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "eval" / "results.json"
OUT = PROJECT_ROOT / "docs" / "figures" / "gate-scores.svg"

# Same visual style as every other figure in docs/figures.
STYLE = """<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#0b0b0b;--hair:#c3c2b7;--grid:#e1e0d9;--gray:#ecebe7;--bar:#b9b8b2;--accent:#2a78d6;--accent-wash:#e3eefb}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#d6d5d0;--hair:#4a4a45;--grid:#2c2c2a;--gray:#2a2a28;--bar:#5c5b57;--accent:#3987e5;--accent-wash:#17293f}}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink)}
.bg{fill:var(--surface)}
.zone{fill:var(--accent-wash)}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--hair);stroke-width:1;fill:none}
.thr{stroke:var(--line);stroke-width:1.5;stroke-dasharray:5 4}
.ans{fill:var(--accent);stroke:var(--surface);stroke-width:1.5}
.oos{fill:var(--surface);stroke:var(--line);stroke-width:1.5}
.una{fill:var(--bar);stroke:var(--surface);stroke-width:1.5}
.lead{stroke:var(--ink2);stroke-width:1}
.h{font-size:15px;font-weight:600}
.s{font-size:12px;fill:var(--ink2)}
.m{font-size:11px;fill:var(--muted)}
</style>"""

W, H = 860, 500
X0, X1, Y0, Y1 = 70, 680, 430, 110  # plot area in pixels (y grows downward)
XMIN, XMAX = 0.40, 0.90  # meaning score range shown

LABELS = {  # point id -> (label, dx, dy, anchor) for a few telling points
    "oos-kubernetes": ("“Kubernetes”", 8, -9, "start"),
    "copy-model-api": ("“copy a model via its API”", 8, 16, "start"),
}


def sx(v: float) -> float:
    return X0 + (min(max(v, XMIN), XMAX) - XMIN) / (XMAX - XMIN) * (X1 - X0)


def sy(v: float) -> float:
    return Y0 - min(max(v, 0), 1) * (Y0 - Y1)


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    gate = results["gate"]["hybrid (dense + BM25)"]
    tau, beta = gate["min_similarity"], gate["min_keyword"]
    points = results["per_question"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Refusal gate on the evaluation questions</title>',
        (
            f'<desc id="desc">Each evaluation question plotted by its best meaning score and best keyword '
            f"score. The gate accepts a question when meaning is at least {tau} or keywords at least {beta}. "
            "Answerable questions fall in the accepted area; short acronym questions are rescued by the "
            "keyword score; off-topic questions fall outside.</desc>"
        ),
        STYLE,
        f'<rect class="bg" width="{W}" height="{H}" rx="12"/>',
        '<text class="h" x="24" y="32">What the gate sees</text>',
        f'<text class="s" x="24" y="52">Each dot is one evaluation question. Shaded: answered (meaning ≥ {tau} or keywords ≥ {beta}).</text>',
        # accepted zone: right of tau (full height) + above beta (left part)
        f'<rect class="zone" x="{sx(tau):.1f}" y="{Y1}" width="{X1 - sx(tau):.1f}" height="{Y0 - Y1}"/>',
        f'<rect class="zone" x="{X0}" y="{Y1}" width="{sx(tau) - X0:.1f}" height="{sy(beta) - Y1:.1f}"/>',
    ]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        x = sx(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{Y1}" x2="{x:.1f}" y2="{Y0}"/>')
        parts.append(f'<text class="m" x="{x:.1f}" y="{Y0 + 18}" text-anchor="middle">{t:.1f}</text>')
    for t in (0.25, 0.5, 0.75, 1.0):
        y = sy(t)
        parts.append(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>')
        parts.append(f'<text class="m" x="{X0 - 8}" y="{y + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    parts += [
        f'<text class="m" x="{X0 - 8}" y="{Y0 + 4}" text-anchor="end">0</text>',
        f'<path class="axis" d="M{X0},{Y1} V{Y0} H{X1}"/>',
        f'<line class="thr" x1="{sx(tau):.1f}" y1="{Y1}" x2="{sx(tau):.1f}" y2="{Y0}"/>',
        f'<line class="thr" x1="{X0}" y1="{sy(beta):.1f}" x2="{sx(tau):.1f}" y2="{sy(beta):.1f}"/>',
        f'<text class="s" x="{(X0 + X1) / 2}" y="{Y0 + 40}" text-anchor="middle">best meaning score (cosine similarity)</text>',
        f'<text class="s" x="18" y="{(Y0 + Y1) / 2}" text-anchor="middle" transform="rotate(-90 18 {(Y0 + Y1) / 2})">best keyword score (BM25)</text>',
    ]
    order = {"unanswerable": 0, "out_of_scope": 1, "answerable": 2}  # answerable drawn on top
    cls = {"answerable": "ans", "out_of_scope": "oos", "unanswerable": "una"}
    for p in sorted(points, key=lambda p: order[p["type"]]):
        x, y = sx(p["best_similarity"]), sy(p["best_keyword"])
        parts.append(f'<circle class="{cls[p["type"]]}" cx="{x:.1f}" cy="{y:.1f}" r="5.5"><title>{p["question"]}</title></circle>')
        if p["id"] in LABELS:
            label, dx, dy, anchor = LABELS[p["id"]]
            parts.append(f'<text class="s" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">{label}</text>')
    # The short questions all score 1.00 on keywords: label them as a group above the plot.
    short = [p for p in points if p["style"] == "short"]
    if short:
        xs = [sx(p["best_similarity"]) for p in short]
        lo, hi, y = min(xs), max(xs), sy(1.0) - 14
        parts += [
            f'<path class="lead" fill="none" d="M{lo:.1f},{y + 6:.1f} V{y:.1f} H{hi:.1f} V{y + 6:.1f}"/>',
            f'<text class="s" x="{(lo + hi) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">short questions: “What is QLoRA?”, “What is HNSW?”…</text>',
        ]
    # legend, outside the plot
    lx, ly = X1 + 26, Y1 + 10
    parts += [
        f'<circle class="ans" cx="{lx}" cy="{ly}" r="5.5"/><text class="s" x="{lx + 12}" y="{ly + 4}">answerable</text>',
        f'<circle class="una" cx="{lx}" cy="{ly + 20}" r="5.5"/><text class="s" x="{lx + 12}" y="{ly + 24}">on topic, not answerable</text>',
        f'<circle class="oos" cx="{lx}" cy="{ly + 40}" r="5.5"/><text class="s" x="{lx + 12}" y="{ly + 44}">off-topic</text>',
        "</svg>",
    ]
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
