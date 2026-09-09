#!/usr/bin/env python3
# Fetch the complete public Labless ledger and turn it into manuscript-ready trend and lineage figures.
# Run with: uv run --with matplotlib --with networkx python labless/plot_experiments.py

import csv
import datetime as dt
import http.client
import json
import re
from pathlib import Path
from urllib.parse import urlencode, urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

API = "https://api.labless.dev/api/nano-projects/nanopath/experiment-log"
ROOT = Path(__file__).resolve().parents[1]
THEMES = {
    "Tuning / schedules": r"(^|[^a-z])(lr|warmup|beta|seed|reseed|dose|weight|fraction|schedule|temperature|temp|kde|mask[-_ ]?(ratio|scale|weight)|local[-_ ]?crop|global[-_ ]?crop|local[-_ ]?views?|crop[-_ ]?scale|batch[-_ ]?size|drop[-_ ]?path|width)([^a-z]|$)",
    "Data / augmentation": r"tissue|curat|sampl|dedup|stain|\bhed\b|hed[-_]|\bhsv\b|augment|color[-_ ]?jitter|grayscale|blur|magnification|\bmpp\b|tile[-_ ]?filter|frequency|rgb|local[-_ ]?crop|global[-_ ]?crop|sizemix",
    "Objectives / losses": r"jepa|simclr|vicreg|koleo|gram[-_ ]?anchor|coding[-_ ]?rate|\bibot\b|focal|contrastive|invariance|entropy|sinkhorn|\bmae\b|barlow|uniformity|sigreg|pcgrad|gradnorm|capi|genediv|new[-_ ]?loss",
    "Molecular / text": r"fino|molecular|expr512|\bfga\b|subtype|genom|\bcnv\b|caption|molcap|text[-_ ]?encoder|patient[-_ ]?metadata|\bomics\b|gene[-_ ]",
    "Architecture / backbone": r"dinov3|vit[-_ ]?5|\bvitb\b|mamba|monarch|backbone|register|\bqkv\b|swiglu|architect|specializ|layernorm|layerscale|headpool|predictor[-_ ]?(depth|width|block)",
    "Readout / post-train": r"readout|probe_features|block[-_ ]?strided|multi[-_ ]?depth|cls[-_ ]?concat|densefuse|densif|\bjbu\b|pooling|\bpca\b|shrink|robust[-_ ]?norm|post[-_ ]?train|\btta\b|suppression|readout[-_ ]?tap|cls[-_ ]?tap",
}
PALETTE = {
    "Tuning / schedules": "#E69F00",
    "Data / augmentation": "#56B4E9",
    "Objectives / losses": "#D55E00",
    "Molecular / text": "#CC79A7",
    "Architecture / backbone": "#F0E442",
    "Readout / post-train": "#009E73",
    "Other / controls": "#A7A9AC",
}
METRICS = ["linear", "knn", "few_shot", "seg_jaccard", "progression_auc", "mutation_auc", "survival_cindex", "robustness"]
METRIC_LABELS = ["Linear", "kNN", "16-shot", "Seg.", "Progress.", "Mutation", "Survival", "Robust."]


def fetch_runs():
    parsed, params, runs = urlparse(API), {"limit": 100}, []
    while True:
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=45)
        conn.request("GET", f"{parsed.path}?{urlencode(params)}", headers={"Accept": "application/json", "User-Agent": "nanopath-paper-analysis/1.0"})
        response = conn.getresponse()
        raw = response.read().decode()
        conn.close()
        if response.status >= 400:
            raise RuntimeError(f"Labless returned HTTP {response.status}: {raw[:300]}")
        page = json.loads(raw)
        runs.extend(page["runs"])
        if not page.get("next_after_updated_at"):
            break
        params = {"limit": 100, "after_updated_at": page["next_after_updated_at"], "after_run_id": page["next_after_run_id"]}
    if len(runs) != len({run["run_id"] for run in runs}):
        raise RuntimeError("Labless pagination returned duplicate run ids")
    return sorted(runs, key=lambda run: (run["submitted_at"], run["run_id"]))


def run_text(run):
    return " ".join(str(run.get(key) or "") for key in ("title", "summary", "changes")).lower()


def run_themes(run):
    text = run_text(run)
    matched = [name for name, pattern in THEMES.items() if re.search(pattern, text)]
    return matched or ["Other / controls"]


def write_index(runs):
    ids = {run["run_id"] for run in runs}
    path = ROOT / "labless" / "experiment_index.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "submitted_at", "title", "score", "validation", "contributor", "themes", "explicit_parent_ids", "comparison_base_id", "api_url"])
        writer.writeheader()
        for run in runs:
            refs = [ref for ref in dict.fromkeys(re.findall(r"run_sub_[0-9a-f]{10}", run_text(run))) if ref in ids and ref != run["run_id"]]
            writer.writerow({
                "run_id": run["run_id"], "submitted_at": run["submitted_at"], "title": run["title"],
                "score": run["metric_value"], "validation": run["validation"], "contributor": run["contributor"],
                "themes": "; ".join(run_themes(run)), "explicit_parent_ids": "; ".join(refs),
                "comparison_base_id": run.get("repo", {}).get("main_context", {}).get("run_id", ""), "api_url": run["api_url"],
            })
    return path


def plot_summary(runs):
    by_title = {run["title"]: run for run in runs}
    dates = [dt.datetime.fromisoformat(run["submitted_at"]) for run in runs]
    scores = np.asarray([run["metric_value"] for run in runs])
    validated = [run for run in runs if run["validation"] != "unvalidated"]
    discoveries = [
        ("Tissue + LR", "tissue-curate", "lr-and-curation"),
        ("I-JEPA", "jepa-mask10", "I-JEPA contig patch"),
        ("JEPA + FINO", "jf-hed07", "jepa-fino-s2026"),
        ("Strided readout", "block-strided-214", "block-strided-cls"),
        ("KDE 0.05", "bsc-s7777-k10", "bsc-s932-k10"),
        ("Local context", "lctx14-kde05-r3", "lctx14-kde05-s7375"),
        ("Robust-norm family", "robust-norm", "robust-norm-s9876"),
    ]
    milestones = ["dinov2-s-kde", "lr-and-curation", "I-JEPA contig patch", "jepa-fino-s2026", "block-strided-cls", "robust-norm-s9876"]
    milestone_labels = ["DINO+KDE", "Tissue + LR", "I-JEPA", "JEPA + FINO", "Strided readout", "Robust leader"]

    fig = plt.figure(figsize=(12.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12), width_ratios=(0.92, 1.45))
    ax_a, ax_b, ax_c, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    ax_a.scatter(dates, scores, s=11, color="#7F8C8D", alpha=0.27, linewidths=0, rasterized=True, label=f"All runs (n={len(runs):,})")
    order = np.argsort(dates)
    ax_a.step(np.asarray(dates)[order], np.maximum.accumulate(scores[order]), where="post", color="#D55E00", lw=1.8, label="Discovery envelope")
    vdates = [dt.datetime.fromisoformat(run["submitted_at"]) for run in validated]
    vscores = np.asarray([run["metric_value"] for run in validated])
    ax_a.step(vdates, np.maximum.accumulate(vscores), where="post", color="#0072B2", lw=2.4, label="Validated envelope")
    ax_a.scatter(vdates, vscores, marker="D", s=40, color="#0072B2", edgecolor="white", linewidth=0.6, zorder=4)
    for title, label, offset in [("dinov2-s-kde", "DINO+KDE", (5, -23)), ("I-JEPA contig patch", "I-JEPA", (6, -24)), ("block-strided-cls", "Strided readout", (5, 7)), ("robust-norm-s9876", "Robust leader", (-72, 7))]:
        run = by_title[title]
        when = dt.datetime.fromisoformat(run["submitted_at"])
        ax_a.annotate(f"{label}\n{run['metric_value']:.4f}", (when, run["metric_value"]), xytext=offset, textcoords="offset points", fontsize=7, color="#17324D")
    ax_a.set(ylabel="Mean probe score", ylim=(0.44, 0.683), title="Public experiment ledger and validated progress")
    ax_a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2)); ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_a.legend(loc="lower right", frameon=False, fontsize=7)

    themed = {name: [run["metric_value"] for run in runs if name in run_themes(run)] for name in [*THEMES, "Other / controls"]}
    names = list(themed)
    box = ax_b.boxplot([themed[name] for name in names], vert=False, patch_artist=True, showfliers=False, widths=0.62, medianprops={"color": "white", "lw": 1.3})
    for patch, name in zip(box["boxes"], names):
        patch.set(facecolor=PALETTE[name], edgecolor="#3D4852", alpha=0.82, linewidth=0.6)
    rng = np.random.default_rng(7)
    for y, name in enumerate(names, 1):
        vals = themed[name]
        ax_b.scatter(vals, y + rng.normal(0, 0.055, len(vals)), s=5, color="#263238", alpha=0.12, linewidths=0, rasterized=True)
        ax_b.text(0.681, y, f"n={len(vals)}", va="center", ha="left", fontsize=7)
    ax_b.set(xlim=(0.44, 0.704), yticks=range(1, len(names) + 1), yticklabels=names, xlabel="Mean probe score", title="Experiment themes and performance")
    ax_b.text(0.0, -0.19, "Keyword-derived, non-exclusive themes; boxes show IQR and median.", transform=ax_b.transAxes, fontsize=7, color="#586069")

    gaps = []
    for y, (label, discovery, validation) in enumerate(discoveries):
        x0, x1 = by_title[discovery]["metric_value"], by_title[validation]["metric_value"]
        gaps.append(x1 - x0)
        ax_c.plot([x0, x1], [y, y], color="#A7A9AC", lw=1.5, zorder=1)
        ax_c.scatter(x0, y, color="#D55E00", s=37, zorder=2)
        ax_c.scatter(x1, y, color="#0072B2", marker="D", s=37, zorder=2)
        ax_c.text(0.679, y, f"{x1-x0:+.4f}", va="center", ha="right", fontsize=7, color="#17324D")
    ax_c.axvline(by_title["block-strided-cls"]["metric_value"], color="#B8C2CC", lw=0.8, ls=":")
    ax_c.set(xlim=(0.625, 0.681), yticks=range(len(discoveries)), yticklabels=[x[0] for x in discoveries], xlabel="Mean probe score", title=f"Discovery-to-validation gaps (median {np.median(gaps):+.4f})")
    ax_c.invert_yaxis()
    ax_c.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#D55E00", label="Discovery"), Line2D([], [], marker="D", color="none", markerfacecolor="#0072B2", label="Validation")], loc="lower left", frameon=False, fontsize=7)

    values = np.asarray([[by_title[name]["metrics"][metric] for metric in METRICS] for name in milestones])
    delta = values - values[0]
    image = ax_d.imshow(delta, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.065, vcenter=0, vmax=0.065))
    for row in range(delta.shape[0]):
        for col in range(delta.shape[1]):
            ax_d.text(col, row, f"{delta[row, col]:+.3f}", ha="center", va="center", fontsize=7, color="white" if abs(delta[row, col]) > 0.032 else "#17202A")
    ax_d.set(xticks=range(len(METRIC_LABELS)), xticklabels=METRIC_LABELS, yticks=range(len(milestone_labels)), yticklabels=milestone_labels, title="Task-wise change from the initial DINO+KDE run")
    ax_d.tick_params(axis="x", rotation=35)
    fig.colorbar(image, ax=ax_d, shrink=0.78, pad=0.02, label="Absolute score change")

    fig.suptitle("Nanopath development: broad search, compositional gains, and a validation gap", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.015, f"Labless public API snapshot {dt.datetime.now(dt.UTC):%Y-%m-%d}; {len(runs):,} full-tier records, {len(validated)} validated/leader records. Theme assignments are descriptive and validation uses a changed seed.", ha="center", fontsize=7, color="#586069")
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E5E7EB", lw=0.6, zorder=0)
    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "imgs" / f"labless_experiment_trends.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_breadth(runs):
    order = ["Tuning / schedules", "Objectives / losses", "Data / augmentation", "Molecular / text", "Architecture / backbone", "Readout / post-train", "Other / controls"]
    labels = ["Hyperparameters & schedules", "Objectives & losses", "Data & augmentation", "Molecular & text supervision", "Architecture & backbone", "Readout & post-training", "Other & controls"]
    dates = [dt.datetime.fromisoformat(run["submitted_at"]) for run in runs]
    first_monday = min(dates).date() - dt.timedelta(days=min(dates).weekday())
    week = np.asarray([(date.date() - first_monday).days // 7 for date in dates])
    weeks = np.arange(week.max() + 1)
    counts = np.zeros((len(order), len(weeks)), dtype=int)
    for run, column in zip(runs, week):
        for theme in run_themes(run):
            counts[order.index(theme), column] += 1
    totals, weekly = counts.sum(axis=1), np.bincount(week, minlength=len(weeks))

    fig = plt.figure(figsize=(17, 8.3))
    gs = fig.add_gridspec(2, 2, height_ratios=(4.3, 1.15), width_ratios=(4.9, 1.45), hspace=0.30, wspace=0.06)
    ax_heat, ax_total, ax_week = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    peak = counts.max()
    ax_heat.set_facecolor("#F4F6F8")
    for row, theme in enumerate(order):
        for column, count in enumerate(counts[row]):
            if not count:
                continue
            strength = (count / peak) ** 0.55
            ax_heat.add_patch(Rectangle((column - 0.47, row - 0.39), 0.94, 0.78, facecolor=PALETTE[theme], edgecolor="white", lw=0.7, alpha=0.16 + 0.84 * strength))
            ink = "white" if strength > 0.62 and theme in {"Objectives / losses", "Readout / post-train"} else "#17202A"
            ax_heat.text(column, row, str(count), ha="center", va="center", fontsize=7.2, fontweight="bold" if count >= 50 else "normal", color=ink)
    ticks = weeks[::2]
    tick_labels = [(first_monday + dt.timedelta(days=int(7 * value))).strftime("%b %d") for value in ticks]
    ax_heat.set(xlim=(-0.52, weeks[-1] + 0.52), ylim=(len(order) - 0.5, -0.5), yticks=range(len(order)), yticklabels=labels, xticks=ticks, xticklabels=[], title="Weekly experiment density by theme")
    ax_heat.tick_params(axis="y", length=0, pad=8)
    for label, theme in zip(ax_heat.get_yticklabels(), order):
        label.set_color(PALETTE[theme] if theme != "Architecture / backbone" else "#9A8500")
        label.set_fontweight("bold")
    for x in np.arange(-0.5, len(weeks), 1):
        ax_heat.axvline(x, color="white", lw=0.7, zorder=0)

    bars = ax_total.barh(range(len(order)), totals, color=[PALETTE[theme] for theme in order], height=0.62, edgecolor="white", linewidth=0.7)
    ax_total.set(ylim=(len(order) - 0.5, -0.5), yticks=[], xlabel="Runs", title="Runs touching each theme")
    ax_total.set_xlim(0, totals.max() * 1.34)
    for bar, count in zip(bars, totals):
        ax_total.text(count + totals.max() * 0.025, bar.get_y() + bar.get_height() / 2, f"{count:,}  ({count / len(runs):.0%})", va="center", fontsize=8, color="#17202A")

    ax_week.bar(weeks, weekly, width=0.82, color="#8292A2", edgecolor="white", linewidth=0.5, label="Runs submitted per week")
    ax_cumulative = ax_week.twinx()
    ax_cumulative.plot(weeks, np.cumsum(weekly), color="#17324D", lw=2.4, marker="o", ms=3.5, label="Cumulative runs")
    ax_cumulative.text(weeks[-1] + 0.18, len(runs), f"{len(runs):,}", va="center", ha="left", color="#17324D", fontsize=9, fontweight="bold")
    ax_week.set(xlim=(-0.52, weeks[-1] + 0.65), xticks=ticks, xticklabels=tick_labels, ylabel="Runs / week", xlabel="Submission week", title="All v1 experiment activity")
    ax_cumulative.set(ylabel="Cumulative runs", ylim=(0, len(runs) * 1.12))
    ax_week.legend(handles=[Patch(facecolor="#8292A2", label="Runs submitted per week"), Line2D([], [], color="#17324D", lw=2.4, marker="o", label="Cumulative runs")], loc="upper left", frameon=False, ncol=2)

    fig.suptitle(f"Nanopath v1: {len(runs):,} experiments across the full training stack", x=0.055, y=0.985, ha="left", fontsize=18, fontweight="bold", color="#17202A")
    fig.text(0.055, 0.936, f"{totals.sum():,} non-exclusive theme assignments from Labless titles and notes — a run can test architecture, loss, data, and hyperparameters together.", ha="left", fontsize=9, color="#586069")
    fig.text(0.055, 0.012, f"Labless public API snapshot {dt.datetime.now(dt.UTC):%Y-%m-%d}. Counts are descriptive keyword classifications, not mutually exclusive causal attributions.", fontsize=7.5, color="#586069")
    for ax, label in ((ax_heat, "A"), (ax_total, "B"), (ax_week, "C")):
        ax.text(-0.035 if ax is ax_heat else -0.06, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is ax_week else "x", color="#E5E7EB", lw=0.6, zorder=0)
    ax_cumulative.spines["top"].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "imgs" / f"labless_experiment_breadth.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lineage(runs):
    by_title = {run["title"]: run for run in runs}
    nodes = {
        "dinov2-s-kde": ((0, 0.4), "DINO+KDE", "Other / controls"),
        "tissue-curate": ((1, 2.0), "Tissue curate", "Data / augmentation"),
        "lr-and-curation": ((2, 2.0), "LR + curation", "Tuning / schedules"),
        "gentleaug-l112": ((1, 0.9), "Gentle aug", "Data / augmentation"),
        "jepa-mask10": ((2, 0.9), "I-JEPA discovery", "Objectives / losses"),
        "I-JEPA contig patch": ((3, 0.9), "I-JEPA validation", "Objectives / losses"),
        "jf-hed07": ((4, 0.9), "JEPA+FINO discovery", "Molecular / text"),
        "jepa-fino-s2026": ((5, 0.9), "JEPA+FINO validation", "Molecular / text"),
        "block-strided-214": ((5, -0.25), "Strided readout discovery", "Readout / post-train"),
        "block-strided-cls": ((6, 0.9), "Strided readout validation", "Readout / post-train"),
        "bsc-s7777-k10": ((7, 2.35), "KDE 0.05 discovery", "Tuning / schedules"),
        "bsc-s932-k10": ((8, 2.35), "KDE 0.05 validation", "Tuning / schedules"),
        "lctx14-kde05-r3": ((7, 1.55), "Local-context discovery", "Data / augmentation"),
        "lctx14-kde05-s7375": ((8, 1.55), "Local-context validation", "Data / augmentation"),
        "jepa-register-005": ((6, -0.7), "JEPA/register sweep", "Architecture / backbone"),
        "dinov3-vits-jepa": ((2, -1.55), "DINOv3+JEPA", "Architecture / backbone"),
        "vit5-vitb-kde": ((1, -1.55), "ViT-5-B+KDE", "Architecture / backbone"),
        "simclr-t0810-repro": ((7, -1.55), "DINOv3→SimCLR", "Objectives / losses"),
        "robust-norm": ((7, -0.35), "Robust-norm discovery", "Readout / post-train"),
        "rot180-tta": ((8, -0.7), "Rotation TTA", "Readout / post-train"),
        "robust-norm-s160": ((8, -0.05), "Local size 160", "Tuning / schedules"),
        "robust-norm-s9876": ((9, 0.9), "Robust validated leader", "Readout / post-train"),
        "simclr-posttrain": ((9, -1.55), "SimCLR + robust post", "Readout / post-train"),
    }
    explicit = [
        ("tissue-curate", "lr-and-curation"), ("jepa-mask10", "I-JEPA contig patch"),
        ("jf-hed07", "jepa-fino-s2026"), ("jepa-fino-s2026", "block-strided-cls"),
        ("block-strided-214", "block-strided-cls"), ("block-strided-cls", "lctx14-kde05-r3"),
        ("lctx14-kde05-r3", "lctx14-kde05-s7375"), ("bsc-s7777-k10", "bsc-s932-k10"),
        ("robust-norm", "rot180-tta"), ("robust-norm", "robust-norm-s160"),
        ("simclr-t0810-repro", "simclr-posttrain"), ("robust-norm-s9876", "simclr-posttrain"),
    ]
    inferred = [
        ("dinov2-s-kde", "tissue-curate"), ("dinov2-s-kde", "gentleaug-l112"),
        ("gentleaug-l112", "jepa-mask10"), ("I-JEPA contig patch", "jf-hed07"),
        ("dinov2-s-kde", "vit5-vitb-kde"), ("dinov2-s-kde", "dinov3-vits-jepa"),
        ("dinov3-vits-jepa", "simclr-t0810-repro"), ("block-strided-cls", "bsc-s7777-k10"),
        ("I-JEPA contig patch", "jepa-register-005"), ("block-strided-cls", "robust-norm"),
        ("block-strided-cls", "robust-norm-s9876"), ("lctx14-kde05-s7375", "robust-norm-s9876"),
        ("robust-norm", "robust-norm-s9876"),
    ]
    graph, pos = nx.DiGraph(), {name: spec[0] for name, spec in nodes.items()}
    graph.add_nodes_from(nodes); graph.add_edges_from(explicit + inferred)
    fig, ax = plt.subplots(figsize=(15.2, 6.5))
    fig.subplots_adjust(bottom=0.20, top=0.90, left=0.03, right=0.99)
    nx.draw_networkx_edges(graph, pos, edgelist=explicit, ax=ax, edge_color="#59636E", width=1.3, arrowsize=13, node_size=1800, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(graph, pos, edgelist=inferred, ax=ax, edge_color="#9AA5B1", width=1.0, style="dashed", arrowsize=12, node_size=1800, connectionstyle="arc3,rad=-0.05")
    for name, (xy, label, theme) in nodes.items():
        run, validation = by_title[name], by_title[name]["validation"]
        edge = "#B8860B" if validation == "leader" else "#17324D" if validation == "validated" else "#667085"
        width = 2.4 if validation in {"validated", "leader"} else 0.9
        status = "  LEADER" if validation == "leader" else "  VALIDATED" if validation == "validated" else ""
        ax.text(*xy, f"{label}\n{run['metric_value']:.4f}{status}", ha="center", va="center", fontsize=6.7,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": PALETTE[theme], "edgecolor": edge, "linewidth": width, "alpha": 0.96}, zorder=3)
    ax.set(xlim=(-0.6, 9.65), ylim=(-2.05, 2.75), title="Selected Nanopath development lineage (1,242-run ledger collapsed to landmark branches)")
    ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color="#59636E", lw=1.4, label="Run notes cite parent/base"),
        Line2D([], [], color="#9AA5B1", lw=1.1, ls="--", label="Lineage inferred from notes/source patch"),
        Patch(facecolor="white", edgecolor="#17324D", linewidth=2.2, label="Validated"),
        Patch(facecolor="white", edgecolor="#B8860B", linewidth=2.4, label="Current leader"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=7)
    fig.text(0.5, 0.025, "Scores are mean_probe_score. Multiple incoming arrows denote a compositional recipe, not a controlled attribution of gain.", ha="center", fontsize=7, color="#586069")
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "imgs" / f"labless_run_lineage.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    runs = fetch_runs()
    index = write_index(runs)
    plot_summary(runs)
    plot_breadth(runs)
    plot_lineage(runs)
    print(json.dumps({"runs": len(runs), "validated_or_leader": sum(run["validation"] != "unvalidated" for run in runs), "index": str(index), "figures": [str(ROOT / "imgs" / name) for name in ("labless_experiment_trends.pdf", "labless_experiment_trends.png", "labless_experiment_breadth.pdf", "labless_experiment_breadth.png", "labless_run_lineage.pdf", "labless_run_lineage.png")]}, indent=2))


if __name__ == "__main__":
    main()
