#!/usr/bin/env python3
# Render a dated, manuscript-ready Sankey of the conceptual Nanopath lineages and their mergers.
# Run with: uv run --with matplotlib python labless/plot_conceptual_sankey.py

import csv
import datetime as dt
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_CSV = ROOT / "labless" / "experiment_lineage.csv"
COLORS = {
    "DINO / KDE": "#4477AA", "I-JEPA": "#EE9933", "FINO": "#CC6677",
    "Dense readout": "#228833", "DINOv3": "#AA4499", "ViT-5": "#999933",
    "Data curation": "#66CCEE", "CAPI": "#882255", "MolCap": "#DDCC77",
    "SAM": "#44AA99", "SimCLR": "#888888", "Robust norm": "#332288",
    "DINO + JEPA": "#D89000", "JEPA + FINO": "#C44E52",
    "Mature merged trunk": "#117755", "Robust leader": "#1E3A5F",
    "SimCLR + post-train": "#4B4B4B",
}


def iso(value):
    parsed = dt.datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


rows = list(csv.DictReader(LINEAGE_CSV.open()))
by_title = {row["child_title"]: row for row in rows}
campaign_counts = Counter(row["campaign"] for row in rows)
search_windows = [
    ("2026-06-20", "2026-06-25"), ("2026-06-26", "2026-06-30"),
    ("2026-07-01", "2026-07-13"), ("2026-07-14", "2026-07-20"),
    ("2026-07-21", "2026-08-08"), ("2026-08-09", "2026-08-12"),
    ("2026-08-13", "2026-08-16"),
]
search_counts = [sum(row["campaign"].startswith("Cycle ") and start <= row["ended_at"][:10] <= end for row in rows) for start, end in search_windows]


def score(title):
    return float(by_title[title]["score"])


def when(title):
    return iso(by_title[title]["ended_at"])


def count_titles(fragment):
    return sum(fragment.lower() in row["child_title"].lower() for row in rows)


def best_titles(fragment):
    return max(float(row["score"]) for row in rows if fragment.lower() in row["child_title"].lower())


def node(node_id, date, y, label, lineage, kind="branch", count=1, run="", note="", label_side="top"):
    return {
        "id": node_id, "date": date if isinstance(date, dt.datetime) else iso(date),
        "y": y, "label": label, "lineage": lineage, "kind": kind, "count": count,
        "run": run, "score": score(run) if run else "", "note": note, "label_side": label_side,
    }


nodes = [
    node("dino", when("dinov2-s-kde"), .60, f"DINOv2 + KDE\n{score('dinov2-s-kde'):.4f}", "DINO / KDE", "origin", run="dinov2-s-kde"),
    node("lr", when("lr-and-curation"), .60, f"LR + tissue curation\n{score('lr-and-curation'):.4f}", "DINO / KDE", run="lr-and-curation", label_side="bottom"),
    node("jepa", when("jepa-dino"), .84, "I-JEPA\nnew objective", "I-JEPA", "origin", count=count_titles("jepa"), run="jepa-dino"),
    node("dino3", when("dinov3-vits-jepa"), .94, f"DINOv3 branch\n×{count_titles('dinov3')} · best {best_titles('dinov3'):.4f}", "DINOv3", "origin", count=count_titles("dinov3"), run="dinov3-vits-jepa"),
    node("vit5", when("vit5-vitb-kde"), .73, f"ViT-5 branch\n×{count_titles('vit5')} · best {best_titles('vit5'):.4f}", "ViT-5", "origin", count=count_titles("vit5"), run="vit5-vitb-kde", label_side="bottom"),
    node("jepa_valid", when("I-JEPA contig patch"), .66, f"Validated I-JEPA\n{score('I-JEPA contig patch'):.4f}", "DINO + JEPA", "merge", run="I-JEPA contig patch"),
    node("fino", "2026-06-11T18:00:00", .40, "FINO\nmolecular signal", "FINO", "origin", count=count_titles("fino"), label_side="bottom"),
    node("jf", when("jepa-fino-s2026"), .61, f"JEPA + FINO\n×{count_titles('jepa-fino')} · {score('jepa-fino-s2026'):.4f}", "JEPA + FINO", "merge", count=count_titles("jepa-fino"), run="jepa-fino-s2026", label_side="bottom"),
    node("readout", when("block-strided-214"), .34, "Dense multi-depth\nreadout", "Dense readout", "origin", run="block-strided-214", label_side="bottom"),
    node("bsc", when("block-strided-cls"), .58, f"Merged trunk\n{score('block-strided-cls'):.4f}", "Mature merged trunk", "merge", run="block-strided-cls"),
    node("pre", "2026-06-19T00:00:00", .58, f"Pre-cycle search\n×{campaign_counts['Pre-cycle search']}", "Mature merged trunk", count=campaign_counts["Pre-cycle search"], label_side="bottom"),
    node("search1", "2026-06-20T07:04:53", .58, f"Automated search\n×{search_counts[0]}", "Mature merged trunk", count=search_counts[0]),
    node("curation", when("curation-unbalanced"), .22, f"Data-curation branch\n×{count_titles('curat')} · best {best_titles('curat'):.4f}", "Data curation", "origin", count=count_titles("curat"), run="curation-unbalanced", label_side="bottom"),
    node("kde", when("bsc-s932-k10"), .75, f"KDE follow-up\n{score('bsc-s932-k10'):.4f} validated", "Mature merged trunk", "merge", run="bsc-s932-k10"),
    node("search2", "2026-06-26T20:01:12", .58, f"Automated search\n×{search_counts[1]}", "Mature merged trunk", count=search_counts[1], label_side="bottom"),
    node("capi", when("capi-ibot-clean"), .10, "CAPI + iBOT\nnew objective", "CAPI", "origin", run="capi-ibot-clean", label_side="bottom"),
    node("search3", "2026-07-01T07:59:39", .58, f"Automated search\n×{search_counts[2]}", "Mature merged trunk", count=search_counts[2]),
    node("capi_trials", "2026-07-15T02:51:00", .11, f"CAPI family\n×{count_titles('capi')} · best {best_titles('capi'):.4f}", "CAPI", count=count_titles("capi"), label_side="bottom"),
    node("molcap", "2026-07-11T12:00:00", .25, "MolCap\ntext signal", "MolCap", "origin", count=count_titles("molcap"), label_side="bottom"),
    node("molcap_merge", when("molcap-text-s7777"), .36, f"MolCap on trunk\n×{count_titles('molcap')} · {score('molcap-text-s7777'):.4f}", "Mature merged trunk", "merge", count=count_titles("molcap"), run="molcap-text-s7777", label_side="bottom"),
    node("search4", "2026-07-14T17:47:43", .58, f"Automated search\n×{search_counts[3]}", "Mature merged trunk", count=search_counts[3]),
    node("sam", when("sam-jepa-mol"), .04, f"SAM branch\n×{count_titles('sam-')} · best {best_titles('sam-'):.4f}", "SAM", "origin", count=count_titles("sam-"), run="sam-jepa-mol", label_side="bottom"),
    node("sam_tuned", when("sam-hp-fix"), .12, "SAM tuned", "SAM", run="sam-hp-fix", label_side="bottom"),
    node("lctx", when("lctx14-kde05-r3"), .76, f"Local context + KDE\n{score('lctx14-kde05-r3'):.4f}", "Mature merged trunk", "merge", run="lctx14-kde05-r3"),
    node("search5", "2026-07-21T13:48:18", .49, f"Automated search\n×{search_counts[4]}", "Mature merged trunk", count=search_counts[4], label_side="bottom"),
    node("lctx_valid", when("lctx14-kde05-s7375"), .76, f"Local-context validation\n{score('lctx14-kde05-s7375'):.4f}", "Mature merged trunk", run="lctx14-kde05-s7375"),
    node("search6", "2026-08-09T06:46:16", .58, f"Automated search\n×{search_counts[5]}", "Mature merged trunk", count=search_counts[5]),
    node("search7", "2026-08-13T00:14:38", .58, f"Automated search\n×{search_counts[6]}", "Mature merged trunk", count=search_counts[6]),
    node("simclr", "2026-08-16T12:00:00", .97, "SimCLR\nnew objective", "SimCLR", "origin", count=count_titles("simclr")),
    node("simclr_merge", when("simclr-t0810-repro"), .88, f"DINOv3 + SimCLR\n×2 · best {score('simclr-t0810-repro'):.4f}", "SimCLR + post-train", "merge", count=2, run="simclr-t0810-repro"),
    node("robust", when("robust-norm"), .30, "Robust norm\nnew post-train", "Robust norm", "origin", count=count_titles("robust-norm"), run="robust-norm", label_side="bottom"),
    node("leader", when("robust-norm-s9876"), .61, f"Current leader\n{score('robust-norm-s9876'):.4f}", "Robust leader", "merge", run="robust-norm-s9876"),
    node("simclr_post", when("simclr-posttrain"), .83, f"SimCLR + robust post-train\n{score('simclr-posttrain'):.4f}", "SimCLR + post-train", "merge", run="simclr-posttrain", label_side="bottom"),
]

# Weights are log-scaled experiment counts: large sweeps remain visible without overwhelming mergers.
edges = [
    ("dino", "lr", 2.5), ("lr", "jepa_valid", 3), ("jepa", "jepa_valid", 3),
    ("jepa_valid", "jf", 3.5), ("fino", "jf", 3.5), ("jf", "bsc", 4),
    ("readout", "bsc", 4), ("bsc", "pre", math.log2(campaign_counts["Pre-cycle search"] + 1)),
    ("pre", "search1", math.log2(search_counts[0] + 1)),
    ("bsc", "kde", 2.2), ("dino", "kde", 1.8),
    ("search1", "search2", math.log2(search_counts[1] + 1)),
    ("search2", "search3", math.log2(search_counts[2] + 1)),
    ("search3", "search4", math.log2(search_counts[3] + 1)),
    ("search4", "search5", math.log2(search_counts[4] + 1)),
    ("search4", "search6", math.log2(search_counts[5] + 1)),
    ("search6", "search7", math.log2(search_counts[6] + 1)),
    ("capi", "capi_trials", math.log2(count_titles("capi") + 1)),
    ("molcap", "molcap_merge", 1.5), ("kde", "molcap_merge", 1.5),
    ("sam", "sam_tuned", math.log2(count_titles("sam-") + 1)),
    ("kde", "lctx", 2.5), ("search4", "lctx", 2.5), ("lctx", "lctx_valid", 2.8),
    ("dino3", "simclr_merge", 1.6), ("simclr", "simclr_merge", 1.6),
    ("search7", "leader", 2.2), ("lctx_valid", "leader", 2.2), ("robust", "leader", 2.2),
    ("simclr_merge", "simclr_post", 1.6), ("leader", "simclr_post", 1.6),
]

by_id = {item["id"]: item for item in nodes}
incoming, outgoing = defaultdict(list), defaultdict(list)
for index, (source, target, value) in enumerate(edges):
    outgoing[source].append((index, target, value))
    incoming[target].append((index, source, value))

flow_scale, node_width = .009, .72
source_slots, target_slots = {}, {}
for item in nodes:
    outs = sorted(outgoing[item["id"]], key=lambda edge: by_id[edge[1]]["y"])
    ins = sorted(incoming[item["id"]], key=lambda edge: by_id[edge[1]]["y"])
    for adjacency, slots in ((outs, source_slots), (ins, target_slots)):
        total = sum(edge[2] for edge in adjacency)
        cursor = item["y"] - total * flow_scale / 2
        for index, _, value in adjacency:
            slots[index] = (cursor, cursor + value * flow_scale)
            cursor += value * flow_scale

fig, ax = plt.subplots(figsize=(25, 10.5), constrained_layout=False)
for index, (source_id, target_id, value) in enumerate(edges):
    source, target = by_id[source_id], by_id[target_id]
    x0 = mdates.date2num(source["date"]) + node_width / 2
    x1 = mdates.date2num(target["date"]) - node_width / 2
    sy0, sy1 = source_slots[index]
    ty0, ty1 = target_slots[index]
    bend = max((x1 - x0) * .42, .12)
    vertices = [(x0, sy0), (x0 + bend, sy0), (x1 - bend, ty0), (x1, ty0),
                (x1, ty1), (x1 - bend, ty1), (x0 + bend, sy1), (x0, sy1), (x0, sy0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=COLORS[source["lineage"]],
                           edgecolor=COLORS[source["lineage"]], lw=.45, alpha=.63, zorder=1))

for item in nodes:
    total_in = sum(edge[2] for edge in incoming[item["id"]])
    total_out = sum(edge[2] for edge in outgoing[item["id"]])
    height = max(.025, max(total_in, total_out) * flow_scale)
    x = mdates.date2num(item["date"])
    edgecolor, linewidth = ("#F2C14E", 2.8) if item["kind"] == "merge" and item["id"] == "leader" else ("#222222", 1.15)
    ax.add_patch(Rectangle((x - node_width / 2, item["y"] - height / 2), node_width, height,
                           facecolor=COLORS[item["lineage"]], edgecolor=edgecolor, lw=linewidth, zorder=3))
    above = item["label_side"] == "top"
    ax.text(x, item["y"] + (height / 2 + .011) * (1 if above else -1), item["label"],
            ha="center", va="bottom" if above else "top", fontsize=7.4, linespacing=1.1,
            color="#15202B", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .84, "pad": .55}, zorder=5)

ax.set_xlim(dt.datetime(2026, 6, 1, tzinfo=dt.UTC), dt.datetime(2026, 8, 24, tzinfo=dt.UTC))
ax.set_ylim(-.01, 1.03)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(axis="x", color="#D8DDE3", linewidth=.65, alpha=.8, zorder=0)
ax.tick_params(axis="x", labelsize=9, pad=8, length=0)
ax.set_yticks([])
for spine in ("left", "right", "top"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#AEB6BF")
ax.set_title("Nanopath experiment lineages: independent ideas merge into the dominant trajectory", loc="left", fontsize=18, weight="bold", pad=26)
ax.text(0, 1.012, "1,242 Labless runs, June 3–August 21, 2026  ·  x-axis is experiment date  ·  sweep widths use log₂(run count + 1)",
        transform=ax.transAxes, fontsize=10.5, color="#4B5563", va="bottom")

legend_order = ["DINO / KDE", "I-JEPA", "FINO", "Dense readout", "DINOv3", "ViT-5",
                "Data curation", "CAPI", "MolCap", "SAM", "SimCLR", "Robust norm",
                "DINO + JEPA", "JEPA + FINO", "Mature merged trunk", "Robust leader", "SimCLR + post-train"]
handles = [Patch(facecolor=COLORS[name], edgecolor="#333333", linewidth=.5, label=name) for name in legend_order]
ax.legend(handles=handles, title="Origin colors and new colors adopted after major merges", ncol=6,
          loc="upper center", bbox_to_anchor=(.5, -.10), frameon=False, fontsize=8.4, title_fontsize=9.5,
          handlelength=1.35, columnspacing=1.5)
fig.subplots_adjust(left=.025, right=.99, top=.88, bottom=.20)

out = ROOT / "imgs" / "labless_conceptual_lineage_sankey"
fig.savefig(out.with_suffix(".png"), dpi=240, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
plt.close(fig)

with (ROOT / "labless" / "conceptual_lineage_nodes.csv").open("w", newline="") as handle:
    fields = ["node_id", "date", "label", "kind", "lineage", "run_count", "representative_run", "score", "interpretation"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for item in nodes:
        writer.writerow(dict(zip(fields, [item["id"], item["date"].isoformat(), item["label"].replace("\n", " / "), item["kind"],
                                              item["lineage"], item["count"], item["run"], item["score"], item["note"]])))

with (ROOT / "labless" / "conceptual_lineage_edges.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["source", "target", "weight", "inherited_color", "interpretation"])
    writer.writeheader()
    for source, target, value in edges:
        writer.writerow({"source": source, "target": target, "weight": value,
                         "inherited_color": by_id[source]["lineage"],
                         "interpretation": "conceptual ingredient or development continuation"})

print(f"Wrote {out.with_suffix('.png')}")
print(f"Wrote {out.with_suffix('.pdf')}")
