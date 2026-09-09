#!/usr/bin/env python3
# Render a dated, source-evidence Sankey from the audited Labless run lineage.
# Run with: uv run --with matplotlib --with pandas python labless/plot_git_source_sankey.py

import csv
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
BIN_DAYS = 3
CAMPAIGN_ORDER = ["Early manual", "Numbered search", "Pre-cycle search", "Other manual", "Automated search", "Late manual"]
CAMPAIGN_SHORT = {"Early manual": "Early", "Numbered search": "Numbered", "Pre-cycle search": "Pre-search", "Other manual": "Manual", "Automated search": "Automated search", "Late manual": "Late"}
PHASE_COLOR = {
    "Foundation": "#2F6BFF", "Manual branches": "#00A087", "Automated search": "#E69F00",
}
METHOD_COLOR = {
    "Git/API source fork": "#0072B2", "Named base": "#F0A202", "Named relation": "#EF476F",
    "Campaign anchor": "#00A087", "Cited ingredient": "#9B5DE5",
}


def phase(campaign):
    if campaign in {"Early manual", "Numbered search", "Pre-cycle search"}:
        return "Foundation"
    if campaign in {"Other manual", "Late manual"}:
        return "Manual branches"
    return "Automated search"


def display_campaign(campaign):
    return "Automated search" if campaign.startswith("Cycle ") else campaign


def method_class(method, kind):
    if kind == "secondary":
        return "Cited ingredient"
    if method == "source_branch":
        return "Git/API source fork"
    if method == "campaign_baseline":
        return "Campaign anchor"
    if method == "named_base":
        return "Named base"
    return "Named relation"


def ribbon(ax, x0, y0, x1, y1, width, color):
    if x1 <= x0:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=-0.42", arrowstyle="-", color=color, lw=2.0 + 2.2 * width, alpha=0.67, zorder=1))
        return
    bend = max((x1 - x0) * 0.42, 0.8)
    vertices = [(x0, y0 - width), (x0 + bend, y0 - width), (x1 - bend, y1 - width), (x1, y1 - width),
                (x1, y1 + width), (x1 - bend, y1 + width), (x0 + bend, y0 + width), (x0, y0 + width), (x0, y0 - width)]
    codes = [MplPath.MOVETO, *([MplPath.CURVE4] * 3), MplPath.LINETO, *([MplPath.CURVE4] * 3), MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="white", lw=0.16, alpha=0.64, zorder=1))


def main():
    lineage = pd.read_csv(ROOT / "labless" / "experiment_lineage.csv", dtype=str)
    groups = pd.read_csv(ROOT / "labless" / "experiment_lineage_groups.csv", dtype=str)
    lineage["ended"] = pd.to_datetime(lineage["ended_at"], utc=True).dt.tz_convert(None)
    first_day = lineage["ended"].min().normalize()
    lineage["time_bin"] = ((lineage["ended"] - first_day).dt.days // BIN_DAYS).astype(int)
    run_to_group = groups.set_index("run_id")["group_id"].to_dict()
    members = groups.merge(lineage[["child_run_id", "ended", "time_bin", "campaign"]].rename(columns={"child_run_id": "run_id"}), on="run_id", validate="one_to_one")
    members["display_campaign"] = members["campaign"].map(display_campaign)
    group_data = members.groupby("group_id", as_index=False).agg(time_bin=("time_bin", "min"), campaign=("campaign", "first"), display_campaign=("display_campaign", "first"), run_count=("run_id", "size"), first_ended=("ended", "min"))
    group_data["node_id"] = group_data["time_bin"].astype(str) + "|" + group_data["display_campaign"]
    group_to_node = group_data.set_index("group_id")["node_id"].to_dict()
    nodes = group_data.groupby(["node_id", "time_bin", "display_campaign"], as_index=False).agg(run_count=("run_count", "sum"), group_count=("group_id", "size"), first_ended=("first_ended", "min"), raw_campaigns=("campaign", lambda values: "; ".join(sorted(set(values)))))
    nodes = nodes.rename(columns={"display_campaign": "campaign"})
    nodes["date"] = first_day + pd.to_timedelta(nodes["time_bin"] * BIN_DAYS, unit="D")

    raw_edges, primary_pairs = [], set()
    for row in lineage.itertuples():
        if pd.notna(row.primary_parent_id):
            raw_edges.append((row.primary_parent_id, row.child_run_id, "primary", row.relation_method))
            primary_pairs.add((row.primary_parent_id, row.child_run_id))
    for row in lineage.itertuples():
        if pd.notna(row.secondary_parent_ids):
            raw_edges.extend((parent.strip(), row.child_run_id, "secondary", "cited_influence") for parent in row.secondary_parent_ids.split(";") if parent.strip() and (parent.strip(), row.child_run_id) not in primary_pairs)
    edge_rows = []
    for parent, child, kind, method in raw_edges:
        source_group, target_group = run_to_group[parent], run_to_group[child]
        source, target = group_to_node[source_group], group_to_node[target_group]
        edge_rows.append({"source": source, "target": target, "kind": kind, "method": method, "method_class": method_class(method, kind), "internal": source == target})
    edge_frame = pd.DataFrame(edge_rows)
    links = edge_frame.groupby(["source", "target", "kind", "method", "method_class", "internal"], as_index=False).size().rename(columns={"size": "edge_count"})

    node_lookup = nodes.set_index("node_id").to_dict("index")
    audit = []
    for row in nodes.itertuples():
        internal = links[(links.source == row.node_id) & (links.target == row.node_id)].edge_count.sum()
        audit.append({"record_type": "node", "source_node": row.node_id, "target_node": "", "source_date": row.date.date(), "target_date": "", "source_campaign": row.campaign, "target_campaign": "", "source_raw_campaigns": row.raw_campaigns, "target_raw_campaigns": "", "kind": "", "method": "", "method_class": "", "internal": "", "run_count": row.run_count, "group_count": row.group_count, "edge_count": "", "internal_edge_count": int(internal)})
    for row in links.itertuples():
        source, target = node_lookup[row.source], node_lookup[row.target]
        audit.append({"record_type": "link", "source_node": row.source, "target_node": row.target, "source_date": source["date"].date(), "target_date": target["date"].date(), "source_campaign": source["campaign"], "target_campaign": target["campaign"], "source_raw_campaigns": source["raw_campaigns"], "target_raw_campaigns": target["raw_campaigns"], "kind": row.kind, "method": row.method, "method_class": row.method_class, "internal": row.internal, "run_count": "", "group_count": "", "edge_count": row.edge_count, "internal_edge_count": ""})
    audit_path = ROOT / "labless" / "git_source_sankey_audit.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit[0].keys())
        writer.writeheader(); writer.writerows(audit)

    lane = {name: index for index, name in enumerate(CAMPAIGN_ORDER)}
    x = {row.node_id: mdates.date2num(row.date) for row in nodes.itertuples()}
    y = {row.node_id: len(CAMPAIGN_ORDER) - 1 - lane[row.campaign] for row in nodes.itertuples()}
    heights = {row.node_id: 0.34 + 0.065 * math.sqrt(row.run_count) for row in nodes.itertuples()}
    offsets = {name: (index - 2) * 0.075 for index, name in enumerate(METHOD_COLOR)}

    fig, ax = plt.subplots(figsize=(23, 7.5))
    for index in range(len(CAMPAIGN_ORDER)):
        if index % 2 == 0:
            ax.axhspan(index - 0.48, index + 0.48, color="#F8FAFC", zorder=0)
    for row in links[~links.internal].sort_values("edge_count", ascending=False).itertuples():
        width = 0.025 + 0.026 * math.sqrt(row.edge_count)
        ribbon(ax, x[row.source] + 0.43, y[row.source] + offsets[row.method_class], x[row.target] - 0.43, y[row.target] + offsets[row.method_class], width, METHOD_COLOR[row.method_class])
    for row in links[links.internal].sort_values("edge_count", ascending=False).itertuples():
        width = 0.025 + 0.026 * math.sqrt(row.edge_count)
        ribbon(ax, x[row.source] + 0.40, y[row.source] + offsets[row.method_class], x[row.target] + 0.41, y[row.target] + offsets[row.method_class] + 0.06, width, METHOD_COLOR[row.method_class])
    for row in nodes.itertuples():
        height = heights[row.node_id]
        ax.add_patch(Rectangle((x[row.node_id] - 0.42, y[row.node_id] - height / 2), 0.84, height, facecolor=PHASE_COLOR[phase(row.campaign)], edgecolor="#172B4D", lw=0.75, zorder=3))
        if row.run_count > 1:
            ax.text(x[row.node_id], y[row.node_id], f"×{row.run_count}", color="white", ha="center", va="center", fontsize=5.8, fontweight="bold", zorder=4)

    fig.suptitle("Git/source ancestry of all Nanopath v1 experiments", x=0.08, y=0.965, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.925, f"{len(lineage):,} runs → {len(group_data):,} collapsed families → {len(nodes):,} dated campaign cohorts · all {len(raw_edges):,} recorded ancestry links", fontsize=9.5, color="#475569")
    ax.set_yticks(range(len(CAMPAIGN_ORDER)), [CAMPAIGN_SHORT[name] for name in reversed(CAMPAIGN_ORDER)])
    ax.set_ylim(-0.7, len(CAMPAIGN_ORDER) - 0.3)
    ax.set_xlim(mdates.date2num(first_day) - 1.2, mdates.date2num(lineage.ended.max().normalize()) + 4.0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="x", color="#CBD5E1", lw=0.65, alpha=0.8); ax.tick_params(axis="y", length=0, labelsize=8); ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xlabel("Experiment end date (3-day cohorts)", fontsize=9)
    link_legend = [Line2D([], [], color=color, lw=5, alpha=0.7, label=label) for label, color in METHOD_COLOR.items()]
    phase_legend = [Patch(facecolor=color, edgecolor="#172B4D", label=label) for label, color in PHASE_COLOR.items()]
    ax.legend(handles=[*phase_legend, *link_legend], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False, fontsize=8, columnspacing=1.5)
    fig.text(0.5, 0.018, "Node ×N = raw runs in a campaign/date cohort. Ribbon width = recorded run-level ancestry count; right-side loops are ancestry contained inside one cohort. Colors distinguish campaign phase and evidence type.", ha="center", fontsize=8, color="#475569")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.89, bottom=0.21)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "imgs" / f"labless_git_source_lineage_sankey.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    primary = sum(edge[2] == "primary" for edge in raw_edges)
    secondary = len(raw_edges) - primary
    assert len(lineage) == groups.run_id.nunique() == nodes.run_count.sum()
    assert len(group_data) == groups.group_id.nunique() == nodes.group_count.sum()
    assert primary == len(lineage) - 1 and links.edge_count.sum() == len(raw_edges)
    print({"runs": len(lineage), "groups": len(group_data), "cohorts": len(nodes), "primary_links": primary, "secondary_links": secondary, "internal_links": int(links[links.internal].edge_count.sum()), "audit": str(audit_path)})


if __name__ == "__main__":
    main()
