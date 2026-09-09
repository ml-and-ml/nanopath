#!/usr/bin/env python3
# Reconstruct every Labless run's evidence-ranked experimental ancestry and render an auditable flowchart.
# Run with: uv run --with matplotlib --with networkx python labless/plot_full_lineage.py

import concurrent.futures
import csv
import datetime as dt
import http.client
import json
import math
import re
import subprocess
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode, urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_experiments import API, PALETTE, ROOT, run_themes

SOURCE_CACHE = Path("/tmp/nanopath_labless_source_signatures.json")
COLLAPSE_PATTERNS = [
    ("replication/control", r"baseline|reseed|repro|replic|revalid|control|anchor"),
    ("local crops", r"local.?crop|local.?size|local128|lcc|localsizemix|sizemix"),
    ("global crops", r"global.?crop|gcs"),
    ("masking/JEPA/iBOT", r"mask|jepa|ibot|predictor"),
    ("anti-collapse", r"sigreg|anticollapse|koleo|vicreg|coding.?rate|gram.?anchor|barlow|genediv|sinkhorn"),
    ("KDE/uniformity", r"kde|uniform|isotrop"),
    ("teacher/EMA/temp", r"teacher|ema|temp|momentum"),
    ("LR/optimizer/grad", r"(?:^|-)lr(?:-|$)|optim|pcgrad|gradnorm|weight.?decay|adam|lawa"),
    ("warmup/schedule", r"warmup|schedule|curriculum|ramp|decay"),
    ("metadata/FINO", r"metadata|fino|subtype|expr|fga|gene|molecular|patient"),
    ("sampling/curation", r"sampling|sampler|curat|balance|tissue|dedup"),
    ("stain/color aug", r"stain|hed|hsv|color|jitter|grayscale|rgb"),
    ("geometric aug", r"geometric|rotation|flip|d4|ect|crop.?distort"),
    ("architecture", r"backbone|register|dinov3|vit5|mamba|monarch|width|depth|layer|qkv|swiglu"),
    ("readout/post-train", r"readout|probe|pca|norm|suppression|tta|block.?strided|dense.?fus|jbu"),
    ("contrastive", r"simclr|contrastive|ntxent"),
]


def fetch_runs():
    parsed, params, runs = urlparse(API), {"limit": 100}, []
    while True:
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=45)
        conn.request("GET", f"{parsed.path}?{urlencode(params)}", headers={"Accept": "application/json", "User-Agent": "nanopath-paper-lineage/1.0"})
        response, raw = conn.getresponse(), None
        raw = response.read().decode()
        conn.close()
        if response.status >= 400:
            raise RuntimeError(f"Labless returned HTTP {response.status}: {raw[:300]}")
        page = json.loads(raw)
        runs.extend(page["runs"])
        if not page.get("next_after_updated_at"):
            break
        params = {"limit": 100, "after_updated_at": page["next_after_updated_at"], "after_run_id": page["next_after_run_id"]}
    return sorted(runs, key=lambda run: (run["submitted_at"], run["run_id"]))


def fetch_source_signatures(runs):
    cached = json.loads(SOURCE_CACHE.read_text()) if SOURCE_CACHE.exists() else {}

    def fetch(run):
        request = urllib.request.Request(f"https://api.labless.dev/api/runs/{run['run_id']}/source", headers={"Accept": "application/json", "User-Agent": "nanopath-paper-lineage/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            source = json.load(response)
        files = {item["path"]: item["sha256"] for item in source["files"] if item.get("present")}
        return run["run_id"], {
            "source": source.get("source", ""), "git_commit": source.get("git_commit", ""),
            "core": {path: digest for path, digest in files.items() if not path.startswith("configs/")},
            "configs": {path: digest for path, digest in files.items() if path.startswith("configs/")},
        }

    missing = [run for run in runs if run["run_id"] not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for run_id, signature in pool.map(fetch, missing):
            cached[run_id] = signature
    SOURCE_CACHE.write_text(json.dumps(cached, sort_keys=True))
    return cached


def text(run):
    return " ".join(str(run.get(key) or "") for key in ("title", "summary", "changes")).lower()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def aliases(run):
    names = {run["title"]}
    for match in re.findall(r"full_run_name\s*=\s*([^|,;\n]+)", text(run), re.IGNORECASE):
        names.add(match.strip())
    for match in re.findall(r"\brun_id\s+([a-z0-9_.+-]+)", text(run), re.IGNORECASE):
        names.add(match.strip())
    return {normalize(name) for name in names if name}


def cycle(run):
    lead = f"{run['title']} {str(run.get('summary') or '')[:160]}".lower()
    match = re.search(r"(?:\bcycle[- ]?|\bc)(\d+)(?:\b|[-_])", lead)
    return int(match.group(1)) if match else None


def run_type(run):
    match = re.search(r"\brun_type\s*[=:]\s*([\w-]+)", text(run))
    return match.group(1) if match else ""


def campaign(run):
    number = cycle(run)
    if number:
        return f"Cycle {number}"
    if re.match(r"exp[_-]\d+", run["title"].lower()):
        return "Numbered search"
    day = run["ended_at"][:10]
    if day <= "2026-06-17":
        return "Early manual"
    if day < "2026-06-21":
        return "Pre-cycle search"
    if day >= "2026-08-17":
        return "Late manual"
    return "Other manual"


def collapse_family(run):
    if run["validation"] != "unvalidated":
        return f"run:{run['run_id']}"
    match = re.search(r"full_run_name\s*=\s*([a-z0-9_.+-]+)", text(run))
    name = normalize(match.group(1) if match else run["title"])
    tags = [label for label, pattern in COLLAPSE_PATTERNS if re.search(pattern, name)]
    if tags:
        return " + ".join(tags)
    if not cycle(run):
        return name
    name = re.sub(r"-t\d+(?:-.*)?$", "", name)
    name = re.sub(r"-(?:trial|retry|repro|r)\d+(?:-.*)?$", "", name)
    return re.sub(r"-[0-9a-f]{6}$", "", name)


def reconstruct(runs, signatures):
    by_id, position = {run["run_id"]: run for run in runs}, {run["run_id"]: i for i, run in enumerate(runs)}
    alias_index = defaultdict(list)
    for run in runs:
        for name in aliases(run):
            alias_index[name].append(run["run_id"])
    anchors = [run for run in runs if run_type(run) in {"baseline", "reseed_baseline", "anchor"}]

    def earlier(run, run_id):
        return run_id in position and position[run_id] < position[run["run_id"]]

    def resolve(run, name):
        key = normalize(name).removesuffix("-rebuild")
        choices = [run_id for run_id in alias_index.get(key, []) if earlier(run, run_id)]
        if choices:
            return choices[-1]
        match = re.match(r"c(\d+)-base", key)
        eligible = [anchor for anchor in anchors if earlier(run, anchor["run_id"]) and (cycle(anchor) or 0) < int(match.group(1))] if match else []
        return eligible[-1]["run_id"] if eligible else None

    def resolve_exact(run, name):
        choices = [run_id for run_id in alias_index.get(normalize(name), []) if earlier(run, run_id)]
        return choices[-1] if choices else None

    records, secondary_edges = [], []
    for run in runs:
        body, primary, method, confidence, evidence = text(run), None, "root", "n/a", "earliest public run"
        direct = [ref for ref in dict.fromkeys(re.findall(r"run_sub_[0-9a-f]{10}", body)) if ref != run["run_id"] and earlier(run, ref)]
        if direct:
            primary, method, confidence, evidence = direct[0], "explicit_run_id", "high", f"notes cite {direct[0]}"

        base_names = list(dict.fromkeys(re.findall(r"\b(?:paired_baseline|(?<!paired_)base)\s*=\s*([a-z0-9_.+-]+)", body)))
        for base_name in base_names:
            if normalize(base_name) in aliases(run):
                continue
            candidate = resolve(run, base_name)
            exact_base = any(earlier(run, run_id) for run_id in alias_index.get(normalize(base_name).removesuffix("-rebuild"), []))
            if base_name == "baseline":
                same_cycle = [anchor for anchor in anchors if earlier(run, anchor["run_id"]) and cycle(anchor) == cycle(run)]
                eligible = same_cycle or [anchor for anchor in anchors if earlier(run, anchor["run_id"])]
                candidate = eligible[-1]["run_id"] if eligible else None
                base_method, base_confidence = "campaign_baseline", "medium"
            elif exact_base:
                base_method, base_confidence = "named_base", "high"
            else:
                base_method, base_confidence = "campaign_baseline", "medium"
            if candidate and not primary:
                primary, method, confidence, evidence = candidate, base_method, base_confidence, f"notes specify base={base_name}" + ("; linked to previous available anchor" if not exact_base and base_name != "baseline" else "")
                break

        if not primary and run_type(run) in {"reseed_baseline", "baseline", "anchor"}:
            previous_anchors = [anchor for anchor in anchors if earlier(run, anchor["run_id"])]
            if previous_anchors:
                primary, method, confidence = previous_anchors[-1]["run_id"], "campaign_baseline", "medium"
                evidence = "new campaign anchor; linked to previous anchor"

        relation_names = []
        for pattern in (r"\bbuilt on(?: the)?\s+([a-z0-9_.-]+)", r"\bon top of(?: the)?\s+([a-z0-9_.-]+)", r"\bonto\s+(exp[_-]\d+)"):
            relation_names.extend(re.findall(pattern, body))
        relation_ids = [candidate for name in relation_names if (candidate := resolve_exact(run, name))]
        exp_ids = [candidate for name in dict.fromkeys(re.findall(r"\bexp[_-]\d+\b", body)) if (candidate := resolve_exact(run, name))]
        if not primary and (relation_ids or exp_ids):
            primary = (relation_ids or exp_ids)[0]
            method, confidence, evidence = "named_relation", "high" if relation_ids else "medium", f"notes name {by_id[primary]['title']}"

        if not primary:
            api_base = (run.get("repo", {}).get("main_diff") or {}).get("base_run_id")
            if api_base and earlier(run, api_base):
                primary, method, confidence, evidence = api_base, "source_branch", "source-only", "repo.main_diff.base_run_id"

        cited = [*direct, *relation_ids, *exp_ids]
        full_name = next(iter(re.findall(r"full_run_name\s*=\s*([^|,;\n]+)", body)), "")
        for component in full_name.split("+"):
            candidate = resolve_exact(run, component)
            if candidate:
                cited.append(candidate)
        secondary = [run_id for run_id in dict.fromkeys(cited) if run_id != primary]
        for parent in secondary:
            secondary_edges.append((parent, run["run_id"], "notes cite additional ingredient"))

        parent_run = by_id.get(primary)
        child_core, parent_core = signatures[run["run_id"]]["core"], signatures.get(primary, {}).get("core", {})
        common = set(child_core) & set(parent_core)
        compared_paths = set(child_core) | set(parent_core)
        source_similarity = sum(child_core[path] == parent_core[path] for path in common) / len(compared_paths) if primary and compared_paths else ""
        records.append({
            "child_run_id": run["run_id"], "child_title": run["title"], "ended_at": run["ended_at"],
            "score": run["metric_value"], "validation": run["validation"], "campaign": campaign(run),
            "primary_parent_id": primary or "", "primary_parent_title": parent_run["title"] if parent_run else "",
            "relation_method": method, "confidence": confidence, "evidence": evidence,
            "score_delta": run["metric_value"] - parent_run["metric_value"] if parent_run else "",
            "core_source_similarity": source_similarity, "secondary_parent_ids": "; ".join(secondary),
            "secondary_parent_titles": "; ".join(by_id[parent]["title"] for parent in secondary), "web_url": run["web_url"],
        })
    return records, secondary_edges


def write_collapsed_outputs(runs, records, graph):
    record_by_id, order = {record["child_run_id"]: record for record in records}, {run["run_id"]: index for index, run in enumerate(runs)}
    buckets = defaultdict(list)
    for run in runs:
        record = record_by_id[run["run_id"]]
        buckets[(record["campaign"], record["primary_parent_id"], collapse_family(run))].append(run)
    groups = sorted(buckets.items(), key=lambda item: min(order[run["run_id"]] for run in item[1]))
    node_of, info = {}, {}
    for index, ((group_campaign, _, family), members) in enumerate(groups):
        group_id, best = f"group_{index:04d}", max(members, key=lambda run: run["metric_value"])
        for run in members:
            node_of[run["run_id"]] = group_id
        info[group_id] = {"campaign": group_campaign, "family": family, "members": members, "best": best}

    collapsed, edge_buckets = nx.DiGraph(), defaultdict(list)
    for group_id, item in info.items():
        members, best = item["members"], item["best"]
        collapsed.add_node(group_id, family=item["family"], count=len(members), campaign=item["campaign"], best_run_id=best["run_id"], best_score=best["metric_value"], member_run_ids="; ".join(run["run_id"] for run in members))
    for parent, child, data in graph.edges(data=True):
        source, target = node_of[parent], node_of[child]
        if source != target:
            edge_buckets[(source, target)].append(data)
    for (source, target), edges in edge_buckets.items():
        collapsed.add_edge(source, target, count=len(edges), kinds="; ".join(sorted({edge["kind"] for edge in edges})), methods="; ".join(sorted({edge["method"] for edge in edges})))
    if not nx.is_directed_acyclic_graph(collapsed):
        raise RuntimeError("Collapsed lineage is not a DAG")

    mapping_csv = ROOT / "labless" / "experiment_lineage_groups.csv"
    with mapping_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "title", "group_id", "family", "group_size", "best_run_id"])
        writer.writeheader()
        for run in runs:
            group_id, item = node_of[run["run_id"]], info[node_of[run["run_id"]]]
            writer.writerow({"run_id": run["run_id"], "title": run["title"], "group_id": group_id, "family": item["family"], "group_size": len(item["members"]), "best_run_id": item["best"]["run_id"]})
    graphml = ROOT / "labless" / "experiment_lineage_collapsed.graphml"
    nx.write_graphml(collapsed, graphml)

    campaign_order = sorted({item["campaign"] for item in info.values()}, key=lambda name: min(run["ended_at"] for run in runs if campaign(run) == name))
    lines = ["digraph nanopath_collapsed {", f'graph [rankdir=LR, bgcolor="white", pad="0.15", nodesep="0.05", ranksep="0.48", splines=polyline, fontname="Helvetica", fontsize=18, labelloc=t, label="Nanopath grouped lineage: {len(runs):,} runs → {len(groups):,} boxes\\nxN = similar sibling iterations with the same campaign and parent"];', 'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=8, margin="0.06,0.04", color="#667085", penwidth=0.7];', 'edge [arrowsize=0.5, penwidth=0.7, color="#59636E"];']
    for index, name in enumerate(campaign_order):
        lines.extend([f"subgraph cluster_{index} {{", f'label="{dot_escape(name)}"; color="#D0D5DD"; penwidth=0.7; fontsize=11; fontname="Helvetica";'])
        for group_id, item in info.items():
            if item["campaign"] != name:
                continue
            members, best = item["members"], item["best"]
            values = sorted(run["metric_value"] for run in members)
            status = "leader" if any(run["validation"] == "leader" for run in members) else "validated" if any(run["validation"] == "validated" for run in members) else "unvalidated"
            border, width = ("#B8860B", 2.5) if status == "leader" else ("#17324D", 2.5) if status == "validated" else ("#667085", 0.7)
            family = best["title"] if item["family"].startswith("run:") else item["family"].replace("-", " ")
            label = f"{family[:42]}\\n{best['metric_value']:.4f} · {best['ended_at'][5:10]}" if len(members) == 1 else f"{family[:42]}\\nx{len(members)} iterations\\nbest {best['metric_value']:.4f} · median {values[len(values) // 2]:.4f}"
            names = "; ".join(run["title"] for run in members[:12]) + (f"; … +{len(members) - 12}" if len(members) > 12 else "")
            lines.append(f'{group_id} [label="{dot_escape(label)}", fillcolor="{PALETTE[run_themes(best)[0]]}", color="{border}", penwidth={width}, URL="{best["web_url"]}", target="_blank", tooltip="{dot_escape(names)}"];')
        lines.append("}")
    for source, target, data in collapsed.edges(data=True):
        methods, kinds = data["methods"], data["kinds"]
        if "primary" not in kinds:
            style, color, width = "dashed", "#9C4DCC", 0.65
        elif any(method in methods for method in ("explicit_run_id", "named_base", "named_relation")):
            style, color, width = "solid", "#344054", 0.85
        elif "campaign_baseline" in methods:
            style, color, width = "dashed", "#0072B2", 0.75
        else:
            style, color, width = "dotted", "#98A2B3", 0.55
        lines.append(f'{source} -> {target} [style={style}, color="{color}", penwidth={width}, tooltip="{data["count"]} run-level edge(s)"];')
    lines.append("}")
    dot = ROOT / "labless" / "experiment_lineage_collapsed.dot"
    dot.write_text("\n".join(lines))
    svg, png = ROOT / "imgs" / "labless_collapsed_lineage.svg", ROOT / "imgs" / "labless_collapsed_lineage.png"
    subprocess.run(["dot", "-Tsvg", str(dot), "-o", str(svg)], check=True)
    subprocess.run(["dot", "-Tpng:gd", str(dot), "-o", str(png)], check=True)

    primary = nx.DiGraph((source, target) for source, target, data in collapsed.edges(data=True) if "primary" in data["kinds"])
    primary.add_nodes_from(collapsed)
    primary_edges, secondary_edges = list(primary.edges), [edge for edge in collapsed.edges if "primary" not in collapsed.edges[edge]["kinds"]]
    leader_run = max((run for run in runs if run["validation"] == "leader"), key=lambda run: run["metric_value"])
    root = next(node for node in primary if primary.in_degree(node) == 0)
    leader_nodes = nx.shortest_path(primary, root, node_of[leader_run["run_id"]])
    leader_edges = list(zip(leader_nodes, leader_nodes[1:]))
    latest_date = max(dt.datetime.fromisoformat(run["submitted_at"]) for run in runs)
    latest_cutoff, recent_counts = latest_date - dt.timedelta(days=14), {}
    for group_id, item in info.items():
        recent_counts[group_id] = sum(dt.datetime.fromisoformat(run["submitted_at"]) >= latest_cutoff for run in item["members"])
    node_dates = {group_id: min(dt.datetime.fromisoformat(run["submitted_at"]) for run in item["members"]) for group_id, item in info.items()}
    family_scores = {}
    for group_id, item in info.items():
        scores = sorted(run["metric_value"] for run in item["members"])
        family_scores[group_id] = (scores[(len(scores) - 1) // 2] + scores[len(scores) // 2]) / 2
    positions = {node: (mdates.date2num(node_dates[node]), family_scores[node]) for node in collapsed}
    score_floor, max_y = 0.600, max(family_scores.values())
    fig, ax = plt.subplots(figsize=(22, 10))
    nx.draw_networkx_edges(collapsed, positions, edgelist=primary_edges, edge_color="#64748B", width=0.78, alpha=0.48, arrows=False, ax=ax)
    nx.draw_networkx_edges(collapsed, positions, edgelist=secondary_edges, edge_color="#8E44AD", style="dashed", width=0.65, alpha=0.28, arrows=False, ax=ax)

    path_load = Counter()
    for target, weight in recent_counts.items():
        if not weight:
            continue
        ancestry = nx.ancestors(primary, target) | {target}
        for edge in primary.subgraph(ancestry).edges:
            path_load[edge] += weight
    max_load = max(path_load.values())
    trunk_edges = list(path_load)
    strength = [math.sqrt(path_load[edge] / max_load) for edge in trunk_edges]
    navy = to_rgba("#0B3C5D")
    nx.draw_networkx_edges(collapsed, positions, edgelist=trunk_edges,
                           edge_color=[(*navy[:3], 0.28 + 0.62 * value) for value in strength],
                           width=[0.9 + 4.6 * value for value in strength], arrows=False, ax=ax)
    glow = nx.draw_networkx_edges(collapsed, positions, edgelist=leader_edges, edge_color="#A7F3C1", width=9.0, alpha=0.68, arrows=False, ax=ax)
    green = nx.draw_networkx_edges(collapsed, positions, edgelist=leader_edges, edge_color="#00A651", width=3.4, alpha=1.0, arrows=False, ax=ax)
    glow.set_zorder(2); green.set_zorder(2.1)
    for theme, color in PALETTE.items():
        selected = [(group_id, item) for group_id, item in info.items() if run_themes(item["best"])[0] == theme]
        ax.scatter([positions[group_id][0] for group_id, _ in selected], [positions[group_id][1] for group_id, _ in selected],
                   s=[11 + 7 * math.sqrt(len(item["members"])) for _, item in selected], c=color,
                   edgecolors=["#B8860B" if any(run["validation"] == "leader" for run in item["members"]) else "#17324D" if any(run["validation"] == "validated" for run in item["members"]) else "#667085" for _, item in selected],
                   linewidths=[1.5 if any(run["validation"] in {"leader", "validated"} for run in item["members"]) else 0.25 for _, item in selected], alpha=0.9, zorder=3)
    recent = [group_id for group_id, count in recent_counts.items() if count]
    ax.scatter([positions[group_id][0] for group_id in recent], [positions[group_id][1] for group_id in recent],
               s=[26 + 7 * math.sqrt(len(info[group_id]["members"])) for group_id in recent], facecolors="none", edgecolors="#0B3C5D", linewidths=0.8, zorder=4)
    ax.scatter([positions[group_id][0] for group_id in leader_nodes], [positions[group_id][1] for group_id in leader_nodes],
               s=[42 + 7 * math.sqrt(len(info[group_id]["members"])) for group_id in leader_nodes], facecolors="none", edgecolors="#00A651", linewidths=1.5, zorder=4.5)
    below_scale = [group_id for group_id, score in family_scores.items() if score < score_floor]
    ax.scatter([positions[group_id][0] for group_id in below_scale], [score_floor + 0.0008] * len(below_scale), marker="v", s=28,
               c=[PALETTE[run_themes(info[group_id]["best"])[0]] for group_id in below_scale], edgecolors="#667085", linewidths=0.35, zorder=4)
    label_offsets = {"dinov2-s-kde": (0, 25), "lr-and-curation": (0, -27), "I-JEPA contig patch": (0, 31),
                     "jepa-fino-s2026": (0, -33), "block-strided-cls": (8, 34), "bsc-s932-k10": (0, -29),
                     "lctx14-kde05-s7375": (0, 29), "robust-norm-s9876": (0, -31)}
    official = [run for run in runs if run["validation"] != "unvalidated"]
    for run in official:
        offset = label_offsets[run["title"]]
        group_id, color = node_of[run["run_id"]], "#B8860B" if run["validation"] == "leader" else "#17324D"
        ax.scatter(positions[group_id][0], positions[group_id][1], s=82 if run["validation"] == "leader" else 58, facecolors="none", edgecolors=color, linewidths=1.8, zorder=5)
        ax.annotate(run["title"], positions[group_id], xytext=offset, textcoords="offset points", ha="center",
                    va="bottom" if offset[1] > 0 else "top", fontsize=7, fontweight="bold", color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "lw": 0.55, "alpha": 0.92}, zorder=6)
    ax.set(xlabel="Experiment submission date", ylabel="Family median mean_probe_score (zoomed)", ylim=(score_floor - 0.0015, max_y + 0.003), yticks=[0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68], title=f"Nanopath v1 score-aware ancestry landscape · {len(runs):,} runs → {len(info):,} families · shared paths to {sum(recent_counts.values())} final-14-day runs")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.yaxis.set_major_formatter("{x:.3f}"); ax.tick_params(axis="both", which="both", bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.grid(axis="both", color="#E5E7EB", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    theme_legend = [Patch(facecolor=color, edgecolor="#667085", linewidth=0.4, label=theme) for theme, color in PALETTE.items()]
    size_legend = [Line2D([], [], marker="o", color="none", markerfacecolor="#A7A9AC", markeredgecolor="#667085", markersize=math.sqrt(11 + 7 * math.sqrt(count)), label=f"x{count} runs") for count in (1, 10, 30)]
    path_legend = [Line2D([], [], color="#64748B", lw=1.0, label="All primary ancestry"), Line2D([], [], color="#0B3C5D", lw=3.5, label="Shared path to recent runs"), Line2D([], [], color="#00A651", lw=4.0, label="Path to current leader"), Line2D([], [], color="#8E44AD", lw=1.0, ls="--", label="Secondary influence"), Line2D([], [], marker="v", color="none", markerfacecolor="#A7A9AC", markeredgecolor="#667085", label=f"Below 0.600 (n={len(below_scale)})"), Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#0B3C5D", label="Recent family"), Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#17324D", markeredgewidth=1.6, label="Official validated run"), Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#B8860B", markeredgewidth=1.8, label="Current leader")]
    ax.legend(handles=[*theme_legend, *size_legend, *path_legend], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=7, frameon=False, fontsize=8)
    fig.text(0.5, 0.02, f"Time runs left → right; y = family-median mean_probe_score, zoomed to ≥0.600 (▼ marks {len(below_scale)} lower families). Bright green traces the primary path to {leader_run['title']} ({leader_run['metric_value']:.4f}).", ha="center", fontsize=8, color="#586069")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.20)
    horizontal = [ROOT / "imgs" / f"labless_collapsed_lineage_horizontal.{suffix}" for suffix in ("png", "pdf")]
    for path in horizontal:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return collapsed, [mapping_csv, graphml, dot, svg, png, *horizontal]


def write_outputs(runs, records, secondary_edges):
    lineage_csv = ROOT / "labless" / "experiment_lineage.csv"
    with lineage_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    graph, by_id = nx.DiGraph(), {run["run_id"]: run for run in runs}
    for run in runs:
        graph.add_node(run["run_id"], title=run["title"], score=run["metric_value"], validation=run["validation"], campaign=campaign(run), contributor=run["contributor"], url=run["web_url"])
    for record in records:
        if record["primary_parent_id"]:
            graph.add_edge(record["primary_parent_id"], record["child_run_id"], kind="primary", method=record["relation_method"], confidence=record["confidence"], evidence=record["evidence"])
    for parent, child, evidence in secondary_edges:
        if not graph.has_edge(parent, child):
            graph.add_edge(parent, child, kind="secondary", method="cited_influence", confidence="medium", evidence=evidence)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Reconstructed lineage is not a DAG")
    graphml = ROOT / "labless" / "experiment_lineage.graphml"
    nx.write_graphml(graph, graphml)

    dot = ROOT / "labless" / "experiment_lineage.dot"
    campaign_order = sorted({campaign(run) for run in runs}, key=lambda name: min(run["ended_at"] for run in runs if campaign(run) == name))
    lines = ["digraph nanopath {", f'graph [rankdir=LR, bgcolor="white", pad="0.15", nodesep="0.04", ranksep="0.42", splines=polyline, fontname="Helvetica", fontsize=18, labelloc=t, label="Nanopath: complete {len(runs):,}-run development lineage\\nsolid = named run/base · blue dashed = inferred campaign baseline · gray dotted = API source fork · purple dashed = cited ingredient"];', 'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=7, margin="0.05,0.035", color="#667085", penwidth=0.7];', 'edge [arrowsize=0.45, penwidth=0.65, color="#59636E"];']
    for index, name in enumerate(campaign_order):
        lines.extend([f"subgraph cluster_{index} {{", f'label="{dot_escape(name)}"; color="#D0D5DD"; penwidth=0.7; fontsize=11; fontname="Helvetica";'])
        for run in runs:
            if campaign(run) != name:
                continue
            theme = run_themes(run)[0]
            border = "#B8860B" if run["validation"] == "leader" else "#17324D" if run["validation"] == "validated" else "#667085"
            width = 2.5 if run["validation"] in {"leader", "validated"} else 0.7
            label = dot_escape(run["title"][:34]) + f"\\n{run['metric_value']:.4f} · {run['ended_at'][5:10]}"
            tooltip = dot_escape(f"{run['run_id']} | {run['contributor']} | {run['validation']} | {run['title']}")
            lines.append(f'n_{run["run_id"]} [label="{label}", fillcolor="{PALETTE[theme]}", color="{border}", penwidth={width}, URL="{run["web_url"]}", target="_blank", tooltip="{tooltip}"];')
        lines.append("}")
    for parent, child, data in graph.edges(data=True):
        if data["kind"] == "secondary":
            style, color, width = "dashed", "#9C4DCC", 0.65
        elif data["method"] == "campaign_baseline":
            style, color, width = "dashed", "#0072B2", 0.75
        elif data["method"] == "source_branch":
            style, color, width = "dotted", "#98A2B3", 0.55
        else:
            style, color, width = "solid", "#344054", 0.8
        lines.append(f'n_{parent} -> n_{child} [style={style}, color="{color}", penwidth={width}, tooltip="{dot_escape(data["evidence"])}"];')
    lines.append("}")
    dot.write_text("\n".join(lines))
    full_svg, full_png = ROOT / "imgs" / "labless_full_lineage.svg", ROOT / "imgs" / "labless_full_lineage.png"
    subprocess.run(["dot", "-Tsvg", str(dot), "-o", str(full_svg)], check=True)
    subprocess.run(["dot", "-Tpng:gd", str(dot), "-o", str(full_png)], check=True)
    collapsed, collapsed_outputs = write_collapsed_outputs(runs, records, graph)

    overview_dot = ROOT / "labless" / "experiment_lineage_overview.dot"
    grouped = defaultdict(list)
    for run in runs:
        grouped[campaign(run)].append(run)
    counts = Counter((campaign(by_id[u]), campaign(by_id[v])) for u, v in graph.edges if campaign(by_id[u]) != campaign(by_id[v]))
    overview = ["digraph overview {", 'graph [rankdir=LR, bgcolor="white", pad="0.2", nodesep="0.25", ranksep="0.8", splines=spline, fontname="Helvetica", fontsize=18, labelloc=t, label="Nanopath experiment campaigns and cross-campaign inheritance"];', 'node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#344054", fontname="Helvetica", fontsize=10];', 'edge [color="#667085", arrowsize=0.7, fontname="Helvetica", fontsize=8];']
    for index, name in enumerate(campaign_order):
        values = [run["metric_value"] for run in grouped[name]]
        overview.append(f'c{index} [label="{name}\\n{len(values)} runs · median {sorted(values)[len(values)//2]:.4f}\\nbest {max(values):.4f}"];')
    name_index = {name: index for index, name in enumerate(campaign_order)}
    for (source, target), count in counts.items():
        overview.append(f'c{name_index[source]} -> c{name_index[target]} [label="{count}", penwidth={0.7 + min(count, 40) / 12:.2f}];')
    overview.append("}")
    overview_dot.write_text("\n".join(overview))
    for suffix in ("png", "pdf"):
        subprocess.run(["dot", f"-T{suffix}", str(overview_dot), "-o", str(ROOT / "imgs" / f"labless_lineage_overview.{suffix}")], check=True)
    return graph, collapsed, [lineage_csv, graphml, dot, full_svg, full_png, *collapsed_outputs, overview_dot, ROOT / "imgs" / "labless_lineage_overview.png", ROOT / "imgs" / "labless_lineage_overview.pdf"]


def main():
    runs = fetch_runs()
    signatures = fetch_source_signatures(runs)
    records, secondary_edges = reconstruct(runs, signatures)
    graph, collapsed, outputs = write_outputs(runs, records, secondary_edges)
    print(json.dumps({"runs": len(runs), "grouped_boxes": collapsed.number_of_nodes(), "primary_edges": len(runs) - 1, "secondary_edges": len(secondary_edges), "methods": Counter(record["relation_method"] for record in records), "dag": nx.is_directed_acyclic_graph(graph), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
