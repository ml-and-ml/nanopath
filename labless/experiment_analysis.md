<!-- Manuscript-facing synthesis from 2026-08-23; ancestry and breadth artifacts refreshed from the complete v1 ledger on 2026-09-05. -->
<!-- Regenerate the figures and compact run index with labless/plot_experiments.py. -->

# Nanopath Labless experiment analysis

## Scope and method

The public Labless API contained **1,242 Nanopath run records** submitted from 2026-06-03 through 2026-08-21 by 16 contributors. All are labeled `tier=full`, but they are not 1,242 independent pretraining trajectories: at least 42 notes explicitly describe eval-only, probe-rerun, or post-training work. The ledger contains 1,234 unvalidated runs, seven validated runs, and one current leader.

The plots use every API row with a reported `mean_probe_score`. Experiment themes are transparent keyword tags over titles and notes, checked against representative source patches; they are non-exclusive because many later runs deliberately combine earlier ideas. The compact [experiment index](experiment_index.csv) preserves every run ID, score, theme, explicit parent reference, comparison base, and API URL. The lineage reconstruction additionally fetched and hashed the saved source snapshot for all 1,242 rows.

## What kinds of experiments were run?

| Theme | Tagged runs | Median score | Representative interventions |
|---|---:|---:|---|
| Hyperparameters / schedules | 558 | 0.6493 | learning rate, warmup, KDE weight/concentration, mask ratio, crop count/size, seeds, teacher/EMA schedules |
| Data / augmentation | 315 | 0.6489 | tissue curation, stain/HED/HSV changes, magnification and crop policies, deduplication and sampling |
| New objectives / losses | 385 | 0.6487 | contiguous-block I-JEPA, SimCLR/NT-Xent, VICReg, KoLeo, Gram anchoring, SigReg, CAPI, focal/reweighting losses |
| Molecular / text supervision | 170 | 0.6482 | FINO subtype/expression/FGA targets, molecular captions, genomic and patient-metadata auxiliaries |
| Architecture / backbone | 104 | 0.6474 | DINOv3 and ViT-5 backbones, register-token variants, layer specialization, predictor depth/width |
| Readout / post-training | 84 | **0.6517** | multi-depth CLS concatenation, dense patch fusion/JBU, PCA/shrinkage, robust-norm, TTA |
| Other / controls | 291 | 0.6468 | infrastructure, replications, accounting controls, and notes too generic to classify safely |

So the campaign was emphatically broader than hyperparameter tuning. Hyperparameter and augmentation sweeps were the largest strata, but the validated trajectory required changes to the pretraining objective, auxiliary supervision, representation readout, and post-training normalization.

## Validated development record

| Date | Run | Main intervention | Score | Change from preceding validated milestone |
|---|---|---|---:|---:|
| 2026-06-03 | [`dinov2-s-kde`](https://labless.dev/runs/run_sub_0d8aeb2511) | DINOv2-S + iBOT + KDE starting point | 0.6277 | — |
| 2026-06-05 | [`lr-and-curation`](https://labless.dev/runs/run_sub_6c6c051f71) | tissue threshold, shorter warmup, Adam β2 | 0.6357 | +0.0080 |
| 2026-06-09 | [`I-JEPA contig patch`](https://labless.dev/runs/run_sub_816fc5d0ca) | replace iBOT patch classification with contiguous-block latent regression | 0.6444 | +0.0088 |
| 2026-06-13 | [`jepa-fino-s2026`](https://labless.dev/runs/run_sub_1879d32919) | add subtype, expression-512, and FGA metadata guidance | 0.6485 | +0.0041 |
| 2026-06-17 | [`block-strided-cls`](https://labless.dev/runs/run_sub_59d24d6b7b) | multi-depth CLS readout plus dense multi-block segmentation features | 0.6592 | **+0.0108** |
| 2026-06-23 | [`bsc-s932-k10`](https://labless.dev/runs/run_sub_8b24ba430d) | changed-seed validation of KDE 0.05 candidate | 0.6519 | −0.0074 vs leader |
| 2026-07-24 | [`lctx14-kde05-s7375`](https://labless.dev/runs/run_sub_a80b532bdc) | changed-seed validation of 14×128 local crops + KDE 0.05 | 0.6580 | −0.0012 vs leader |
| 2026-08-20 | [`robust-norm-s9876`](https://labless.dev/runs/run_sub_16b156161d) | 14×128 crops, KDE 0.1, block-2 readout, scanner-axis suppression | **0.6676** | **+0.0083** |

The validated score rose from 0.6277 to 0.6676, an absolute gain of **0.0399** (6.35% relative). The gains were compositional rather than monotonic across tasks:

- I-JEPA was a genuine objective overhaul and produced a validated +0.0088 over the preceding milestone. Its source patch adds a transformer predictor and smooth-L1 regression on contiguous masked blocks.
- FINO was a second major overhaul: categorical metadata updates prototype banks and continuous expression/FGA targets use regression heads. It added +0.0041 after validation—useful, but below Nanopath's 0.006 promotion threshold.
- The largest isolated validated step was representation access, not a new loss. `block-strided-cls` gained +0.0108 by concatenating intermediate CLS states and exposing denser multi-block patch features. Relative to FINO it improved linear/kNN/16-shot/segmentation by +0.026/+0.022/+0.037/+0.039, while progression and mutation fell by −0.029 and −0.013.
- The current leader adds local-context/KDE tuning and scanner-sensitive direction suppression. Relative to `block-strided-cls`, it gained +0.065 progression, +0.018 robustness, +0.014 16-shot, and +0.006 survival, while linear, kNN, and mutation declined. This is evidence of task tradeoffs, not uniform representation improvement.

## Major overhauls and null directions

Several changes were large enough to count as alternate systems rather than tuning:

- **Backbone replacement:** DINOv3-S/16 + JEPA reached 0.6405; ViT-5-B/16 and ViT-5-S/16 reached 0.6149 and 0.6012. None produced a validated leader.
- **Objective replacement:** pure DINOv3-initialized SimCLR with a temperature curriculum reached 0.6513. Transferring robust post-training suppression left it essentially unchanged at 0.6510.
- **Molecular/text supervision:** FINO produced a validated but sub-threshold step; MolCap text supervision reached 0.6652 on a strong unvalidated base but was described as a null overall result.
- **Large JEPA/register search:** the best of a reported 100-run sweep reached 0.6650 but remains unvalidated.
- **Data curation:** a sophisticated downstream-informed curation pipeline reached 0.6653 with curation disabled in the submitted control, so it does not establish a curation benefit.
- **TTA:** `rot180-tta` is the highest raw score at 0.6758, but its own note reports only +0.0002 over its parent—well inside the same-seed band. It should not be presented as an independent breakthrough.

## The strongest meta-result: discovery scores overstate gains

Across seven directly matched discovery→validation families, six validation scores were lower. The median gap was **−0.0064** and the mean gap was **−0.0069**, approximately the full promotion threshold. The largest drops were KDE 0.05 (−0.0141), local context (−0.0104), and the robust-norm family (−0.0080). This is the cleanest NeurIPS-level lesson from the ledger: broad single-seed search is productive for proposing mechanisms, but independent changed-seed replication is necessary before attributing improvements.

The search was also highly concentrated: RyanKim17920 submitted 1,057 of 1,242 records (85.1%), many in explicitly labeled automated cycles. For the paper, the 1,242-run scale should therefore be described as a dense collaborative/agentic experiment ledger, not as 1,242 statistically independent confirmations.

## Figure guide and caveats

- [Experiment-breadth figure](../imgs/labless_experiment_breadth.pdf): a category-colored weekly density map, non-exclusive theme totals, and the cumulative activity curve for all 1,252 v1 runs. It shows the breadth of architecture, objective/loss, data, molecular/text, readout, and hyperparameter experimentation without pretending combined runs belong to only one category. A PNG preview is [here](../imgs/labless_experiment_breadth.png).
- [Trend/performance figure](../imgs/labless_experiment_trends.pdf): all runs, theme distributions, validation gaps, and task-level changes. A PNG preview is [here](../imgs/labless_experiment_trends.png).
- [Development-lineage figure](../imgs/labless_run_lineage.pdf): landmark branches with explicit parent references shown as solid arrows and source/notes-based inferences shown as dashed arrows. A PNG preview is [here](../imgs/labless_run_lineage.png).
- Complete run-level flowchart: all 1,252 v1 API rows as a browser-searchable linked [SVG](../imgs/labless_full_lineage.svg) or full-resolution [PNG](../imgs/labless_full_lineage.png). The compact campaign overview is available as [PDF](../imgs/labless_lineage_overview.pdf) and [PNG](../imgs/labless_lineage_overview.png); the editable network is [GraphML](experiment_lineage.graphml).
- Collapsed-family views: the recommended minimally labeled v1 ancestry landscape is available as [PNG](../imgs/labless_collapsed_lineage_horizontal.png) and [PDF](../imgs/labless_collapsed_lineage_horizontal.pdf). It retains every aggregate connection across all 1,252 runs and 695 families, labels only the seven official validated runs and current leader, outlines the 12 families containing the 15 final-14-day runs, and draws their 19-edge shared ancestry in navy with width proportional to downstream recent-run count. A bright green halo traces the unique primary path from the root to the current validated leader. Time is the x-axis; y is the median `mean_probe_score` among runs in each collapsed family (and the exact score for singleton families). The display is zoomed to 0.600–0.684 so the dense performance band is readable; downward triangles mark the 11 lower-scoring families at the boundary. Color encodes intervention family and node area encodes collapsed run count. The labeled audit flowchart ([SVG](../imgs/labless_collapsed_lineage.svg), [PNG](../imgs/labless_collapsed_lineage.png)) prints `xN iterations` inside each multi-run box.
- Conceptual Sankey: the dated, manuscript-facing lineage synthesis is available as [PNG](../imgs/labless_conceptual_lineage_sankey.png) and [PDF](../imgs/labless_conceptual_lineage_sankey.pdf). Independent objective, supervision, architecture, readout, and post-training ideas begin as separate colors; major combinations adopt a new lineage color, and automated-search widths use `log2(run count + 1)`. Autoresearch cycle names are excluded: the search boxes are neutral date-window cohorts. Its curated [node](conceptual_lineage_nodes.csv) and [edge](conceptual_lineage_edges.csv) tables make every interpretive merge auditable. Regenerate it with `uv run --with matplotlib python labless/plot_conceptual_sankey.py`.
- Git/source Sankey: the provenance-only counterpart is available as [PNG](../imgs/labless_git_source_lineage_sankey.png) and [PDF](../imgs/labless_git_source_lineage_sankey.pdf). The 2026-09-05 refresh maps all 1,252 v1 runs through 695 collapsed families into 40 dated three-day cohorts and preserves 1,251 primary plus 26 secondary ancestry records; 130 within-cohort links appear as right-side loops. Visible autoresearch cycles are collapsed into one neutral `Automated search` lane, while their original submitted labels remain available only in the audit data. Ribbon color denotes evidence type rather than scientific interpretation. Its complete cohort-edge accounting is in the [audit CSV](git_source_sankey_audit.csv); regenerate it with `uv run --with matplotlib --with pandas python labless/plot_git_source_sankey.py`.
- [Auditable edge table](experiment_lineage.csv): one row per run with its primary parent, any secondary cited ingredients, inference method, confidence, evidence, score delta, and source-hash similarity. Regenerate all lineage artifacts with `uv run --with matplotlib --with networkx python labless/plot_full_lineage.py`.

The refreshed family view reduces 1,252 runs to 695 nodes by combining only runs with the same campaign, primary parent, and semantic intervention-family signature; all validated/leader runs remain individual. The complete graph is evidence-ranked rather than uniformly certain: 11 primary edges cite exact run IDs, 337 resolve a named base, 80 resolve a named prior experiment, 245 connect to the nearest prior campaign baseline, and 578 retain only Labless's exact `repo.main_diff.base_run_id` source fork. The latter are labeled `source-only`: they establish the code branch point, not that the child conceptually depended on every intervening idea. Identical source hashes were used to corroborate edges, never to turn neighboring sweep runs into a fabricated chain. Labless exposes final metrics and source diffs but not raw console logs or full per-step histories, so these figures summarize downstream endpoints rather than training dynamics.

The Sankey is intentionally conceptual rather than a claim that every colored input is a Git parent. For example, FINO and MolCap are drawn as new supervision streams when they first enter an existing recipe. SAM is retained as a distinct three-run recipe family. No public title, note, or unique `model.py` snapshot contains Mamba/SSM/selective-scan code, so no unsupported Mamba lineage is shown.
