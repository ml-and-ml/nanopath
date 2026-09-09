# Frozen gate ablation: preserve the trained checkpoint and run the fixed full probe suite.
# python evaluate.py configs/main.yaml checkpoint_path=/data/.../latest.pt output_dir=/data/.../gate-off
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
import wandb
import yaml

from probe import collect_probe_results, completed_probe_summary, prepare_probe_state, queue_probe_job


def main():
    opts = dict(arg.split("=", 1) for arg in sys.argv[2:])
    checkpoint_path = Path(os.path.expandvars(opts["checkpoint_path"])).expanduser().resolve()
    output_dir = Path(os.path.expandvars(opts["output_dir"])).expanduser().resolve()
    parent_summary = json.loads((checkpoint_path.parent / "summary.json").read_text())
    with checkpoint_path.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = {key: checkpoint[key] for key in ("model", "model_ema", "step", "config")}
    cfg = checkpoint["config"]
    roots = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))["probe"]["dataset_roots"]
    cfg["probe"]["dataset_roots"] = roots
    cfg["project"].update(name=output_dir.name, output_dir=str(output_dir), recipe_id=parent_summary["recipe_id"] + "-gate-off")
    cfg["evaluation"] = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256, "gate_floor": 1.0}
    # A unit floor makes every typicality weight one; all learned tensors and other readouts stay fixed.
    for key in ("model", "model_ema"):
        checkpoint[key]["ct_lo"].fill_(1.0)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    source_dir = output_dir / "labless_source"
    shutil.copytree(checkpoint_path.parent / "labless_source", source_dir)
    shutil.copy2(__file__, source_dir / "evaluate.py")
    cfg["config_path"] = str(source_dir / "configs" / "main.yaml")
    Path(cfg["config_path"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
    wandb_run = wandb.init(project="nanopath", name=output_dir.name, dir=cfg["project"]["wandb_dir"], config=cfg)
    started = time.monotonic()
    state = prepare_probe_state(cfg, output_dir)
    step = checkpoint["step"]
    queue_probe_job(state, checkpoint, step, parent_summary["train_flops"], parent_summary["sample_fraction"])
    metrics_path = output_dir / "metrics.jsonl"
    collect_probe_results(state, wandb_run, metrics_path)
    # Keep inherited training costs explicit: this evaluation performs no further model fitting.
    summary = {**parent_summary, **completed_probe_summary(output_dir), "project": output_dir.name,
               "recipe_id": cfg["project"]["recipe_id"], "config_path": cfg["config_path"],
               "evaluation_only": True, "evaluation": cfg["evaluation"], "parent_run_dir": str(checkpoint_path.parent),
               "parent_wandb": parent_summary["wandb"], "parent_slurm_job_id": parent_summary["slurm_job_id"],
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "additional_train_flops": 0, "additional_tile_presentations": 0,
               "evaluation_wall_seconds": time.monotonic() - started,
               "wandb": {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id,
                         "name": output_dir.name, "url": wandb_run.url, "source_dir": str(source_dir)}}
    summary["mean_probe_score"] = sum(summary["final_probe_" + key] for key in (
        "linear_mean_f1", "knn_mean_f1", "fewshot_mean_f1", "seg_mean_f1", "slide_mean_auc",
        "auc_mean", "survival_mean_cindex", "robustness_mean")) / 8
    comparison = {key: summary[key] for key in ("final_score", "mean_probe_score")}
    comparison.update({"delta_" + key: summary[key] - parent_summary[key] for key in comparison.copy()})
    summary.update(comparison)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with metrics_path.open("a") as handle:
        handle.write(json.dumps({"event": "frozen_eval", "final": True, "step": step, **comparison}) + "\n")
    wandb_run.summary.update(comparison)
    wandb_run.log(comparison, step=step + 1)
    wandb_run.finish()
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
