"""Audit shared data, train-only statistics and checkpoint budgets before scores."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from blind_diffusion.data.transforms import compute_mean_std
from blind_diffusion.utils.checkpoint import load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--task",choices=["lift","can"],required=True)
    args=parser.parse_args()
    task_root=args.root/args.task
    wm=load_checkpoint(str(task_root/"wm/checkpoints/best.pt"),map_location="cpu")
    keys=wm["config"]["task"]["obs_keys"]
    split=wm["split"]
    with h5py.File(args.data/args.task/"low_dim_v141.hdf5","r") as handle:
        if set(split["train"]) & set(split["val"]) or set(split["train"]+split["val"])!=set(handle["data"]):
            raise ValueError("Episode split does not partition this dataset")
        observations=np.concatenate([np.concatenate([handle[f"data/{name}/obs/{key}"][:] for key in keys],-1)
                                     for name in split["train"]])
        mean,std=compute_mean_std(observations)
        if not np.allclose(mean,wm["norm"]["obs_mean"],rtol=1e-6,atol=1e-9) or not np.allclose(std,wm["norm"]["obs_std"],rtol=1e-6,atol=1e-9):
            raise ValueError("Checkpoint normalization does not match training-only statistics of the current data")
        windows=sum(max(0,min(len(handle[f"data/{name}/actions"])-1,
                               len(handle[f"data/{name}/actions"])-32+8)+1) for name in split["train"])
    expected_updates=100*((windows+127)//128)
    models={}
    for method in ("cfm","ddim","bc"):
        path=task_root/("bc/best.pt" if method=="bc" else f"{method}/checkpoints/best.pt")
        checkpoint=load_checkpoint(str(path),map_location="cpu")
        if checkpoint["split"]!=split:
            raise ValueError(f"{method}: different episode split")
        if method=="bc":
            if checkpoint["config"]["epochs"]!=100 or checkpoint["config"]["batch_size"]!=128:
                raise ValueError("BC used a different update budget")
            metric=json.loads((task_root/"bc/metrics.json").read_text())
            if metric["updates"]!=expected_updates:
                raise ValueError("BC update count does not match the generative budget")
        else:
            cfg=checkpoint["config"]
            expected_kind="flow_matching" if method=="cfm" else "diffusion"
            checks=[cfg.get("policy_type","diffusion")==expected_kind,cfg["epochs"]==100,
                    cfg["batch_size"]==128,cfg["max_train_steps"] is None,
                    cfg["diffusion"]["architecture"]=="skip_v2",cfg["diffusion"]["base_ch"]==128,
                    cfg["diffusion"]["horizon"]==8,cfg["prior_aug_prob"]==.5,
                    cfg["normalize_obs"] and not cfg["normalize_action"],cfg["ema_decay"]==.995]
            if not all(checks):
                raise ValueError(f"{method}: architecture or training protocol mismatch")
        models[method]={"same_split":True,"ema_selected":"ema_model" in checkpoint}
    result={"task":args.task,"status":"passed","train_demos":len(split["train"]),"val_demos":len(split["val"]),
            "training_only_normalization_matches":True,"train_windows":windows,
            "updates_per_learned_policy":expected_updates,"models":models}
    (args.root/f"input_audit_{args.task}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
