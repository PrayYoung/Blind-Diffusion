"""Epoch-boundary recovery without changing the optimizer or random stream."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(2**20),b""):
            digest.update(block)
    return digest.hexdigest()


def signature(config, dataset, world_model=None):
    values=dict(config)
    for key in ("run_dir","exp_name","resume_checkpoint","wm_checkpoint","dataset","world_model","output"):
        values.pop(key,None)
    values["dataset_sha256"]=file_sha256(dataset)
    if world_model:
        values["world_model_sha256"]=file_sha256(world_model)
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def capture_rng():
    return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy":np.random.get_state(),"python":random.getstate()}


def restore_rng(value):
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"]:
        if not torch.cuda.is_available() or len(value["cuda"])!=torch.cuda.device_count():
            raise ValueError("Resuming requires the same CUDA device count")
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])


def make_state(optim,scheduler,training_signature,best_path,**progress):
    return {"version":1,"signature":training_signature,"optimizer":optim.state_dict(),
            "scheduler":scheduler.state_dict(),"rng":capture_rng(),
            "best_checkpoint":load_checkpoint(str(best_path),map_location="cpu") if Path(best_path).exists() else None,
            **progress}


def restore_state(checkpoint,optim,scheduler,training_signature,best_path):
    state=checkpoint["training_state"]
    if state["version"]!=1 or state["signature"]!=training_signature:
        raise ValueError("Training recovery signature does not match data, budget or hyperparameters")
    optim.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state["best_checkpoint"] is not None:
        save_checkpoint(str(best_path),state["best_checkpoint"])
    elif Path(best_path).exists():
        Path(best_path).replace(str(best_path)+f".interrupted-{time.time_ns()}")
    restore_rng(state["rng"])
    return state


def reconcile_log(path,step,epoch):
    """Retain interrupted-attempt evidence, trimming records beyond recovery."""
    path=Path(path)
    if not path.exists():
        return
    lines=path.read_text().splitlines()
    kept=[]
    for line in lines:
        row=json.loads(line)
        if "epoch" in row:
            valid=row["epoch"]<epoch
        elif "step" in row:
            valid=row["step"]<=step if "val_loss" in row else row["step"]<step
        else:
            valid=True
        if valid:
            kept.append(line)
    if len(kept)!=len(lines):
        shutil.copy2(path,str(path)+f".interrupted-{time.time_ns()}")
        path.write_text("\n".join(kept)+("\n" if kept else ""))
