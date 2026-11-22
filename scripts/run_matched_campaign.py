"""Workload orchestration only; cloud lifecycle belongs to the external runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import torch

from blind_diffusion.eval.matched_baselines import digest_json, evaluate_episode, file_digest
from blind_diffusion.utils.checkpoint import load_checkpoint
from collect_matched_results import collect


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def save_evaluation_snapshot(output, task, method, duration):
    """Publish completed records atomically for downloads during a live cell."""
    destination=output/"snapshots"/f"evaluate_{task}_{method}_{duration}.tar.gz"
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(".incoming")
    files=[output/"protocol.json"]
    files += sorted((output/"episodes"/task/method/f"d{duration}").glob("seed*.json"))
    files += sorted((output/"initial_states"/task).glob("seed*.json"))
    with tarfile.open(temporary,"w:gz") as archive:
        for path in files:
            archive.add(path,arcname=str(Path("outputs/matched_baselines")/path.relative_to(output)))
    os.replace(temporary,destination)
    return destination


def train_stage(name, command, output, artifact, protocol):
    marker = output / "stages" / f"{name}.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["protocol_sha256"] != digest_json(protocol) or prior["checkpoint_sha256"] != file_digest(str(artifact)):
            raise ValueError(f"Saved training stage is incompatible: {name}")
        return
    last=artifact.parent/"last.pt"
    if last.exists():
        checkpoint=load_checkpoint(str(last),map_location="cpu")
        if "training_state" not in checkpoint:
            raise ValueError("Incomplete legacy weights lack optimizer/RNG state; do not silently warm-start them")
        if name.endswith("_bc"):
            command += ["--resume-checkpoint",str(last)]
        else:
            command += [f"resume_checkpoint={last}"]
    progress = output / "progress.json"
    print("MATCHED_STAGE " + name,flush=True)
    log_path = output / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    with log_path.open("a") as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                row = {"status":"running", "stage":name,"elapsed_seconds":int(time.monotonic()-started)}
                for history in (artifact.parent.parent/"diffusion_logs.jsonl",
                                artifact.parent.parent/"logs.jsonl",artifact.parent/"logs.jsonl"):
                    if history.exists():
                        entries=history.read_text().splitlines()
                        if entries:
                            row["latest"]=json.loads(entries[-1])
                        break
                atomic_json(progress,row)
                print("MATCHED_PROGRESS " + json.dumps(row),flush=True)
    if code:
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-50:]),flush=True)
        raise RuntimeError(f"Training {name} exited {code}")
    checkpoint = load_checkpoint(str(artifact),map_location="cpu")
    tensors = checkpoint.get("ema_model",checkpoint["model"])
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise FloatingPointError(f"Nonfinite checkpoint: {name}")
    atomic_json(marker,{"stage":name,"protocol_sha256":digest_json(protocol),
                        "checkpoint_sha256":file_digest(str(artifact)),"seconds":time.monotonic()-started})


def checkpoint_paths(output,task):
    return {"wm":output/task/"wm/checkpoints/best.pt", "cfm":output/task/"cfm/checkpoints/best.pt",
            "ddim":output/task/"ddim/checkpoints/best.pt", "bc":output/task/"bc/best.pt"}


def ensure_dataset(data,task):
    target = data/task/"low_dim_v141.hdf5"
    if not target.exists():
        subprocess.run([sys.executable,"-m","robomimic.scripts.download_datasets","--tasks",task,
            "--dataset_types","ph","--hdf5_types","low_dim","--download_dir",str(data)],check=True)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(data/task/"ph/low_dim_v141.hdf5",target)
    return target


def training(output,inputs,data,protocol,target,task_filter=None):
    tasks = [target.split("_")[0]] if target else [task_filter] if task_filter else list(protocol["methods"])
    for task in tasks:
        dataset = ensure_dataset(data,task)
        paths = checkpoint_paths(output,task)
        for name in ("wm","cfm","ddim","bc"):
            stage = f"{task}_{name}"
            if target and stage != target:
                continue
            if task == "lift" and name in ("wm","cfm"):
                source = inputs/f"lift_{name}.pt"
                paths[name].parent.mkdir(parents=True,exist_ok=True)
                if paths[name].exists() and file_digest(str(paths[name])) != file_digest(str(source)):
                    raise ValueError("Refuse to overwrite a different reused checkpoint")
                shutil.copy2(source,paths[name])
                atomic_json(output/"stages"/f"{stage}.json",{"stage":stage,"reused":True,
                    "protocol_sha256":digest_json(protocol),"checkpoint_sha256":file_digest(str(paths[name]))})
                continue
            common = [f"task={task}",f"run_dir={output/task}","seed=0","num_workers=0"]
            if name == "wm":
                command = [sys.executable,"-m","blind_diffusion.train.train_world_model",
                    "--config-name","train_world_model_recovery",*common,"exp_name=wm"]
            elif name in ("cfm","ddim"):
                command = [sys.executable,"-m","blind_diffusion.train.train_diffusion","--config-name",
                    "train_flow_matching_recovery" if name == "cfm" else "train_diffusion_recovery",
                    *common,f"exp_name={name}",f"wm_checkpoint={paths['wm']}"]
            else:
                command = [sys.executable,"-m","blind_diffusion.train.train_bc_matched","--dataset",str(dataset),
                    "--world-model",str(paths["wm"]),"--output",str(output/task/"bc")]
            train_stage(stage,command,output,paths[name],protocol)


def make_jobs(output,data,protocol,task_filter=None):
    jobs = []
    for task, methods in protocol["methods"].items():
        if task_filter and task != task_filter:
            continue
        dataset = ensure_dataset(data,task)
        paths = checkpoint_paths(output,task)
        for method in methods:
            policy = paths["cfm"] if method.startswith("cfm") else paths["bc"] if method == "bc" else paths["ddim"]
            for duration in protocol["blackout_durations"]:
                for seed in range(protocol["episodes"][task]):
                    jobs.append({"task":task,"method":method,"duration":duration,"seed":seed,
                        "dataset":str(dataset),"wm":str(paths["wm"]),"policy":str(policy) if method != "mpc" else None,
                        "device":"cuda" if method == "mpc" and torch.cuda.is_available() else "cpu",
                        "initial_states_dir":str(output/"initial_states"),
                        "protocol":protocol,"output":str(output/"episodes"/task/method/f"d{duration}"/f"seed{seed}.json")})
    return jobs


def evaluate(output,data,protocol,workers,cell=None,task_filter=None):
    jobs = make_jobs(output,data,protocol,task_filter)
    if cell:
        task,method,duration=cell.split(":")
        jobs=[job for job in jobs if (job["task"],job["method"],job["duration"])==(task,method,int(duration))]
        if not jobs:
            raise ValueError("Requested cell is outside the frozen benchmark grid")
    for job in jobs:
        if not Path(job["wm"]).exists() or (job["policy"] and not Path(job["policy"]).exists()):
            raise FileNotFoundError("Retrieve and verify all trained checkpoints before evaluation")
    started, completed, last_snapshot = time.monotonic(), 0, 0.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        futures = {pool.submit(evaluate_episode,job):job for job in jobs}
        while futures:
            done, _ = wait(futures,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    atomic_json(output/"engineering_failure.json",{"job":job,"error":repr(error)})
                    for pending in futures:
                        pending.cancel()
                    raise
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    print("MATCHED_EPISODES " + json.dumps({"completed":completed,"total":len(jobs),
                        "latest":{k:result[k] for k in ("task","method","duration","seed","success")}}),flush=True)
            atomic_json(output/"progress.json",{"status":"evaluating","completed":completed,"total":len(jobs),
                                                 "elapsed_seconds":int(time.monotonic()-started)})
            if cell and time.monotonic()-last_snapshot>=20:
                save_evaluation_snapshot(output,task,method,int(duration))
                last_snapshot=time.monotonic()
            if not done:
                print(f"MATCHED_PROGRESS episodes={completed}/{len(jobs)}",flush=True)
    if cell:
        save_evaluation_snapshot(output,task,method,int(duration))
    summary = collect(output,protocol)
    selected_complete = all(value["complete"] for methods in summary["results"].values()
                            for durations in methods.values() for value in durations.values())
    if task_filter:
        selected_complete = all(value["complete"] for durations in summary["results"][task_filter].values()
                                for value in durations.values())
    if not cell and not selected_complete:
        raise RuntimeError("Episode grid is incomplete; no final table may be claimed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("scripts/matched_protocol.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/matched_baselines"))
    parser.add_argument("--inputs",type=Path,default=Path("inputs/matched_baselines"))
    parser.add_argument("--phase",choices=["train","evaluate","smoke"],required=True)
    parser.add_argument("--train-target",choices=[f"{t}_{m}" for t in ("lift","can") for m in ("wm","cfm","ddim","bc")])
    parser.add_argument("--workers",type=int,default=min(4,max(1,os.cpu_count() or 2)))
    parser.add_argument("--cell",help="Optional task:method:duration cell for checkpointed evaluation")
    parser.add_argument("--task",choices=["lift","can"],help="Partition workload without changing the frozen grid")
    args = parser.parse_args()
    output,inputs = args.output.resolve(),args.inputs.resolve()
    data = Path(os.environ.get("ROBO_DATA","robomimic_datasets")).resolve()
    os.environ.update(ROBO_DATA=str(data),MPLBACKEND="Agg",
                      OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    os.environ.setdefault("MUJOCO_GL","egl")
    if os.environ["MUJOCO_GL"]!="disable":
        os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    protocol = json.loads(args.protocol.read_text())
    output.mkdir(parents=True,exist_ok=True)
    frozen = output/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Output directory contains another frozen protocol")
    atomic_json(frozen,protocol)
    if args.phase == "train":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA preflight is required for full training")
        training(output,inputs,data,protocol,args.train_target,args.task)
    elif args.phase == "smoke":
        smoke = json.loads(json.dumps(protocol))
        smoke["id"] += "_engineering_smoke"
        smoke["episodes"] = {task:1 for task in smoke["episodes"]}
        smoke["max_steps"] = {task:7 for task in smoke["max_steps"]}
        smoke["blackout_durations"] = [0,20]
        smoke["paired_tests"] = []
        # Keep checkpoints in the parent output and episode artifacts separate.
        references = {}
        for job in make_jobs(output,data,smoke,args.task):
            job["output"] = str(output/"smoke"/job["task"]/job["method"]/f"d{job['duration']}.json")
            result = evaluate_episode(job)
            initial = (result["initial_observation"],result["initial_simulator_state_sha256"])
            if job["task"] in references and references[job["task"]] != initial:
                raise RuntimeError("Engineering smoke failed exact simulator-state pairing")
            references[job["task"]] = initial
        marker = f"smoke_result_{args.task}.json" if args.task else "smoke_result.json"
        atomic_json(output/marker,{"status":"completed","kind":"engineering_only"})
    else:
        selected_task = args.cell.split(":")[0] if args.cell else args.task
        smoke_markers = [output/f"smoke_result_{selected_task}.json"] if selected_task else [output/f"smoke_result_{t}.json" for t in protocol["methods"]]
        if not (output/"smoke_result.json").exists() and not all(p.exists() for p in smoke_markers):
            raise RuntimeError("Run and retrieve the seven-step engineering smoke first")
        evaluate(output,data,protocol,args.workers,args.cell,selected_task)
    print("MATCHED_PHASE_COMPLETE " + args.phase,flush=True)


if __name__ == "__main__":
    main()
