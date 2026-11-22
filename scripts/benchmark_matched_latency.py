"""Batch-one latency on explicitly labeled CPU/GPU, excluding simulation."""
import argparse
import json
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import torch

from blind_diffusion.eval.matched_baselines import MatchedAgent
from blind_diffusion.utils.checkpoint import load_checkpoint


def measure(agent, observation, blind, count, warmup):
    samples = []
    for i in range(warmup + count):
        if agent.method == "open_loop":
            agent.plan = None
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        agent.action(observation, blind, 0, i).cpu()
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            samples.append((time.perf_counter()-started)*1000)
    result = {"median_ms":float(np.median(samples)),"p95_ms":float(np.percentile(samples,95)),"queries":count}
    if agent.method == "open_loop":
        amortized = []
        for chunk in range(3 + max(1,count//8)):
            agent.plan = None
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(8):
                agent.action(observation,blind,0,chunk*8+step).cpu()
            if agent.device.type == "cuda":
                torch.cuda.synchronize()
            if chunk >= 3:
                amortized.append((time.perf_counter()-started)*1000/8)
        result["chunk_amortized_step_median_ms"] = float(np.median(amortized))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--task",choices=["lift","can"])
    args=parser.parse_args()
    protocol=json.loads((args.root/"protocol.json").read_text())
    torch.set_num_threads(1)
    devices=["cpu"]+(["cuda"] if torch.cuda.is_available() else [])
    results={"hardware":{"cpu_arch":platform.machine(),"torch":torch.__version__,"cpu_threads":1,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"scope":protocol["latency"],"results":{}}
    for task,methods in protocol["methods"].items():
        if args.task and task != args.task:
            continue
        wm=args.root/task/"wm/checkpoints/best.pt"
        checkpoint=load_checkpoint(str(wm),map_location="cpu")
        with h5py.File(args.data/task/"low_dim_v141.hdf5","r") as handle:
            demo=handle["data"][checkpoint["split"]["val"][0]]
            observation=np.concatenate([demo["obs"][key][0] for key in checkpoint["config"]["task"]["obs_keys"]]).astype(np.float32)
        results["results"][task]={}
        for method in methods:
            name="cfm" if method.startswith("cfm") else "bc" if method=="bc" else "ddim"
            policy=args.root/task/("bc/best.pt" if method=="bc" else f"{name}/checkpoints/best.pt")
            results["results"][task][method]={}
            for device in devices:
                agent=MatchedAgent(method,str(wm),str(policy) if method!="mpc" else None,device)
                count=protocol["latency"]["mpc_measured_queries" if method=="mpc" else "measured_queries"]
                values={}
                for blind in (False,True):
                    agent.reset()
                    values["blind" if blind else "visible"]=measure(agent,observation,blind,count,protocol["latency"]["warmup_queries"])
                results["results"][task][method][device]=values
                print(f"MATCHED_LATENCY {task} {method} {device} "+json.dumps(values),flush=True)
                del agent
                if device=="cuda":
                    torch.cuda.empty_cache()
    filename=f"latency_{args.task}.json" if args.task else "latency.json"
    (args.root/filename).write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
