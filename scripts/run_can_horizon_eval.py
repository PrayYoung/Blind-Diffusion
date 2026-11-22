"""Evaluation-only v2; preserve selected weights and the archived v1 benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from blind_diffusion.eval.matched_baselines import file_digest
from blind_diffusion.eval.matched_baselines import worker_assets
from blind_diffusion.eval.start_reference import establish_host_observations
from collect_can_horizons import collect_horizons
from run_matched_campaign import atomic_json, checkpoint_paths, evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cpu-only",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("outputs/matched_baselines"))
    parser.add_argument("--data",type=Path,default=Path("robomimic_datasets"))
    parser.add_argument("--reference",type=Path,default=Path("inputs/can_v1_reference"))
    args=parser.parse_args()
    root=Path.cwd()
    os.environ.update(PYTHONPATH=str(root/"src")+os.pathsep+str(root/"scripts"),
        ROBO_DATA=str(args.data.resolve()),MPLBACKEND="Agg",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    if args.cpu_only:
        os.environ["MUJOCO_GL"]="disable"
        os.environ.pop("PYOPENGL_PLATFORM",None)
    else:
        os.environ.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl")
        subprocess.run([sys.executable,"scripts/cuda_preflight.py"],check=True)
    protocol=json.loads((root/"scripts/can_horizon_protocol_v2.json").read_text())
    output=args.output.resolve()
    frozen=output/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:
        raise ValueError("Output already belongs to another protocol")
    atomic_json(frozen,protocol)
    paths=checkpoint_paths(output,"can")
    for name,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol["checkpoint_sha256"][name]:
            raise ValueError("Selected checkpoint identity changed: "+name)
    subprocess.run([sys.executable,"scripts/validate_matched_inputs.py","--root",str(output),
        "--data",os.environ["ROBO_DATA"],"--task","can"],check=True)
    calibration=protocol.get("host_observation_calibration")
    if calibration:
        env,agent=worker_assets(str(Path(os.environ["ROBO_DATA"])/"can/low_dim_v141.hdf5"),
            "cfm4",str(paths["wm"]),str(paths["cfm"]),"cpu")
        audit=establish_host_observations(env,agent.keys,output/"initial_states/can",calibration["max_abs_delta"])
        atomic_json(output/"host_observation_audit.json",audit)
        print("HOST_OBSERVATION_AUDIT_PASSED "+str(len(audit["seeds"])),flush=True)
    if args.cpu_only:
        # RoboSuite changes MUJOCO_GL while importing its bindings. Restore
        # headless mode before starting fresh interpreter/worker processes.
        os.environ["MUJOCO_GL"]="disable"
        os.environ.pop("PYOPENGL_PLATFORM",None)
    subprocess.run([sys.executable,"scripts/run_matched_campaign.py","--protocol",
        "scripts/can_horizon_protocol_v2.json","--output",str(output),"--phase","smoke","--task","can"],check=True)
    for method in protocol["methods"]["can"]:
        for duration in protocol["blackout_durations"]:
            if args.cpu_only:
                os.environ["MUJOCO_GL"]="disable"
            evaluate(output,Path(os.environ["ROBO_DATA"]),protocol,workers=min(4,os.cpu_count() or 1),
                cell=f"can:{method}:{duration}",task_filter="can")
            summary=collect_horizons(output,protocol,args.reference.resolve())
            print("CAN_HORIZON_CELL "+json.dumps({"method":method,"duration":duration,
                "results":{h:rows[method][str(duration)] for h,rows in summary["results_by_horizon"].items()}}),flush=True)
    if not summary["complete"]:
        raise RuntimeError("The versioned600episode grid is incomplete")
    for name,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol["checkpoint_sha256"][name]:
            raise ValueError("Checkpoint changed during evaluation")
    print("CAN_HORIZON_V2_COMPLETE",flush=True)


if __name__=="__main__":
    main()
