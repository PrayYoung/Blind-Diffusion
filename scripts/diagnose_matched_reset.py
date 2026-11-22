"""Small CPU-only reset probe with incremental evidence and visible progress."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import robomimic.utils.obs_utils as ObsUtils

from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.utils.seed import set_seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=3)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs":{"low_dim":[
        "robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"],"rgb":[]}})
    env=make_env(args.dataset)
    records=[]
    for seed in range(args.seeds):
        set_seed(seed)
        env.reset()
        current=env.get_state()
        attempts=[]
        for attempt in range(8):
            before=np.asarray(current["states"]).copy()
            tick=time.monotonic()
            env.reset_to(current)
            after=np.asarray(env.get_state()["states"]).copy()
            changed=np.flatnonzero(before!=after)
            row={"attempt":attempt,"changed_coordinates":changed.tolist(),
                 "max_abs_diff":float(np.max(np.abs(before-after))),
                 "examples":[{"index":int(i),"before":float(before[i]),"after":float(after[i])} for i in changed[:8]],
                 "seconds":time.monotonic()-tick}
            attempts.append(row)
            snapshot={"completed_seeds":records,"current_seed":seed,"current_attempts":attempts}
            temporary=args.output.with_suffix(".incoming.json")
            temporary.write_text(json.dumps(snapshot,indent=2)+"\n")
            temporary.replace(args.output)
            print("RESET_PROGRESS "+json.dumps({"seed":seed,**row}),flush=True)
            if not len(changed):
                break
            current={**current,"states":after}
        records.append({"seed":seed,"attempts":attempts})
    args.output.write_text(json.dumps({"status":"completed","seeds":records},indent=2)+"\n")
    print("RESET_DIAGNOSTIC_COMPLETE",flush=True)


if __name__=="__main__":
    main()
