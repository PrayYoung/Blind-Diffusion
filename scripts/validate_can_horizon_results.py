"""Validate the completed versioned comparison before interpreting its table."""
import argparse
import hashlib
import json
from pathlib import Path

from blind_diffusion.eval.matched_baselines import digest_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.root/"protocol.json").read_text())
    summary=json.loads((args.root/"horizon_results.json").read_text())
    rows=[json.loads(path.read_text()) for path in (args.root/"episodes/can").rglob("seed*.json")]
    expected={(m,d,s) for m in protocol["methods"]["can"] for d in protocol["blackout_durations"]
              for s in range(protocol["episodes"]["can"])}
    if not summary["complete"] or len(rows)!=600 or {(r["method"],r["duration"],r["seed"]) for r in rows}!=expected:
        raise ValueError("The full600episode grid is not complete")
    pairs={}
    old_hash_matches=0
    for row in rows:
        if row["protocol_sha256"]!=digest_json(protocol) or row["device"]!="cpu":
            raise ValueError("Unexpected protocol or inference device")
        method="cfm" if row["method"]=="cfm4" else "bc" if row["method"]=="bc" else "ddim"
        provenance=row["provenance"]
        if (provenance["world_model_sha256"]!=protocol["checkpoint_sha256"]["wm"]
                or provenance["policy_sha256"]!=protocol["checkpoint_sha256"][method]):
            raise ValueError("Evaluation checkpoint changed")
        pair=(row["initial_simulator_state_sha256"],row["initial_observation"])
        if row["seed"] in pairs and pairs[row["seed"]]!=pair:
            raise ValueError("Starting states or observations are not exactly paired")
        pairs[row["seed"]]=pair
        previous=json.loads((args.reference/"episodes/can"/row["method"]/f"d{row['duration']}"/f"seed{row['seed']}.json").read_text())
        old_hash_matches+=row["initial_simulator_state_sha256"]==previous["initial_simulator_state_sha256"]
    for name,expected_hash in protocol["checkpoint_sha256"].items():
        path=args.root/"can"/name/("best.pt" if name=="bc" else "checkpoints/best.pt")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            raise ValueError("Checkpoint bytes changed")
    for method in protocol["methods"]["can"]:
        for duration in protocol["blackout_durations"]:
            selected=[r for r in rows if (r["method"],r["duration"])==(method,duration)]
            counts=[]
            for horizon in protocol["report_horizons"]:
                actual=sum(r["success"] and r["length"]<=horizon for r in selected)
                reported=summary["results_by_horizon"][str(horizon)][method][str(duration)]
                if actual!=reported["successes"] or not reported["complete"]:
                    raise ValueError("Reported deadline counts disagree with raw trajectories")
                counts.append(actual)
            if counts!=sorted(counts):
                raise ValueError("Deadline success counts must be monotone")
    original=json.loads((args.reference/"results.json").read_text())
    if not original["complete"] or original["results"]["can"]["cfm4"]["0"]["successes"]!=7:
        raise ValueError("Original benchmark is missing or changed")
    result={"status":"passed","episodes":600,"exact_within_run_pairing":True,
        "checkpoints_unchanged":True,"same_trajectory_deadline_counts_verified":True,
        "v1_preserved":True,"v1_initial_state_hash_matches":old_hash_matches,
        "v1_prefix_checks":summary["v1_prefix_comparison"]["compared"],
        "v1_prefix_mismatches":len(summary["v1_prefix_comparison"]["mismatches"]),
        "review_limits":["One training seed;30 evaluation seeds","CPU/backend change can alter old115step prefixes",
            "Deadline effects are measured within the same new trajectories","No exact reproduction claim for external checkpoints/seeds"]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
