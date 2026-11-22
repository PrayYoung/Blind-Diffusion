"""Derive deadline scores from first-success lengths of the same trajectories."""
import argparse
import json
from pathlib import Path

from collect_matched_results import collect, wilson


def collect_horizons(root,protocol,reference=None):
    base=collect(root,protocol)
    thresholds=protocol["report_horizons"]
    n=protocol["episodes"]["can"]
    tables={str(h):{} for h in thresholds}
    episodes=[]
    mismatches=[]
    compared=0
    for method in protocol["methods"]["can"]:
        for h in thresholds:
            tables[str(h)][method]={}
        for duration in protocol["blackout_durations"]:
            rows=[]
            for path in sorted((root/"episodes/can"/method/f"d{duration}").glob("seed*.json")):
                row=json.loads(path.read_text())
                first=row["length"] if row["success"] else None
                if first is not None and not 1<=first<=protocol["max_steps"]["can"]:
                    raise ValueError("Successful trajectory has an invalid first-success step")
                derived={"method":method,"duration":duration,"seed":row["seed"],"first_success_step":first,
                    "success_by_step":{str(h):first is not None and first<=h for h in thresholds}}
                episodes.append(derived)
                rows.append(derived)
                if reference:
                    old=reference/"episodes/can"/method/f"d{duration}"/path.name
                    if old.exists():
                        previous=json.loads(old.read_text())
                        compared+=1
                        actual=first is not None and first<=115
                        if actual!=previous["success"] or (actual and first!=previous["length"]):
                            mismatches.append({"method":method,"duration":duration,"seed":row["seed"],
                                "v1_success":previous["success"],"v1_length":previous["length"],
                                "v2_first_success_step":first})
            for h in thresholds:
                successes=sum(row["success_by_step"][str(h)] for row in rows)
                tables[str(h)][method][str(duration)]={"completed":len(rows),"expected":n,
                    "successes":successes,"complete":len(rows)==n,"rate":successes/n if len(rows)==n else None,
                    "wilson95":wilson(successes,n) if len(rows)==n else None}
    summary={"id":protocol["id"],"complete":base["complete"],"primary_horizon":protocol["primary_horizon"],
        "results_by_horizon":tables,"episodes":episodes,"primary_paired_tests":base["paired_tests"],
        "v1_prefix_comparison":{"compared":compared,"mismatches":mismatches}}
    lines=["# Can: fixed checkpoints, longer-horizon comparison","",
        "30 paired seeds0–29; one training seed; all deadlines scored from each same trajectory. No retraining.",""]
    for h in thresholds:
        lines += [f"## Success by step {h}"+(" (primary)" if h==protocol["primary_horizon"] else " (secondary)"),"",
            "| Method | "+" | ".join(f"D={d}" for d in protocol["blackout_durations"])+" |",
            "| --- | "+" | ".join("---:" for _ in protocol["blackout_durations"])+" |"]
        for method,ds in tables[str(h)].items():
            values=[f"{ds[str(d)]['successes']}/{n} ({100*ds[str(d)]['rate']:.1f}%)" if ds[str(d)]["complete"]
                    else f"Pending ({ds[str(d)]['completed']}/{n})" for d in protocol["blackout_durations"]]
            lines.append("| "+method+" | "+" | ".join(values)+" |")
        lines += [""]
    lines += [f"Archived v1 prefix checks: {compared}; mismatches: {len(mismatches)}.",
        "The original115-step benchmark is preserved separately. This run does not reproduce the external reference's different seeds/checkpoints.",""]
    for name,value in (("horizon_results.json",json.dumps(summary,indent=2,allow_nan=False)+"\n"),
                       ("horizon_results.md","\n".join(lines))):
        path=root/name
        temporary=path.with_suffix(".incoming")
        temporary.write_text(value)
        temporary.replace(path)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    collect_horizons(args.root,json.loads(args.protocol.read_text()),args.reference)
