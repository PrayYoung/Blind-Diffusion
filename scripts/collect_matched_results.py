"""Verify complete paired records and write a table, never filling missing cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z*z/count
    center = (p + z*z/(2*count)) / divisor
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / divisor
    return [max(0., center-half), min(1., center+half)]


def collect(root, protocol):
    output, records, reference = {}, {}, {}
    protocol_hash = hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    world_models, policy_families, engines = {}, {}, set()
    for task, methods in protocol["methods"].items():
        output[task] = {}
        n = protocol["episodes"][task]
        for method in methods:
            output[task][method] = {}
            for duration in protocol["blackout_durations"]:
                rows = []
                for seed in range(n):
                    path = root / "episodes" / task / method / f"d{duration}" / f"seed{seed}.json"
                    if not path.exists():
                        continue
                    row = json.loads(path.read_text())
                    if row["protocol_sha256"] != protocol_hash:
                        raise ValueError("Episode belongs to a different frozen protocol")
                    if (row["task"], row["method"], row["duration"], row["seed"]) != (task, method, duration, seed):
                        raise ValueError(f"Episode identity mismatch: {path}")
                    pair = (row["initial_observation"], row["initial_simulator_state_sha256"], row["dataset_sha256"])
                    key = (task, seed)
                    if key in reference and reference[key] != pair:
                        raise ValueError(f"Pairing mismatch for {task} seed{seed}; comparison invalid")
                    reference[key] = pair
                    wm_hash = row["provenance"]["world_model_sha256"]
                    if task in world_models and world_models[task] != wm_hash:
                        raise ValueError("Methods do not share the same world model")
                    world_models[task] = wm_hash
                    engines.add(row["provenance"].get("evaluation_engine_sha256","unknown"))
                    if len(engines)>1:
                        raise ValueError("Mixed evaluation engine versions")
                    family="cfm" if method.startswith("cfm") else "ddim" if method in ("ddim10","open_loop") else method
                    identity=row["provenance"].get("policy_sha256")
                    family_key=(task,family)
                    if family_key in policy_families and policy_families[family_key] != identity:
                        raise ValueError("A shared policy family uses different checkpoints")
                    policy_families[family_key]=identity
                    rows.append(row)
                if rows and any(r["provenance"] != rows[0]["provenance"] or
                                r["protocol_sha256"] != rows[0]["protocol_sha256"] for r in rows):
                    raise ValueError("Mixed checkpoint/protocol identities inside an evaluation cell")
                records[(task, method, duration)] = {row["seed"]: row for row in rows}
                successes = sum(row["success"] for row in rows)
                output[task][method][str(duration)] = {"completed": len(rows), "expected": n,
                    "successes": successes, "complete": len(rows) == n,
                    "rate": successes/n if len(rows) == n else None,
                    "wilson95": wilson(successes, n) if len(rows) == n else None}
    tests = []
    for task, a, b, duration in protocol["paired_tests"]:
        left, right = records[(task,a,duration)], records[(task,b,duration)]
        if len(left) != protocol["episodes"][task] or len(right) != len(left):
            continue
        a_only = sum(left[s]["success"] and not right[s]["success"] for s in left)
        b_only = sum(right[s]["success"] and not left[s]["success"] for s in left)
        discordant = a_only + b_only
        p = min(1., 2 * sum(math.comb(discordant,k) for k in range(min(a_only,b_only)+1)) / 2**discordant) if discordant else 1.
        tests.append({"task":task,"methods":[a,b],"duration":duration,"a_only":a_only,"b_only":b_only,"mcnemar_exact_p":p})
    # Adjust only when the entire predeclared family is available.
    if len(tests) == len(protocol["paired_tests"]):
        previous = 0.
        for rank, row in enumerate(sorted(tests,key=lambda r:r["mcnemar_exact_p"])):
            previous = min(1.,max(previous,(len(tests)-rank)*row["mcnemar_exact_p"]))
            row["holm_adjusted_p"] = previous
    summary = {"id": protocol["id"], "results": output, "paired_tests": tests,
               "pairing_verified_for_available_records": True,
               "complete": all(cell["complete"] for methods in output.values() for durations in methods.values() for cell in durations.values())}
    lines = ["# Matched baseline comparison", "", "One training seed. Success at any time within the task horizon. Intervals: Wilson 95%.", ""]
    labels = {"cfm4":"OT-path CFM, Euler4 + prior", "cfm2":"OT-path CFM, Euler2 + prior", "ddim10":"Direct-x0 DDIM10 + prior", "open_loop":"DDIM10, H8 chunks, zero-filled observations", "bc":"Reactive BC, zero-filled observations", "mpc":"Latent CEM, learned sparse reward"}
    for task, methods in output.items():
        lines += [f"## {task.capitalize()} (N={protocol['episodes'][task]}, max steps={protocol['max_steps'][task]})", "",
                  "| Method | " + " | ".join(f"D={d}" for d in protocol["blackout_durations"]) + " |",
                  "| --- | " + " | ".join("---:" for _ in protocol["blackout_durations"]) + " |"]
        for method, durations in methods.items():
            values = []
            for duration in protocol["blackout_durations"]:
                cell = durations[str(duration)]
                if cell["complete"]:
                    lo,hi=cell["wilson95"]
                    values.append(f"{cell['successes']}/{cell['expected']} ({100*cell['rate']:.1f}%; {100*lo:.1f}–{100*hi:.1f})")
                else:
                    values.append(f"Pending ({cell['completed']}/{cell['expected']} evaluated)")
            lines.append("| " + labels[method] + " | " + " | ".join(values) + " |")
        lines.append("")
        latency_path=root/f"latency_{task}.json"
        if latency_path.exists():
            latency=json.loads(latency_path.read_text())
            lines += ["Latency: batch-one observation-to-action computation, no simulation; CPU uses one thread.",
                      f"GPU: {latency['hardware']['gpu']}; PyTorch: {latency['hardware']['torch']}.","",
                      "| Method | CPU visible median / p95 (ms) | GPU visible median / p95 (ms) |",
                      "| --- | ---: | ---: |"]
            for method, devices in latency["results"][task].items():
                values=[]
                for device in ("cpu","cuda"):
                    if device not in devices:
                        values.append("Not measured")
                    else:
                        row=devices[device]["visible"]
                        values.append(f"{row['median_ms']:.2f} / {row['p95_ms']:.2f}")
                lines.append("| "+labels[method]+" | "+" | ".join(values)+" |")
            lines += ["", "Open-loop latency above is per plan query; its measured chunk-amortized per-step cost and blind-path timings are retained in the latency JSON.",""]
    lines += ["D is the number of blind steps per 20-step cycle after five visible observations; D=20 is permanent blackout.",
              "These are explicitly defined implementation baselines, not an audited reproduction of external results.", ""]
    (root / "results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (root / "results.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=Path("scripts/matched_protocol.json"))
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    print(json.dumps(collect(args.root,json.loads(args.protocol.read_text()))))
