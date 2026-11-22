"""Audit host-derived sensor roundoff without changing canonical physics."""
import json
from pathlib import Path

import numpy as np

from blind_diffusion.eval.matched_baselines import vector
from blind_diffusion.utils.seed import set_seed


def establish_host_observations(env,keys,directory,max_abs_delta):
    prepared=[]
    report=[]
    for path in sorted(Path(directory).glob("seed*.json")):
        template=json.loads(path.read_text())
        restored={"states":np.asarray(template["states"],dtype=np.float64)}
        if template.get("model") is not None:
            restored["model"]=template["model"]
        captures=[]
        for _ in range(2):
            set_seed(template["seed"])
            env.reset()
            obs=env.reset_to(restored)
            state=np.asarray(env.get_state()["states"],dtype=np.float64)
            if not np.array_equal(state,np.asarray(template["restored_states"],dtype=np.float64)):
                raise ValueError(f"Canonical physics changed for seed{template['seed']}")
            captures.append(vector(obs,keys))
        if not np.array_equal(captures[0],captures[1]):
            raise ValueError(f"Observations do not repeat exactly on this host for seed{template['seed']}")
        original=template.get("source_v1_restored_observation",template["restored_observation"])
        previous=np.asarray(original,dtype=np.float32)
        delta=float(np.max(np.abs(captures[0]-previous)))
        if not np.isfinite(delta) or delta>max_abs_delta:
            raise ValueError(f"Cross-host observation delta{delta} exceeds declared bound for seed{template['seed']}")
        updated={**template,"source_v1_restored_observation":original,
            "restored_observation":captures[0].tolist(),"cross_host_observation_max_abs_delta":delta}
        prepared.append((path,updated))
        report.append({"seed":template["seed"],"physics_exact":True,"observations_repeat_exactly":True,
            "cross_host_max_abs_delta":delta,"changed_coordinates":np.flatnonzero(captures[0]!=previous).tolist()})
    if not prepared:
        raise ValueError("No archived canonical states were supplied")
    # Validate the complete set before changing any host-derived references.
    for path,updated in prepared:
        temporary=path.with_suffix(".incoming.json")
        temporary.write_text(json.dumps(updated,allow_nan=False)+"\n")
        temporary.replace(path)
    return {"status":"passed","seeds":report,"declared_max_abs_delta":max_abs_delta,
        "canonical_physics_unchanged":True,"within_host_pairing_requires_exact_equality":True}
