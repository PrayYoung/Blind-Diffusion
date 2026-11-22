import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from blind_diffusion.eval.matched_baselines import digest_json


class HorizonMetricsTests(unittest.TestCase):
    def test_same_trajectory_deadlines_are_inclusive_and_missing_cells_stay_pending(self):
        scripts=Path(__file__).resolve().parents[1]/"scripts"
        with patch.object(sys,"path",[str(scripts),*sys.path]):
            collect=runpy.run_path(str(scripts/"collect_can_horizons.py"))["collect_horizons"]
        protocol={"id":"deadline_test","methods":{"can":["cfm4"]},"episodes":{"can":4},
            "max_steps":{"can":300},"blackout_durations":[0],"report_horizons":[115,250,300],
            "primary_horizon":300,"paired_tests":[]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/"new"
            records=output/"episodes/can/cfm4/d0"
            records.mkdir(parents=True)
            reference=root/"old"
            old_records=reference/"episodes/can/cfm4/d0"
            old_records.mkdir(parents=True)
            for seed,first in enumerate((115,250,300,None)):
                row={"task":"can","method":"cfm4","duration":0,"seed":seed,
                    "success":first is not None,"length":first or 300,"protocol_sha256":digest_json(protocol),
                    "dataset_sha256":"data","provenance":{"world_model_sha256":"wm",
                        "policy_sha256":"policy","evaluation_engine_sha256":"engine"},
                    "initial_observation":[seed],"initial_simulator_state_sha256":str(seed)}
                (records/f"seed{seed}.json").write_text(json.dumps(row))
                old={**row,"success":seed in (0,2),"length":115 if seed!=2 else 100}
                (old_records/f"seed{seed}.json").write_text(json.dumps(old))
            result=collect(output,protocol,reference)
            counts=[result["results_by_horizon"][str(h)]["cfm4"]["0"]["successes"] for h in (115,250,300)]
            self.assertEqual(counts,[1,2,3])
            self.assertIsNone(result["episodes"][-1]["first_success_step"])
            self.assertFalse(any(result["episodes"][-1]["success_by_step"].values()))
            self.assertEqual(result["v1_prefix_comparison"]["compared"],4)
            self.assertEqual(len(result["v1_prefix_comparison"]["mismatches"]),1)
            (records/"seed3.json").unlink()
            partial=collect(output,protocol)
            self.assertFalse(partial["complete"])
            self.assertIsNone(partial["results_by_horizon"]["300"]["cfm4"]["0"]["rate"])
