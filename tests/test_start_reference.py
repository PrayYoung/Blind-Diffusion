import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from blind_diffusion.eval.start_reference import establish_host_observations


class HostObservationTests(unittest.TestCase):
    def template(self,root):
        path=root/"seed0.json"
        value={"seed":0,"states":[1.],"restored_states":[1.],"restored_observation":[.5]}
        path.write_text(json.dumps(value))
        return path,value

    def test_fixed_physics_and_small_host_roundoff_keep_exact_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,value=self.template(root)
            observation=np.nextafter(np.float32(.5),np.float32(1.))
            env=SimpleNamespace(reset=lambda:None,reset_to=lambda state:{"obs":np.array([observation])},
                get_state=lambda:{"states":np.array([1.])})
            result=establish_host_observations(env,["obs"],root,5e-7)
            saved=json.loads(path.read_text())
            self.assertEqual(saved["states"],value["states"])
            self.assertEqual(saved["restored_states"],value["restored_states"])
            self.assertEqual(saved["source_v1_restored_observation"],value["restored_observation"])
            self.assertEqual(saved["restored_observation"],[float(observation)])
            self.assertTrue(result["within_host_pairing_requires_exact_equality"])
            establish_host_observations(env,["obs"],root,5e-7)
            self.assertEqual(json.loads(path.read_text())["source_v1_restored_observation"],value["restored_observation"])

    def test_physics_or_large_sensor_changes_do_not_rewrite_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_=self.template(root);before=path.read_bytes()
            env=SimpleNamespace(reset=lambda:None,reset_to=lambda state:{"obs":np.array([.51])},
                get_state=lambda:{"states":np.array([1.])})
            with self.assertRaisesRegex(ValueError,"exceeds declared bound"):
                establish_host_observations(env,["obs"],root,5e-7)
            self.assertEqual(path.read_bytes(),before)
            env.get_state=lambda:{"states":np.array([2.])}
            with self.assertRaisesRegex(ValueError,"physics changed"):
                establish_host_observations(env,["obs"],root,5e-7)
            self.assertEqual(path.read_bytes(),before)
