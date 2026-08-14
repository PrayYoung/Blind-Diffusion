import json
import tempfile
import unittest
from pathlib import Path

import h5py

from blind_diffusion.env.robomimic_env import _build_robosuite_kwargs, _load_env_meta


class EnvMetadataTests(unittest.TestCase):
    def test_load_env_meta_reads_env_args_json(self):
        env_args = {
            "env_name": "Lift",
            "env_version": "1.4.1",
            "type": 1,
            "env_kwargs": {
                "robots": ["Panda"],
                "control_freq": 20,
                "use_object_obs": True,
                "use_camera_obs": False,
                "reward_shaping": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.hdf5"
            with h5py.File(path, "w") as h5:
                data = h5.create_group("data")
                data.attrs["env_args"] = json.dumps(env_args)
            loaded = _load_env_meta(str(path))
        self.assertEqual(loaded, env_args)

    def test_build_robosuite_kwargs_preserves_lowdim_protocol(self):
        env_meta = {
            "env_name": "Lift",
            "env_kwargs": {
                "robots": ["Panda"],
                "control_freq": 20,
                "use_object_obs": True,
                "use_camera_obs": False,
                "reward_shaping": False,
            },
        }
        env_name, env_kwargs = _build_robosuite_kwargs(env_meta)
        self.assertEqual(env_name, "Lift")
        self.assertEqual(env_kwargs["robots"], ["Panda"])
        self.assertEqual(env_kwargs["control_freq"], 20)
        self.assertFalse(env_kwargs["use_camera_obs"])
        self.assertFalse(env_kwargs["has_offscreen_renderer"])
        self.assertFalse(env_kwargs["has_renderer"])

    def test_build_robosuite_kwargs_enables_image_obs(self):
        env_meta = {
            "env_name": "Lift",
            "env_kwargs": {
                "robots": ["Panda"],
                "control_freq": 20,
                "use_object_obs": True,
                "use_camera_obs": False,
                "reward_shaping": False,
            },
        }
        _env_name, env_kwargs = _build_robosuite_kwargs(
            env_meta,
            image_keys=["agentview_image"],
            image_size=96,
        )
        self.assertTrue(env_kwargs["use_camera_obs"])
        self.assertTrue(env_kwargs["has_offscreen_renderer"])
        self.assertEqual(env_kwargs["camera_names"], ["agentview"])
        self.assertEqual(env_kwargs["camera_heights"], 96)
        self.assertEqual(env_kwargs["camera_widths"], 96)


if __name__ == "__main__":
    unittest.main()
