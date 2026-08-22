import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py

from blind_diffusion.env.robomimic_env import _install_mujoco_py_stub, _load_env_meta, make_env


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
                "controller_configs": {"type": "OSC_POSE"},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.hdf5"
            with h5py.File(path, "w") as h5:
                data = h5.create_group("data")
                data.attrs["env_args"] = json.dumps(env_args)
            loaded = _load_env_meta(str(path))
        self.assertEqual(loaded, env_args)

    def test_install_mujoco_py_stub_exposes_builder_exception(self):
        with mock.patch.dict("sys.modules", {}, clear=False):
            _install_mujoco_py_stub()
            import mujoco_py  # type: ignore

            self.assertTrue(hasattr(mujoco_py, "builder"))
            self.assertTrue(issubclass(mujoco_py.builder.MujocoException, Exception))

    def test_make_env_uses_wrapper_creation_semantics(self):
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
                "controller_configs": {"type": "OSC_POSE"},
            },
        }
        fake_create_env = mock.Mock(return_value="fake-env")
        fake_env_utils = types.SimpleNamespace(create_env_from_metadata=fake_create_env)
        fake_utils = types.SimpleNamespace(env_utils=fake_env_utils)
        fake_robomimic = types.SimpleNamespace(utils=fake_utils)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.hdf5"
            with h5py.File(path, "w") as h5:
                data = h5.create_group("data")
                data.attrs["env_args"] = json.dumps(env_args)

            with mock.patch.dict(
                "sys.modules",
                {
                    "robomimic": fake_robomimic,
                    "robomimic.utils": fake_utils,
                    "robomimic.utils.env_utils": fake_env_utils,
                },
                clear=False,
            ):
                env = make_env(str(path), image_keys=["agentview_image"], image_size=96)

        self.assertEqual(env, "fake-env")
        fake_create_env.assert_called_once()
        call_args = fake_create_env.call_args
        env_meta = call_args.args[0]
        self.assertEqual(env_meta["env_name"], "Lift")
        self.assertEqual(env_meta["env_kwargs"]["controller_configs"], {"type": "OSC_POSE"})
        self.assertEqual(env_meta["env_kwargs"]["camera_names"], ["agentview"])
        self.assertEqual(env_meta["env_kwargs"]["camera_heights"], 96)
        self.assertEqual(env_meta["env_kwargs"]["camera_widths"], 96)
        self.assertTrue(call_args.kwargs["use_image_obs"])
        self.assertTrue(call_args.kwargs["render_offscreen"])


if __name__ == "__main__":
    unittest.main()
