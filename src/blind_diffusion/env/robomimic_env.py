import json
import sys
import types

import h5py


def _load_env_meta(hdf5_path: str) -> dict:
    with h5py.File(hdf5_path, "r") as h5:
        raw = h5["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _install_mujoco_py_stub() -> None:
    if "mujoco_py" in sys.modules:
        return

    class MujocoException(Exception):
        pass

    module = types.ModuleType("mujoco_py")
    module.builder = types.SimpleNamespace(MujocoException=MujocoException)
    sys.modules["mujoco_py"] = module


def make_env(hdf5_path: str, image_keys=None, image_size=84):
    _install_mujoco_py_stub()
    from robomimic.utils.env_utils import create_env_from_metadata

    env_meta = _load_env_meta(hdf5_path)
    env_meta.setdefault("env_kwargs", {})

    use_image = False
    if image_keys:
        use_image = True
        camera_names = [k.replace("_image", "") for k in image_keys]
        env_meta["env_kwargs"]["camera_names"] = camera_names
        env_meta["env_kwargs"]["camera_heights"] = image_size
        env_meta["env_kwargs"]["camera_widths"] = image_size

    return create_env_from_metadata(
        env_meta,
        use_image_obs=use_image,
        render_offscreen=use_image,
    )
