import os
from robomimic.utils.env_utils import create_env_from_metadata
import h5py
import json


def make_env(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        env_meta = f["data"].attrs.get("env_args")
        if isinstance(env_meta, (bytes, str)):
            env_meta = json.loads(env_meta)
    env = create_env_from_metadata(env_meta)
    return env
