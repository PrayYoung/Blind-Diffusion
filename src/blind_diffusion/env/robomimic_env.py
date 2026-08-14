import json
from dataclasses import dataclass

import h5py


def _load_env_meta(hdf5_path: str) -> dict:
    with h5py.File(hdf5_path, "r") as h5:
        raw = h5["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _build_robosuite_kwargs(env_meta: dict, image_keys=None, image_size=84) -> tuple[str, dict]:
    env_name = env_meta["env_name"]
    env_kwargs = dict(env_meta.get("env_kwargs", {}))

    # Older robomimic datasets were collected with robosuite <= 1.4.1. Keep
    # robosuite in compatibility mode instead of using the newer lite physics path.
    env_kwargs.setdefault("lite_physics", False)

    if image_keys:
        camera_names = [k.replace("_image", "") for k in image_keys]
        env_kwargs["use_camera_obs"] = True
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["camera_names"] = camera_names
        env_kwargs["camera_heights"] = image_size
        env_kwargs["camera_widths"] = image_size
        env_kwargs.setdefault("camera_depths", False)
    else:
        env_kwargs["use_camera_obs"] = False
        env_kwargs["has_offscreen_renderer"] = False

    env_kwargs["has_renderer"] = False
    return env_name, env_kwargs


@dataclass
class _RobosuiteCompatEnv:
    env: object

    def __post_init__(self):
        action_spec = getattr(self.env, "action_spec")
        low, _high = action_spec
        self.action_dimension = int(low.shape[0])
        if hasattr(self.env, "horizon"):
            self.horizon = self.env.horizon
        if hasattr(self.env, "horizon"):
            self._max_episode_steps = self.env.horizon

    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, info
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated or truncated), info
        raise ValueError(f"Unexpected robosuite step return length: {len(result)}")


def make_env(hdf5_path: str, image_keys=None, image_size=84):
    import robosuite as suite

    env_meta = _load_env_meta(hdf5_path)
    env_name, env_kwargs = _build_robosuite_kwargs(env_meta, image_keys=image_keys, image_size=image_size)
    env = suite.make(env_name, **env_kwargs)
    return _RobosuiteCompatEnv(env)
