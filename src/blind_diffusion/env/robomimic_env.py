import robomimic.utils.file_utils as FileUtils
from robomimic.utils.env_utils import create_env_from_metadata


def make_env(hdf5_path: str):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["controller_configs"] = None
    env = create_env_from_metadata(env_meta)
    return env
