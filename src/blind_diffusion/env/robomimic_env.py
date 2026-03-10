import robomimic.utils.file_utils as FileUtils
from robomimic.utils.env_utils import create_env_from_metadata


def make_env(hdf5_path: str, image_keys=None, image_size=84):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["controller_configs"] = None

    use_image = None
    if image_keys:
        use_image = True
        camera_names = [k.replace("_image", "") for k in image_keys]
        env_meta["env_kwargs"]["camera_names"] = camera_names
        env_meta["env_kwargs"]["camera_heights"] = image_size
        env_meta["env_kwargs"]["camera_widths"] = image_size
    env = create_env_from_metadata(
        env_meta,
        use_image_obj=use_image,
        render_offscreen=use_image)
    return env
