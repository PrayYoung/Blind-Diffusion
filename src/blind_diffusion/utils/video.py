import os
import imageio.v2 as imageio


def save_video(frames, path, fps=30):
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)
