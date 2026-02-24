#!/usr/bin/env python
from blind_diffusion.baselines.open_loop_bc_image import train_bc_image
from blind_diffusion.utils.hydra import parse_config
from omegaconf import OmegaConf
import os


def main():
    cfg = parse_config()
    task_cfg = OmegaConf.load(os.path.join("configs/task", f"{cfg.task}.yaml"))
    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    train_bc_image(cfg, task_cfg, run_dir)


if __name__ == "__main__":
    main()
