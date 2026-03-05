import os
import hydra

from blind_diffusion.baselines.open_loop_bc_image import train_bc_image


@hydra.main(version_base=None, config_path="../../configs", config_name="train_bc_image")
def main(cfg):
    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    train_bc_image(cfg, cfg.task, run_dir)


if __name__ == "__main__":
    main()
