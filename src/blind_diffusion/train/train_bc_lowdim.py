import os
import hydra

from blind_diffusion.baselines.open_loop_bc import train_bc


@hydra.main(version_base=None, config_path="../configs", config_name="eval_mpc")
def main(cfg):
    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    train_bc(cfg, cfg.task, run_dir)


if __name__ == "__main__":
    main()
