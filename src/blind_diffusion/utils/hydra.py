import argparse
from omegaconf import OmegaConf


def parse_config() -> OmegaConf:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg
