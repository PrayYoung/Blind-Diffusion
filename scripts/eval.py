#!/usr/bin/env python
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    mode = None
    for o in args.overrides:
        if o.startswith("eval.mode="):
            mode = o.split("=", 1)[1]
            break

    if mode is None or mode in {"mpc", "bc_train", "bc_eval", "random", "open_loop"}:
        from blind_diffusion.eval.evaluate_lowdim import main as eval_lowdim
        eval_lowdim()
    elif mode in {"rhc", "image_rhc", "image_open_loop", "image_bc_eval"}:
        from blind_diffusion.eval.evaluate_image import main as eval_image
        eval_image()
    else:
        raise ValueError(f"Unknown eval.mode: {mode}")


if __name__ == "__main__":
    main()
