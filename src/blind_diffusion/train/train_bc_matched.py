"""Reactive BC with the generative benchmark's split, query and update budgets."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from blind_diffusion.baselines.open_loop_bc import BCPolicy
from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.train.beliefs import sample_conditioning_queries
from blind_diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from blind_diffusion.utils.ema import ModelEMA
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.training_state import signature, make_state, restore_state, reconcile_log


def query_batch(batch, probability):
    """Match the state-query distribution; BC itself always trains on real obs."""
    starts, _ = sample_conditioning_queries(batch["valid"], batch["is_start"], 8, 4,
                                          probability=probability, warmup=5, max_duration=20)
    index = torch.arange(len(starts), device=starts.device)
    return batch["obs"][index, starts], batch["actions"][index, starts]


@torch.no_grad()
def validate(model, loader, device, seed):
    total, count = 0., 0
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed + 10000)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            for _ in range(4):
                obs, actions = query_batch(batch, 0.)
                total += float((model(obs) - actions).square().mean()) * len(obs)
                count += len(obs)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = load_checkpoint(args.world_model, map_location="cpu")
    obs_keys = wm["config"]["task"]["obs_keys"]
    train, val, split = build_episode_datasets(args.dataset, obs_keys, 32, 4, True, False, 8,
                                             norm_stats=wm["norm"], split=wm["split"])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sample = train[0]
    model = BCPolicy(sample["obs"].shape[-1], sample["actions"].shape[-1], 256).to(device)
    ema = ModelEMA(model, .995)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs * len(train_loader))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, steps, start_epoch = float("inf"), 0, 0
    training_signature=signature(vars(args),args.dataset,args.world_model)
    if args.resume_checkpoint:
        saved=load_checkpoint(args.resume_checkpoint,map_location=device)
        model.load_state_dict(saved["model"])
        ema.model.load_state_dict(saved["ema_model"])
        state=restore_state(saved,optim,scheduler,training_signature,output/"best.pt")
        steps,start_epoch,best,ema.updates=state["step"],state["epoch"],state["best_val"],state["ema_updates"]
        reconcile_log(output/"logs.jsonl",steps,start_epoch)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            obs, actions = query_batch(batch, .5)
            loss = (model(obs) - actions).square().mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optim.step()
            scheduler.step()
            ema.update(model)
            steps += 1
        value = validate(ema.model, val_loader, device, args.seed)
        row = {"epoch": epoch, "steps": steps, "val_mse": value, "last_train_mse": float(loss.detach())}
        with (output / "logs.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        payload = {"model": model.state_dict(), "ema_model": ema.model.state_dict(),
                   "kind": "reactive_bc", "hidden_dim": 256, "norm": wm["norm"], "split": split,
                   "obs_keys": obs_keys, "obs_dim": sample["obs"].shape[-1],
                   "action_dim": sample["actions"].shape[-1], "config": vars(args), "selection": row}
        if value < best:
            best = value
            save_checkpoint(str(output / "best.pt"), payload)
        payload["training_state"]=make_state(optim,scheduler,training_signature,output/"best.pt",
            step=steps,epoch=epoch+1,best_val=best,ema_updates=ema.updates)
        save_checkpoint(str(output/"last.pt"),payload)
    (output / "metrics.json").write_text(json.dumps({"best_val_mse": best, "updates": steps}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
