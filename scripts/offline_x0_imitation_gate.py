#!/usr/bin/env python
"""Offline held-out imitation gate for causal beliefs plus x0 diffusion."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.heads import DoneHead, ObsHead, RewardHead
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.train.beliefs import compute_pre_action_beliefs


class SmallWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MLPEncoder(19, 64, 2); self.rssm = RSSM(7, 64, 96, 16, 96)
        self.obs_head = ObsHead(112, 19, 64); self.reward_head = RewardHead(112, 64); self.done_head = DoneHead(112, 64)
    def forward(self, obs, actions):
        previous = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
        post = self.rssm.observe(self.encoder(obs), previous); feat = torch.cat([post["h"], post["z"]], -1)
        return post, self.obs_head(feat), self.reward_head(feat), self.done_head(feat)


def ddim_x0(model, diffusion, cond, shape):
    x = torch.randn(shape); schedule = torch.linspace(99, 0, 10).round().long()
    for index, t_scalar in enumerate(schedule):
        t = torch.full((shape[0],), int(t_scalar), dtype=torch.long); alpha_t = diffusion.alphas_cumprod[t_scalar]
        alpha_prev = diffusion.alphas_cumprod[schedule[index + 1]] if index + 1 < len(schedule) else torch.ones(())
        x0 = model(x, t, cond); eps = (x - torch.sqrt(alpha_t) * x0) / torch.sqrt(1.0 - alpha_t)
        x = torch.sqrt(alpha_prev) * x0 + torch.sqrt((1.0 - alpha_prev).clamp_min(0.0)) * eps
    return x


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--output", required=True); p.add_argument("--wm-steps", type=int, default=300); p.add_argument("--diff-steps", type=int, default=1000); args = p.parse_args()
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    with h5py.File(args.dataset, "r") as h5:
        names = sorted(h5["data"].keys()); train_names, held_names = names[:160], names[160:]
        def rows(demos):
            out=[]
            for name in demos:
                group=h5["data"][name]
                for start in range(group["actions"].shape[0]-49):
                    out.append((np.concatenate([group["obs"][key][start:start+50] for key in keys], -1), group["actions"][start:start+50], group["rewards"][start:start+50], group["dones"][start:start+50]))
            return out
        train, held = rows(train_names), rows(held_names)
    def tensors(rows): return tuple(torch.tensor(np.stack([r[i] for r in rows]), dtype=torch.float32) for i in range(4))
    train_o, train_a, train_r, train_d = tensors(train); held_o, held_a, _, _ = tensors(held)
    mean, std = train_o.reshape(-1,19).mean(0), train_o.reshape(-1,19).std(0).clamp_min(1e-6)
    train_o, held_o = (train_o-mean)/std, (held_o-mean)/std
    wm=SmallWorldModel(); wm_opt=torch.optim.Adam(wm.parameters(),lr=3e-4)
    for _ in range(args.wm_steps):
        ids=torch.randint(len(train),(32,)); post,op,rp,dp=wm(train_o[ids],train_a[ids])
        kl=kl_normal(post["post_mean"],post["post_std"],post["prior_mean"],post["prior_std"]).sum(-1).clamp_min(1.0)
        loss=F.mse_loss(op[:,6:],train_o[ids,6:])+F.mse_loss(rp[:,6:],train_r[ids,5:-1])+F.binary_cross_entropy_with_logits(dp[:,6:],train_d[ids,5:-1])+kl[:,6:].mean()
        wm_opt.zero_grad();loss.backward();wm_opt.step()
    wm.eval()
    for v in wm.parameters(): v.requires_grad=False
    model=UNet1D(7,8,112,32); diffusion=GaussianDiffusion(model,100,"cosine"); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    for _ in range(args.diff_steps):
        ids=torch.randint(len(train),(32,)); t0=torch.randint(0,43,(32,))
        with torch.no_grad(): beliefs=compute_pre_action_beliefs(wm,train_o[ids],train_a[ids])
        target=torch.stack([train_a[ids[i],t0[i]:t0[i]+8] for i in range(32)]).transpose(1,2); cond=torch.stack([beliefs[i,t0[i]] for i in range(32)])
        t=torch.randint(0,100,(32,)); noise=torch.randn_like(target); xt=diffusion.q_sample(target,t,noise)
        loss=F.mse_loss(model(xt,t,cond),target); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        count=min(128,len(held)); t0=torch.arange(count)%43; beliefs=compute_pre_action_beliefs(wm,held_o[:count],held_a[:count])
        target=torch.stack([held_a[i,t0[i]:t0[i]+8] for i in range(count)]).transpose(1,2); cond=torch.stack([beliefs[i,t0[i]] for i in range(count)])
        torch.manual_seed(123); pred=ddim_x0(model,diffusion,cond,target.shape)
    p_np,y_np=pred.transpose(1,2).numpy(),target.transpose(1,2).numpy(); corrs=[]
    for dim in range(7):
        corrs.append(float(np.corrcoef(p_np[...,dim].ravel(),y_np[...,dim].ravel())[0,1]))
    mse=float(np.mean((p_np-y_np)**2)); corr=float(np.nanmean(corrs)); absmax=float(np.abs(p_np).max())
    result={"wm_steps":args.wm_steps,"diffusion_steps":args.diff_steps,"heldout_demos":len(held_names),"heldout_chunks":count,"ddim10_mse":mse,"mean_correlation":corr,"correlations":corrs,"sample_absmax":absmax,"gate":{"max_mse":0.45,"min_correlation":0.10,"max_abs_action":1.05},"gate_pass":mse<=0.45 and corr>=0.10 and absmax<=1.05}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__": main()
