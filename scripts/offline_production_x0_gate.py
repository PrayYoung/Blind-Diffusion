"""Established causal held-out imitation gate using production diffusion code."""
from offline_x0_imitation_gate import SmallWorldModel
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.train.beliefs import compute_pre_action_beliefs


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output",required=True);p.add_argument("--wm-steps",type=int,default=300);p.add_argument("--diff-steps",type=int,default=1000);a=p.parse_args()
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8);keys=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
    with h5py.File(a.dataset,"r") as h5:
        names=sorted(h5["data"].keys());trn,held_names=names[:160],names[160:]
        def rows(names):
            out=[]
            for name in names:
                g=h5["data"][name]
                for start in range(g["actions"].shape[0]-49):out.append((np.concatenate([g["obs"][k][start:start+50] for k in keys],-1),g["actions"][start:start+50],g["rewards"][start:start+50],g["dones"][start:start+50]))
            return out
        train,held=rows(trn),rows(held_names)
    def tensors(rows):return tuple(torch.tensor(np.stack([r[i] for r in rows]),dtype=torch.float32) for i in range(4))
    to,ta,tr,td=tensors(train);ho,ha,_,_=tensors(held);mean,std=to.reshape(-1,19).mean(0),to.reshape(-1,19).std(0).clamp_min(1e-6);to,ho=(to-mean)/std,(ho-mean)/std
    wm=SmallWorldModel();wo=torch.optim.Adam(wm.parameters(),lr=3e-4)
    for _ in range(a.wm_steps):
        ids=torch.randint(len(train),(32,));post,op,rp,dp=wm(to[ids],ta[ids]);kl=kl_normal(post["post_mean"],post["post_std"],post["prior_mean"],post["prior_std"]).sum(-1).clamp_min(1.)
        loss=F.mse_loss(op[:,6:],to[ids,6:])+F.mse_loss(rp[:,6:],tr[ids,5:-1])+F.binary_cross_entropy_with_logits(dp[:,6:],td[ids,5:-1])+kl[:,6:].mean();wo.zero_grad();loss.backward();wo.step()
    wm.eval();[setattr(v,"requires_grad",False) for v in wm.parameters()]
    diffusion=GaussianDiffusion(UNet1D(7,8,112,32),100,"cosine",prediction_type="x0");opt=torch.optim.Adam(diffusion.parameters(),lr=1e-4)
    for _ in range(a.diff_steps):
        ids=torch.randint(len(train),(32,));t0=torch.randint(0,43,(32,))
        with torch.no_grad():b=compute_pre_action_beliefs(wm,to[ids],ta[ids])
        target=torch.stack([ta[ids[i],t0[i]:t0[i]+8] for i in range(32)]).transpose(1,2);c=torch.stack([b[i,t0[i]] for i in range(32)])
        loss=diffusion.loss(target,c);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        count=min(128,len(held));t0=torch.arange(count)%43;b=compute_pre_action_beliefs(wm,ho[:count],ha[:count]);target=torch.stack([ha[i,t0[i]:t0[i]+8] for i in range(count)]).transpose(1,2);c=torch.stack([b[i,t0[i]] for i in range(count)])
        torch.manual_seed(123);ddim=diffusion.sample(target.shape,c,sampler="ddim",inference_steps=10);torch.manual_seed(123);ddpm=diffusion.sample(target.shape,c,sampler="ddpm")
    dn,ddn,yn=ddim.transpose(1,2).numpy(),ddpm.transpose(1,2).numpy(),target.transpose(1,2).numpy();corr=[float(np.corrcoef(dn[...,i].ravel(),yn[...,i].ravel())[0,1]) for i in range(7)];mse=float(np.mean((dn-yn)**2));mc=float(np.nanmean(corr));am=float(np.abs(dn).max());ddpm_mse=float(np.mean((ddn-yn)**2));ddpm_am=float(np.abs(ddn).max());out={"wm_steps":a.wm_steps,"diffusion_steps":a.diff_steps,"prediction_type":diffusion.prediction_type,"heldout_demos":len(held_names),"heldout_chunks":count,"ddim10_mse":mse,"ddpm100_mse":ddpm_mse,"mean_correlation":mc,"correlations":corr,"ddim_absmax":am,"ddpm_absmax":ddpm_am,"gate":{"max_mse":.45,"min_correlation":.1,"max_abs_action":1.000001},"gate_pass":mse<=.45 and mc>=.1 and am<=1.000001 and ddpm_am<=1.000001};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
