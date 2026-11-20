"""Small WM-only adaptation screen on collected failed-policy transitions."""
import argparse,json
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from omegaconf import OmegaConf
from screen_wm_robustness import load_data,mean_step,replay_peak
from audit_full_checkpoint_imitation import metric_bundle
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.train.beliefs import previous_actions,compute_pre_action_beliefs
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.utils.checkpoint import load_checkpoint

def wm_loss(m,o,a,r,d,teacher=None,anchor=0.):
 p,op,rp,dp=m(o,a); start=7
 loss=F.mse_loss(op[:,start:],o[:,start:])+F.mse_loss(rp[:,start:],r[:,6:-1].squeeze(-1))+F.binary_cross_entropy_with_logits(dp[:,start:],d[:,6:-1].squeeze(-1))+kl_normal(p['post_mean'],p['post_std'],p['prior_mean'],p['prior_std']).sum(-1)[:,start:].clamp_min(1).mean()
 if teacher is not None and anchor:
  with torch.no_grad(): tp,_,_,_=teacher(o,a)
  loss=loss+anchor*(F.mse_loss(p['h'][:,start:],tp['h'][:,start:])+F.mse_loss(p['post_mean'][:,start:],tp['post_mean'][:,start:]))
 return loss
def seqs(episodes,which):
 out=[]
 for i in which:
  tr=episodes[i]['trajectory'];o=np.array([x['observation'] for x in tr],'float32');a=np.array([x['action'] for x in tr],'float32');r=np.array([x['reward'] for x in tr],'float32')[:,None];d=np.array([x['done'] for x in tr],'float32')[:,None]
  out += [(o[j:j+50],a[j:j+50],r[j:j+50],d[j:j+50]) for j in range(len(a)-49)]
 return out
def trajectory_stats(m,episodes,mean,std,dev,seed=123):
 out=[]
 for ei,e in enumerate(episodes):
  tr=e['trajectory'];o=torch.tensor(np.array([x['observation'] for x in tr],'float32'),device=dev);a=torch.tensor(np.array([x['action'] for x in tr],'float32'),device=dev);en=m.encoder((o-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev));s={'h':torch.zeros(1,m.rssm.deter_dim,device=dev),'z':torch.zeros(1,m.rssm.stoch_dim,device=dev)};pm=[];emb=[]
  for t in range(len(tr)):
   s,mu=mean_step(m,en[t:t+1],torch.zeros_like(a[:1]) if t==0 else a[t-1:t],s);pm.append(float(mu.norm()));emb.append(float(en[t].norm()))
  torch.manual_seed(seed+ei);s={'h':torch.zeros(1,m.rssm.deter_dim,device=dev),'z':torch.zeros(1,m.rssm.stoch_dim,device=dev)};zs=[]
  for t in range(len(tr)):
   s=m.rssm.observe_step(en[t:t+1],torch.zeros_like(a[:1]) if t==0 else a[t-1:t],s);zs.append(float(s['z'].norm()))
  out.append({'posterior_mean_peak':max(pm),'sampled_z_peak':max(zs),'encoder_peak':max(emb)})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--offpolicy',required=True);p.add_argument('--wm',required=True);p.add_argument('--diff',required=True);p.add_argument('--trace',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=500);x=p.parse_args();dev=torch.device('cuda');rows,qs=load_data(x.dataset);ck=load_checkpoint(x.wm,map_location=dev);mean=np.array(ck['norm']['obs_mean'],'float32');std=np.array(ck['norm']['obs_std'],'float32');cfg=OmegaConf.load('src/blind_diffusion/configs/model/rssm_small.yaml');pol=json.load(open(x.offpolicy))['episodes'];off=seqs(pol,[0,1,2]);wins=[(o[i:i+50],a[i:i+50]) for o,a in rows for i in range(len(a)-49)];held=wins[-128:];train=wins[:-128];rng=np.random.RandomState(0);expert_ix=rng.randint(len(train),size=(x.steps,32));off_ix=rng.randint(len(off),size=(x.steps,32));dc=load_checkpoint(x.diff,map_location=dev)['config']['diffusion'];dif=GaussianDiffusion(UNet1D(7,8,230,dc['base_ch']).to(dev),dc['timesteps'],dc['schedule'],prediction_type=dc['prediction_type']).to(dev);dif.load_state_dict(load_checkpoint(x.diff,map_location=dev)['model']);dif.eval();trace=json.load(open(x.trace));res={}
 for name,off_every,anchor in [('continuation',0,0.),('policy_induced_7to1_v21',8,0.),('policy_induced_7to1_anchor_v21',8,1.)]:
  torch.manual_seed(0);m=WorldModel(19,7,cfg).to(dev);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
  teacher=WorldModel(19,7,cfg).to(dev);teacher.load_state_dict(ck['model']);teacher.eval()
  for k in range(x.steps):
   isoff=off_every and k%off_every==off_every-1;source=off if isoff else train;ix=off_ix[k] if isoff else expert_ix[k];q=[source[i] for i in ix];o=torch.tensor(np.stack([z[0] for z in q]),device=dev);a=torch.tensor(np.stack([z[1] for z in q]),device=dev);r=torch.tensor(np.stack([z[2] if len(z)>2 else np.zeros((50,1),'float32') for z in q]),device=dev);d=torch.tensor(np.stack([z[3] if len(z)>3 else np.zeros((50,1),'float32') for z in q]),device=dev);o=(o-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);l=wm_loss(m,o,a,r,d,teacher if (anchor and not isoff) else None,anchor);opt.zero_grad();l.backward();opt.step()
  m.eval();ho=torch.tensor(np.stack([z[0]for z in held]),device=dev);ha=torch.tensor(np.stack([z[1]for z in held]),device=dev);hn=(ho-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);torch.manual_seed(123);post=m.rssm.observe(m.encoder(hn),previous_actions(ha));torch.manual_seed(123);tp,_,_,_=teacher(hn,ha);rec=float(F.mse_loss(m.obs_head(torch.cat([post['h'],post['z']],-1))[:,6:],hn[:,6:]));b=compute_pre_action_beliefs(m,hn,ha);t=torch.arange(64,device=dev)%43;tar=torch.stack([ha[i,t[i]:t[i]+8]for i in range(64)]).transpose(1,2);torch.manual_seed(123);pred=dif.sample(tar.shape,torch.stack([b[i,t[i]]for i in range(64)]),sampler='ddim',inference_steps=10).transpose(1,2);res[name]={'expert_reconstruction_mse':rec,'expert_representation_drift':{'h_mse':float(F.mse_loss(post['h'],tp['h'])),'posterior_mean_mse':float(F.mse_loss(post['post_mean'],tp['post_mean']))},'expert_imitation':metric_bundle(pred.cpu().numpy(),tar.transpose(1,2).cpu().numpy()),'old_trace_posterior_mean_peaks':replay_peak(m,trace,qs,mean,std,dev),'policy_trajectory_stats':trajectory_stats(m,pol,mean,std,dev)}
 Path(x.output).parent.mkdir(parents=True,exist_ok=True);Path(x.output).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
