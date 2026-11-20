import argparse,json
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from omegaconf import OmegaConf
from screen_wm_robustness import load_data,mean_step,replay_peak
from audit_full_checkpoint_imitation import WorldModel,metric_bundle
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.train.beliefs import previous_actions,compute_pre_action_beliefs
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.utils.checkpoint import load_checkpoint

def step(m,o,a,perturb,recovery):
 e=m.encoder(o);clean=m.rssm.observe(e,previous_actions(a));f=torch.cat([clean['h'],clean['z']],-1);base=F.mse_loss(m.obs_head(f)[:,6:],o[:,6:])+kl_normal(clean['post_mean'],clean['post_std'],clean['prior_mean'],clean['prior_std']).sum(-1)[:,6:].clamp_min(1).mean()
 if not perturb:return base
 s={'h':clean['h'][:,5].detach(),'z':clean['post_mean'][:,5].detach()};loss=0.
 for t in range(6,10):
  on=(o[:,t]+torch.randn_like(o[:,t])*.25).clamp(-3,3);an=(a[:,t-1]+torch.randn_like(a[:,t-1])*.05).clamp(-1,1);h=m.rssm.gru(torch.cat([s['z'],an],-1),s['h']);mu,sd=m.rssm._get_stats(m.rssm.post_net(torch.cat([h,m.encoder(on)],-1)));prior_mu,prior_sd=m.rssm._get_stats(m.rssm.prior_net(h));loss=loss+F.mse_loss(m.obs_head(torch.cat([h,mu],-1)),o[:,t])+kl_normal(mu,sd,prior_mu,prior_sd).sum(-1).clamp_min(1).mean()
  if recovery:loss=loss+.1*F.mse_loss(h,clean['h'][:,t].detach())+.1*F.mse_loss(mu,clean['post_mean'][:,t].detach())
  s={'h':h,'z':mu}
 return base+loss/4
def reference_ranges(m,rows,mean,std,dev):
 vals={k:[] for k in ('encoder_norm','posterior_mean_norm','posterior_std_norm')}
 with torch.no_grad():
  for raw,a in rows:
   o=torch.tensor((raw-mean)/std,device=dev)[None];ac=torch.tensor(a,device=dev)[None];emb=m.encoder(o.reshape(-1,19)).reshape(1,len(a),-1);s={'h':torch.zeros(1,m.rssm.deter_dim,device=dev),'z':torch.zeros(1,m.rssm.stoch_dim,device=dev)};zero=torch.zeros_like(ac[:,0])
   for t in range(len(a)):
    s,mu=mean_step(m,emb[:,t],zero if t==0 else ac[:,t-1],s);_,sd=m.rssm._get_stats(m.rssm.post_net(torch.cat([s['h'],emb[:,t]],-1)));vals['encoder_norm'].append(float(emb[:,t].norm()));vals['posterior_mean_norm'].append(float(mu.norm()));vals['posterior_std_norm'].append(float(sd.norm()))
 return {k:(min(v),max(v)) for k,v in vals.items()}
def replay_diagnostics(m,trace,qs,mean,std,dev,ref):
 out=[]
 with torch.no_grad():
  for ep in trace['episodes']:
   r=ep['trajectory'];T=len(r);o=np.zeros((T,19),'float32');a=np.array([x['action'] for x in r],'float32')
   for t,x in enumerate(r):
    e=np.array(x['eef_pos'],'float32');p=np.array(x['object_pos'],'float32');o[t,:3]=e;o[t,3:7]=x['eef_quat'];o[t,7:9]=x['gripper_qpos'];o[t,9:12]=p;o[t,16:]=e-p
   x=np.repeat(o[None],len(qs),0);x[:,:,12:16]=qs[:,None];x=torch.tensor((x-mean)/std,device=dev);emb=m.encoder(x.reshape(-1,19)).reshape(len(qs),T,-1);act=torch.tensor(a,device=dev)[None].expand(len(qs),-1,-1);zero=torch.zeros(len(qs),7,device=dev);s={'h':torch.zeros(len(qs),m.rssm.deter_dim,device=dev),'z':torch.zeros(len(qs),m.rssm.stoch_dim,device=dev)};series={k:[] for k in ref}
   for t in range(T):
    s,mu=mean_step(m,emb[:,t],zero if t==0 else act[:,t-1],s);_,sd=m.rssm._get_stats(m.rssm.post_net(torch.cat([s['h'],emb[:,t]],-1)));series['encoder_norm'].append(float(emb[:,t].norm(dim=-1).mean()));series['posterior_mean_norm'].append(float(mu.norm(dim=-1).mean()));series['posterior_std_norm'].append(float(sd.norm(dim=-1).mean()))
   out.append({'posterior_mean_peak':max(series['posterior_mean_norm']),'first_above_demo_max':{k:next((i for i,v in enumerate(vs) if v>ref[k][1]),None) for k,vs in series.items()}})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--trace',required=True);p.add_argument('--wm',required=True);p.add_argument('--diff',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=1000);x=p.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.manual_seed(0);np.random.seed(0);rows,qs=load_data(x.dataset);ck=load_checkpoint(x.wm,map_location=dev);mean=np.array(ck['norm']['obs_mean'],'float32');std=np.array(ck['norm']['obs_std'],'float32');cfg=OmegaConf.load('src/blind_diffusion/configs/model/rssm_small.yaml');trace=json.load(open(x.trace));wins=[(o[i:i+50],a[i:i+50])for o,a in rows for i in range(len(a)-49)];held=wins[-128:];train=wins[:-128];dc=load_checkpoint(x.diff,map_location=dev)['config']['diffusion'];d=GaussianDiffusion(UNet1D(7,8,230,dc['base_ch']).to(dev),dc['timesteps'],dc['schedule'],prediction_type=dc['prediction_type']).to(dev);d.load_state_dict(load_checkpoint(x.diff,map_location=dev)['model']);d.eval();out={}
 for name,pert,rec in [('continuation',False,False),('perturbation_only_v1',True,False),('closed_loop_recovery_v1',True,True)]:
  torch.manual_seed(0);m=WorldModel(19,7,cfg).to(dev);m.load_state_dict(ck['model']);op=torch.optim.Adam(m.parameters(),lr=1e-4)
  for _ in range(x.steps):
   ii=np.random.randint(len(train),size=32);o=torch.tensor(np.stack([train[i][0]for i in ii]),device=dev);a=torch.tensor(np.stack([train[i][1]for i in ii]),device=dev);o=(o-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);l=step(m,o,a,pert,rec);op.zero_grad();l.backward();op.step()
  m.eval();o=torch.tensor(np.stack([z[0]for z in held]),device=dev);a=torch.tensor(np.stack([z[1]for z in held]),device=dev);o=(o-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);post=m.rssm.observe(m.encoder(o),previous_actions(a));r=float(F.mse_loss(m.obs_head(torch.cat([post['h'],post['z']],-1))[:,6:],o[:,6:]));b=compute_pre_action_beliefs(m,o,a);n=64;t=torch.arange(n,device=dev)%43;tar=torch.stack([a[i,t[i]:t[i]+8]for i in range(n)]).transpose(1,2);torch.manual_seed(123);pred=d.sample(tar.shape,torch.stack([b[i,t[i]]for i in range(n)]),sampler='ddim',inference_steps=10).transpose(1,2);ref=reference_ranges(m,rows,mean,std,dev);out[name]={'reconstruction_mse':r,'imitation':metric_bundle(pred.cpu().numpy(),tar.transpose(1,2).cpu().numpy()),'replay_peaks':replay_peak(m,trace,qs,mean,std,dev),'replay_diagnostics':replay_diagnostics(m,trace,qs,mean,std,dev,ref),'teacher_forced_reference_ranges':{k:{'min':v[0],'max':v[1]}for k,v in ref.items()}}
 Path(x.output).parent.mkdir(parents=True,exist_ok=True);Path(x.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
