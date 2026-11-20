"""Small, no-simulator robustness screen for frozen full WM checkpoints."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,torch,torch.nn.functional as F
from omegaconf import OmegaConf
from audit_full_checkpoint_imitation import WorldModel, OBS_KEYS, metric_bundle
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.train.beliefs import previous_actions, compute_pre_action_beliefs
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.utils.checkpoint import load_checkpoint

def mean_step(m,e,a,s):
 h=m.rssm.gru(torch.cat([s['z'],a],-1),s['h']);mean,std=m.rssm._get_stats(m.rssm.post_net(torch.cat([h,e],-1)));return {'h':h,'z':mean},mean
def load_data(path):
 with h5py.File(path) as h:
  rows=[];qs=[]
  for n in sorted(h['data']):
   g=h['data'][n];o=np.concatenate([g['obs'][k][:].astype('float32') for k in OBS_KEYS],-1);a=g['actions'][:].astype('float32');rows.append((o,a));qs.append(o[0,12:16])
 return rows,np.stack(qs)
def replay_peak(m,trace,qs,mean,std,device):
 out=[];hd=m.rssm.deter_dim;zd=m.rssm.stoch_dim
 for ep in trace['episodes']:
  r=ep['trajectory'];T=len(r);o=np.zeros((T,19),'float32');a=np.array([x['action'] for x in r],'float32')
  for t,x in enumerate(r):
   e=np.array(x['eef_pos'],'float32');p=np.array(x['object_pos'],'float32');o[t,:3]=e;o[t,3:7]=x['eef_quat'];o[t,7:9]=x['gripper_qpos'];o[t,9:12]=p;o[t,16:]=e-p
  x=np.repeat(o[None],len(qs),0);x[:,:,12:16]=qs[:,None];x=torch.tensor((x-mean)/std,device=device);emb=m.encoder(x.reshape(-1,19)).reshape(len(qs),T,-1);act=torch.tensor(a,device=device)[None].expand(len(qs),-1,-1);z=torch.zeros(len(qs),7,device=device);s={'h':torch.zeros(len(qs),hd,device=device),'z':torch.zeros(len(qs),zd,device=device)};peaks=[]
  for t in range(T):s,pm=mean_step(m,emb[:,t],z if t==0 else act[:,t-1],s);peaks.append(pm.norm(dim=-1).mean().item())
  out.append(float(max(peaks)))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--trace',required=True);p.add_argument('--wm',required=True);p.add_argument('--diff',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=400);a=p.parse_args();torch.manual_seed(0);np.random.seed(0);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows,qs=load_data(a.dataset);ck=load_checkpoint(a.wm,map_location=dev);mean=np.array(ck['norm']['obs_mean'],'float32');std=np.array(ck['norm']['obs_std'],'float32');cfg=OmegaConf.load('src/blind_diffusion/configs/model/rssm_small.yaml');trace=json.load(open(a.trace));
 # random 50-step windows, normalized against frozen checkpoint stats
 valid=[(o[s:s+50],ac[s:s+50]) for o,ac in rows for s in range(len(ac)-49)];held=valid[-min(128,len(valid)):];train=valid[:-len(held)]
 dck=load_checkpoint(a.diff,map_location=dev);dc=dck['config']['diffusion'];den=UNet1D(7,dc['horizon'],230,base_ch=dc['base_ch']).to(dev);diff=GaussianDiffusion(den,dc['timesteps'],dc['schedule'],prediction_type=dc['prediction_type']).to(dev);diff.load_state_dict(dck['model']);diff.eval()
 results={}
 for name in ['continuation','mean_norm','obs_noise','action_noise']:
  torch.manual_seed(0);m=WorldModel(19,7,cfg).to(dev);m.load_state_dict(ck['model']);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
  for _ in range(a.steps):
   ix=np.random.randint(len(train),size=32);o=torch.tensor(np.stack([train[i][0] for i in ix]),device=dev);ac=torch.tensor(np.stack([train[i][1] for i in ix]),device=dev);o=(o-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);inp=o+torch.randn_like(o)*.05 if name=='obs_noise' else o;pa=previous_actions(ac);pa=pa+torch.randn_like(pa)*.05 if name=='action_noise' else pa;emb=m.encoder(inp);post=m.rssm.observe(emb,pa);feat=torch.cat([post['h'],post['z']],-1);loss=F.mse_loss(m.obs_head(feat)[:,6:],o[:,6:])+kl_normal(post['post_mean'],post['post_std'],post['prior_mean'],post['prior_std']).sum(-1)[:,6:].clamp_min(1).mean();
   if name=='mean_norm':loss=loss+.01*F.relu(post['post_mean'].norm(dim=-1)-8).square().mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),100);opt.step()
  m.eval();
  with torch.no_grad():
   ho=torch.tensor(np.stack([x[0] for x in held]),device=dev);ha=torch.tensor(np.stack([x[1] for x in held]),device=dev);hn=(ho-torch.tensor(mean,device=dev))/torch.tensor(std,device=dev);emb=m.encoder(hn);post=m.rssm.observe(emb,previous_actions(ha));op=m.obs_head(torch.cat([post['h'],post['z']],-1));rec=float(F.mse_loss(op[:,6:],hn[:,6:]));b=compute_pre_action_beliefs(m,hn,ha);n=min(64,len(held));t=torch.arange(n,device=dev)%43;target=torch.stack([ha[i,t[i]:t[i]+8] for i in range(n)]).transpose(1,2);cond=torch.stack([b[i,t[i]] for i in range(n)]);torch.manual_seed(123);pred=diff.sample(target.shape,cond,sampler='ddim',inference_steps=10).transpose(1,2);met=metric_bundle(pred.cpu().numpy(),target.transpose(1,2).cpu().numpy());peak=replay_peak(m,trace,qs,mean,std,dev)
  results[name]={'replay_posterior_mean_peaks':peak,'teacher_forced_reconstruction_mse':rec,'frozen_policy_ddim10_mse':met['mse'],'frozen_policy_mean_correlation':met['mean_correlation'],'action_absmax':met['action_absmax']}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps({'steps':a.steps,'results':results},indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
