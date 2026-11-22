import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from hydra import compose, initialize_config_dir
import torch

from blind_diffusion.utils.checkpoint import load_checkpoint, save_checkpoint


class TrainingResumeTests(unittest.TestCase):
    def fixture(self,root,length=10):
        path=root/"dataset.hdf5"
        with h5py.File(path,"w") as handle:
            for i in range(4):
                demo=handle.create_group(f"data/demo_{i}")
                demo.create_dataset("obs/state",data=np.full((length,4),i*.1,dtype=np.float32))
                demo.create_dataset("actions",data=np.full((length,2),.2,dtype=np.float32))
                for key in ("rewards","dones"):
                    demo.create_dataset(key,data=np.zeros(length,dtype=np.float32))
        config_dir=str(Path(__file__).resolve().parents[1]/"src/blind_diffusion/configs")
        common=[f"task.hdf5_name={path}","task.obs_keys=[state]","seq_len=8","pad_end=2","burn_in=1",
                "batch_size=4","num_workers=0",f"run_dir={root}","model.encoder.hidden_dim=8",
                "model.rssm.deter_dim=7","model.rssm.stoch_dim=3","model.rssm.hidden_dim=11",
                "model.heads.reward_hidden=8","model.heads.done_hidden=8","model.heads.obs_hidden=8"]
        with initialize_config_dir(version_base=None,config_dir=config_dir):
            wm=compose(config_name="train_world_model_recovery",overrides=common+[
                "exp_name=wm","max_steps=10","val_every=2","imag_horizon=3"])
            flow=compose(config_name="train_flow_matching_recovery",overrides=common+[
                "exp_name=flow",f"wm_checkpoint={root}/wm/checkpoints/best.pt","epochs=3",
                "diffusion.base_ch=8","diffusion.horizon=4","val_mc_samples=1"])
        return path,wm,flow

    def compare(self,left,right,keys):
        a,b=load_checkpoint(str(left),"cpu"),load_checkpoint(str(right),"cpu")
        for key in keys:
            for name,value in a[key].items():
                self.assertTrue(torch.equal(value,b[key][name]),f"{key}.{name} changed after recovery")

    def interrupt(self,path,payload):
        save_checkpoint(path,payload)
        if Path(path).name=="last.pt":
            raise RuntimeError("simulated runtime expiry")

    def test_world_model_resume_preserves_updates_and_rng(self):
        from blind_diffusion.train import train_world_model as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,cfg,_=self.fixture(root)
            with patch.object(module,"get_device",return_value=torch.device("cpu")):
                module.main.__wrapped__(cfg)
                partial=copy.deepcopy(cfg)
                partial.exp_name="wm_resumed"
                with patch.object(module,"save_checkpoint",side_effect=self.interrupt):
                    with self.assertRaisesRegex(RuntimeError,"simulated runtime expiry"):
                        module.main.__wrapped__(partial)
                partial.resume_checkpoint=str(root/"wm_resumed/checkpoints/last.pt")
                module.main.__wrapped__(partial)
            self.compare(root/"wm/checkpoints/last.pt",root/"wm_resumed/checkpoints/last.pt",["model"])

    def test_flow_resume_preserves_ema_optimizer_and_rng(self):
        from blind_diffusion.train import train_world_model as wm_module, train_diffusion as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,wm,cfg=self.fixture(root)
            wm.max_steps=2
            with patch.object(wm_module,"get_device",return_value=torch.device("cpu")):
                wm_module.main.__wrapped__(wm)
            with patch.object(module,"get_device",return_value=torch.device("cpu")):
                module.main.__wrapped__(cfg)
                partial=copy.deepcopy(cfg)
                partial.exp_name="flow_resumed"
                with patch.object(module,"save_checkpoint",side_effect=self.interrupt):
                    with self.assertRaisesRegex(RuntimeError,"simulated runtime expiry"):
                        module.main.__wrapped__(partial)
                partial.resume_checkpoint=str(root/"flow_resumed/checkpoints/last.pt")
                module.main.__wrapped__(partial)
            self.compare(root/"flow/checkpoints/last.pt",root/"flow_resumed/checkpoints/last.pt",["model","ema_model"])

    def test_bc_resume_preserves_ema_optimizer_and_rng(self):
        from blind_diffusion.train import train_world_model as wm_module, train_bc_matched as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset,wm,_=self.fixture(root,length=40)
            wm.max_steps=2
            with patch.object(wm_module,"get_device",return_value=torch.device("cpu")):
                wm_module.main.__wrapped__(wm)
            def run(output,resume=None):
                argv=["bc","--dataset",str(dataset),"--world-model",str(root/"wm/checkpoints/best.pt"),
                      "--output",str(output),"--epochs","3","--batch-size","16"]
                if resume:argv += ["--resume-checkpoint",str(resume)]
                with patch("sys.argv",argv):module.main()
            run(root/"bc")
            with patch.object(module,"save_checkpoint",side_effect=self.interrupt):
                with self.assertRaisesRegex(RuntimeError,"simulated runtime expiry"):
                    run(root/"bc_resumed")
            run(root/"bc_resumed",root/"bc_resumed/last.pt")
            self.compare(root/"bc/last.pt",root/"bc_resumed/last.pt",["model","ema_model"])


if __name__=="__main__":
    unittest.main()
