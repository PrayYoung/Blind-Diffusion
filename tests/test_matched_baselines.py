import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import h5py
from omegaconf import OmegaConf
import torch

from blind_diffusion.diffusion.policy import build_action_policy
from blind_diffusion.eval.matched_baselines import MatchedAgent, digest_json, evaluate_episode, file_digest
from blind_diffusion.train.train_world_model import WorldModel


def checkpoints(root):
    model_cfg={"encoder":{"hidden_dim":8,"layers":2},
        "rssm":{"deter_dim":7,"stoch_dim":3,"hidden_dim":11,"min_std":.1,"max_std":1.},
        "heads":{"reward_hidden":8,"done_hidden":8,"obs_hidden":8}}
    wm=WorldModel(4,2,OmegaConf.create(model_cfg))
    split={"train":["demo_0"],"val":["demo_1"]}
    torch.save({"model":wm.state_dict(),"config":{"model":model_cfg,"task":{"obs_keys":["state"]}},
        "norm":{"obs_mean":np.zeros(4),"obs_std":np.ones(4),"act_mean":None,"act_std":None},
        "split":split},root/"wm.pt")
    config={"policy_type":"flow_matching","diffusion":{"horizon":8,"base_ch":8,"architecture":"skip_v2"}}
    policy=build_action_policy(config,2,10)
    torch.save({"model":policy.state_dict(),"config":config,"split":split},root/"policy.pt")


class MatchedBaselineTests(unittest.TestCase):
    def test_live_snapshot_merge_preserves_results_and_rejects_changed_records(self):
        import runpy
        import sys
        from unittest.mock import patch
        scripts=Path(__file__).resolve().parents[1]/"scripts"
        with patch.object(sys,"path",[str(scripts),*sys.path]):
            snapshot=runpy.run_path(str(scripts/"run_matched_campaign.py"))["save_evaluation_snapshot"]
            merge=runpy.run_path(str(scripts/"watch_runpod_snapshots.py"))["merge_snapshots"]
        protocol={"id":"recovery_test","methods":{"lift":["cfm4"]},"episodes":{"lift":2},
                  "blackout_durations":[0,20],"max_steps":{"lift":7},"paired_tests":[]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"remote"
            records=root/"episodes/lift/cfm4/d0"
            records.mkdir(parents=True)
            (root/"protocol.json").write_text(json.dumps(protocol))
            row={"task":"lift","method":"cfm4","duration":0,"seed":0,"success":False,
                "protocol_sha256":digest_json(protocol),"dataset_sha256":"data",
                "provenance":{"world_model_sha256":"wm","policy_sha256":"policy","evaluation_engine_sha256":"engine"},
                "initial_observation":[0.],"initial_simulator_state_sha256":"same"}
            record=records/"seed0.json"
            record.write_text(json.dumps(row))
            snapshot(root,"lift","cfm4",0)
            local=Path(directory)/"local"
            local.mkdir()
            (local/"protocol.json").write_text(json.dumps(protocol))
            merge(root/"snapshots",local,protocol)
            self.assertEqual(json.loads((local/"episodes/lift/cfm4/d0/seed0.json").read_text()),row)
            self.assertEqual(json.loads((local/"results.json").read_text())["results"]["lift"]["cfm4"]["0"]["completed"],1)
            record.write_text(json.dumps({**row,"success":True}))
            snapshot(root,"lift","cfm4",0)
            with self.assertRaisesRegex(ValueError,"completed record"):
                merge(root/"snapshots",local,protocol)

    def test_reset_roundoff_is_paired_exactly_and_real_mismatch_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            data=root/"dataset"
            data.write_text("same dataset")
            counter={"state":1.,"mismatch":False}
            def reset():
                counter["state"]=1.
                return {"state":np.zeros(4,dtype=np.float32)}
            def restore(value):
                counter["state"]=float(np.nextafter(value["states"][0],np.inf))
                if counter["mismatch"]:
                    counter["state"]=float(np.nextafter(counter["state"],np.inf))
                return {"state":np.zeros(4,dtype=np.float32)}
            env=SimpleNamespace(reset=reset,reset_to=restore,
                get_state=lambda:{"states":np.array([counter["state"]])},
                step=lambda action:({"state":np.zeros(4,dtype=np.float32)},0.,True,{}))
            agent=SimpleNamespace(keys=["state"],reset=lambda:None,
                provenance={"world_model_sha256":file_digest(str(root/"wm.pt")),
                            "policy_sha256":file_digest(str(root/"policy.pt"))},
                action=lambda *args:torch.zeros(1,2))
            job={"task":"lift","method":"cfm4","seed":0,"wm":str(root/"wm.pt"),
                "policy":str(root/"policy.pt"),"dataset":str(data),"device":"cpu",
                "initial_states_dir":str(root/"initial_states"),
                "protocol":{"visible_warmup":5,"blackout_period":20,"max_steps":{"lift":1}}}
            with patch("blind_diffusion.eval.matched_baselines.worker_assets",return_value=(env,agent)):
                left=evaluate_episode({**job,"duration":0,"output":str(root/"left.json")})
                right=evaluate_episode({**job,"duration":20,"output":str(root/"right.json")})
                self.assertEqual(left["initial_simulator_state_sha256"],right["initial_simulator_state_sha256"])
                template=json.loads((root/"initial_states/lift/seed0.json").read_text())
                self.assertNotEqual(template["states"],template["restored_states"])
                counter["mismatch"]=True
                with self.assertRaisesRegex(ValueError,"different canonical starting state"):
                    evaluate_episode({**job,"duration":5,"output":str(root/"bad.json")})

    def test_partial_cell_snapshot_recovers_only_completed_records(self):
        import runpy
        import sys
        import tarfile
        from unittest.mock import patch
        script=Path(__file__).resolve().parents[1]/"scripts/run_matched_campaign.py"
        with patch.object(sys,"path",[str(script.parent),*sys.path]):
            snapshot=runpy.run_path(str(script))["save_evaluation_snapshot"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"outputs/matched_baselines"
            records=root/"episodes/lift/cfm4/d0"
            states=root/"initial_states/lift"
            records.mkdir(parents=True)
            states.mkdir(parents=True)
            (root/"protocol.json").write_text('{"id":"frozen"}')
            (records/"seed0.json").write_text('{"seed":0,"success":false}')
            (records/"seed1.tmp").write_text('{"unfinished":')
            (states/"seed0.json").write_text('{"states":[1,2,3]}')
            first=snapshot(root,"lift","cfm4",0)
            # Simulate an additional episode finishing before the next backup.
            (records/"seed1.json").write_text('{"seed":1,"success":true}')
            final=snapshot(root,"lift","cfm4",0)
            self.assertEqual(first,final)
            recovered=Path(directory)/"recovered"
            with tarfile.open(final,"r:gz") as archive:
                archive.extractall(recovered,filter="data")
            restored=recovered/"outputs/matched_baselines"
            self.assertEqual(json.loads((restored/"episodes/lift/cfm4/d0/seed0.json").read_text()),
                             {"seed":0,"success":False})
            self.assertTrue((restored/"episodes/lift/cfm4/d0/seed1.json").exists())
            self.assertFalse((restored/"episodes/lift/cfm4/d0/seed1.tmp").exists())
            self.assertEqual(json.loads((restored/"initial_states/lift/seed0.json").read_text()),
                             {"states":[1,2,3]})

    def test_canonical_state_is_restored_for_every_condition(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            data=root/"dataset"
            data.write_text("mock dataset identity")
            counter={"reset":0,"state":0.,"restored":[]}
            def reset():
                counter["reset"]+=1
                counter["state"]=float(counter["reset"])
                return {"state":np.zeros(4,dtype=np.float32)}
            def restore(value):
                counter["state"]=float(value["states"][0])
                counter["restored"].append(counter["state"])
                return {"state":np.zeros(4,dtype=np.float32)}
            env=SimpleNamespace(reset=reset,reset_to=restore,
                get_state=lambda:{"states":np.array([counter["state"],0.,0.])},
                step=lambda action:({"state":np.zeros(4,dtype=np.float32)},0.,True,{}))
            agent=SimpleNamespace(keys=["state"],reset=lambda:None,
                provenance={"world_model_sha256":file_digest(str(root/"wm.pt")),
                            "policy_sha256":file_digest(str(root/"policy.pt"))},action=lambda *args:torch.zeros(1,2))
            base={"task":"lift","method":"cfm4","seed":0,"wm":str(root/"wm.pt"),
                "policy":str(root/"policy.pt"),"dataset":str(data),"device":"cpu",
                "initial_states_dir":str(root/"initial_states"),
                "protocol":{"visible_warmup":5,"blackout_period":20,"max_steps":{"lift":1}}}
            with patch("blind_diffusion.eval.matched_baselines.worker_assets",return_value=(env,agent)):
                left=evaluate_episode({**base,"duration":0,"output":str(root/"left.json")})
                right=evaluate_episode({**base,"duration":20,"output":str(root/"right.json")})
            self.assertEqual(counter["restored"],[1.,1.])
            self.assertEqual(left["initial_simulator_state_sha256"],right["initial_simulator_state_sha256"])

    def test_nonfinite_policy_is_recorded_and_not_retried(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            data=root/"dataset"
            data.write_text("mock simulator dataset identity")
            agent=SimpleNamespace(keys=["state"],reset=lambda:None,
                provenance={"world_model_sha256":file_digest(str(root/"wm.pt")),
                            "policy_sha256":file_digest(str(root/"policy.pt"))},
                action=lambda *args:torch.full((1,2),float("nan")))
            env=SimpleNamespace(reset=lambda:{"state":np.zeros(4,dtype=np.float32)},
                                get_state=lambda:{"states":np.zeros(3)})
            job={"task":"lift","method":"cfm4","duration":20,"seed":0,
                "wm":str(root/"wm.pt"),"policy":str(root/"policy.pt"),"dataset":str(data),"device":"cpu",
                "protocol":{"visible_warmup":5,"blackout_period":20,"max_steps":{"lift":7}},
                "output":str(root/"episode.json")}
            with patch("blind_diffusion.eval.matched_baselines.worker_assets",return_value=(env,agent)):
                result=evaluate_episode(job)
            self.assertFalse(result["success"])
            self.assertEqual(result["termination_reason"],"nonfinite_action")
            self.assertEqual(result["length"],0)
            with patch("blind_diffusion.eval.matched_baselines.worker_assets",side_effect=AssertionError("Unexpected retry")):
                self.assertEqual(evaluate_episode(job),result)

    def test_bc_training_saves_heldout_ema_with_shared_split(self):
        from unittest.mock import patch
        from blind_diffusion.train.train_bc_matched import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            dataset=root/"data.hdf5"
            with h5py.File(dataset,"w") as handle:
                for index in range(2):
                    group=handle.create_group(f"data/demo_{index}")
                    group.create_dataset("obs/state",data=np.full((40,4),index*.1,dtype=np.float32))
                    group.create_dataset("actions",data=np.full((40,2),.2,dtype=np.float32))
                    for key in ("rewards","dones"):
                        group.create_dataset(key,data=np.zeros(40,dtype=np.float32))
            with patch("sys.argv",["train_bc_matched","--dataset",str(dataset),"--world-model",str(root/"wm.pt"),
                                    "--output",str(root/"bc"),"--epochs","1","--batch-size","16"]):
                main()
            result=torch.load(root/"bc/best.pt",weights_only=False)
            self.assertEqual(result["split"],{"train":["demo_0"],"val":["demo_1"]})
            self.assertIn("ema_model",result)
            self.assertEqual(result["selection"]["steps"],2)
            self.assertTrue(np.isfinite(result["selection"]["val_mse"]))

    def test_blind_flow_does_not_encode_current_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            agent=MatchedAgent("cfm4",str(root/"wm.pt"),str(root/"policy.pt"))
            def reject(*args):
                raise AssertionError("Blind policy encoded a sensor observation")
            hook=agent.wm.encoder.register_forward_pre_hook(reject)
            first=agent.action(np.zeros(4,dtype=np.float32),True,3,5)
            agent.reset()
            second=agent.action(np.full(4,100,dtype=np.float32),True,3,5)
            hook.remove()
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(torch.isfinite(first).all())

    def test_open_loop_zero_fills_and_executes_the_cached_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoints(root)
            payload=torch.load(root/"policy.pt",weights_only=False)
            config={"policy_type":"diffusion","diffusion":{"horizon":8,"base_ch":8,
                "architecture":"skip_v2","timesteps":100,"schedule":"cosine","prediction_type":"x0"}}
            policy=build_action_policy(config,2,10)
            payload.update(config=config,model=policy.state_dict())
            torch.save(payload,root/"policy.pt")
            agent=MatchedAgent("open_loop",str(root/"wm.pt"),str(root/"policy.pt"))
            inputs=[]
            hook=agent.wm.encoder.register_forward_pre_hook(lambda module,args:inputs.append(args[0].clone()))
            agent.action(np.full(4,10,dtype=np.float32),True,0,5)
            expected=agent.plan[:,:,1].clone()
            actual=agent.action(np.full(4,-10,dtype=np.float32),True,0,6)
            hook.remove()
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(all(torch.count_nonzero(x)==0 for x in inputs))

    def test_collector_rejects_unpaired_states_and_marks_missing_cells(self):
        script=Path(__file__).resolve().parents[1]/"scripts/collect_matched_results.py"
        spec=importlib.util.spec_from_file_location("collect_matched_results",script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        protocol={"id":"test","methods":{"lift":["cfm4","bc"]},"episodes":{"lift":2},
                  "blackout_durations":[0,20],"max_steps":{"lift":7},"paired_tests":[]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=module.collect(root,protocol)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["results"]["lift"]["bc"]["20"]["rate"])
            for method,state in [("cfm4","same"),("bc","different")]:
                path=root/"episodes/lift"/method/"d0/seed0.json"
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps({"task":"lift","method":method,"duration":0,"seed":0,
                    "protocol_sha256":digest_json(protocol),"initial_observation":[0.],
                    "initial_simulator_state_sha256":state,"dataset_sha256":"dataset",
                    "provenance":{"world_model_sha256":"wm"},"success":False}))
            with self.assertRaisesRegex(ValueError,"Pairing mismatch"):
                module.collect(root,protocol)


if __name__=="__main__":
    unittest.main()
