"""Preserve scientific snapshots via read-only managed sync; never own a Pod."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

from filelock import FileLock

from collect_matched_results import collect


def merge_snapshots(directory,output,protocol):
    """Merge immutable completed records after validating each atomic archive."""
    count=0
    for archive_path in sorted(directory.glob("*.tar.gz")):
        with tempfile.TemporaryDirectory() as temporary:
            staging=Path(temporary)
            with tarfile.open(archive_path,"r:gz") as archive:
                archive.extractall(staging,filter="data")
            saved=staging/"outputs/matched_baselines"
            if json.loads((saved/"protocol.json").read_text())!=protocol:
                raise ValueError("Live snapshot belongs to a different frozen protocol")
            collect(saved,protocol)
            for group in ("episodes","initial_states"):
                for source in (saved/group).rglob("*.json"):
                    if group=="initial_states" and "restored_states" not in json.loads(source.read_text()):
                        continue
                    destination=output/source.relative_to(saved)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists() and destination.read_bytes()!=source.read_bytes():
                        raise ValueError("A completed record or paired starting state changed")
                    incoming=destination.with_suffix(".live-incoming")
                    shutil.copy2(source,incoming)
                    incoming.replace(destination)
                    count+=group=="episodes"
    collect(output,protocol)
    if protocol.get("report_horizons"):
        from collect_can_horizons import collect_horizons
        collect_horizons(output,protocol)
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--protocol",type=Path)
    args=parser.parse_args()
    args.artifacts=args.artifacts.resolve()
    repo=Path(__file__).resolve().parents[1]
    protocol=json.loads((args.protocol or repo/"scripts/matched_protocol.json").read_text())
    output=args.artifacts/"outputs/matched_baselines"
    state_path=args.artifacts/"live_backup_state.json"
    lock=FileLock(str(args.artifacts/"live_snapshot_watch.lock"))
    last_sync=0.
    seen={}
    with lock:
        while True:
            campaign=json.loads((args.artifacts/"runpod_campaign.json").read_text())
            if campaign["status"] in ("complete","needs_reconciliation"):
                return
            run_id=campaign.get("run_id")
            if run_id and Path(run_id).name==run_id and time.monotonic()-last_sync>=120:
                run_dir=repo/".runpod/runs"/run_id
                metadata=json.loads((run_dir/"metadata.json").read_text())
                if metadata.get("status")=="ACTIVE" and metadata.get("phase")=="RUN":
                    with (args.artifacts/"live_snapshot_sync.log").open("a") as log:
                        code=subprocess.run(["runpod-run","sync","--run-dir",str(run_dir),
                            "--path","outputs/matched_baselines/snapshots"],stdout=log,stderr=subprocess.STDOUT).returncode
                    if code==0:
                        directory=run_dir/"live_artifacts/outputs/matched_baselines/snapshots"
                        current={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.glob("*.tar.gz")}
                        if current and current!=seen:
                            merge_snapshots(directory,output,protocol)
                            count=len(list((output/"episodes").rglob("seed*.json")))
                            state={"status":"verified_live_backup","run_id":run_id,
                                "locally_preserved_primary_episodes":count,"checked_at_unix":time.time(),
                                "lifecycle_unchanged":True}
                            incoming=state_path.with_suffix(".incoming.json")
                            incoming.write_text(json.dumps(state,indent=2)+"\n")
                            incoming.replace(state_path)
                            print("LIVE_EPISODES_SAVED "+json.dumps(state),flush=True)
                            seen=current
                    last_sync=time.monotonic()
            time.sleep(30)


if __name__=="__main__":
    main()
