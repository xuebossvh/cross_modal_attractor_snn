"""Publish a read-only snapshot; the shared workspace plan remains authoritative."""
from pathlib import Path
import argparse
import hashlib
import json

def export(plan_path, destination):
    source=Path(plan_path);raw=source.read_bytes();plan=json.loads(raw)
    snapshot={'snapshot_only':True,'authoritative_workspace_plan':'04_实验计划/v15_v16实验改进计划.json',
              'source_sha256':hashlib.sha256(raw).hexdigest(),'source_updated':plan['updated'],
              'note':'Generated release snapshot. Do not edit task status here. Local evidence paths refer to the original workspace; weights and experiment outputs are not included in Git.',
              'plan':plan}
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',default='docs/experiment_plan_snapshot.json')
    a=p.parse_args();print(export(a.plan,a.output))
