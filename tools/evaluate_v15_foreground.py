"""Supplementary v15 evaluation: paired location policies and foreground metrics.

Consumes losslessly packaged original clean inputs and unchanged inference-only
weights. All true regions are confined to corruption generation and evaluation.
No training runtime module or original result is modified.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
import platform
import time
import numpy as np
import torch

from models.network import CrossModalSNN
from scripts.evaluate_v15pro import verify_checkpoint_config
from scripts.noise_metrics import _conditions, detection_rates
from tools.export_v15pro import runtime_source_fingerprint
from tools.v15_foreground_protocol import DEFAULT_OPTIONS, make_cues

REGIONS = ('global','corrupt','clean','foreground','corrupt_foreground',
           'corrupt_background','clean_foreground','clean_background')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def batch_metrics(cue,recovered,target,truth,predicted,modality,options):
    """Vectorized sufficient statistics, equivalent to noise_metrics.sample_metrics."""
    arrays=[np.asarray(x) for x in (cue,recovered,target,truth,predicted)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim not in (3,4) or len(arrays[0])==0:
        raise ValueError('Metric arrays must have matching batched image/audio shapes')
    if any(not np.isfinite(x).all() for x in arrays):raise ValueError('Non-finite metric input')
    cue,recovered,target,truth,predicted=[x.reshape(len(x),-1) for x in arrays]
    if not (np.isin(truth,[0,1]).all() and np.isin(predicted,[0,1]).all()):
        raise ValueError('Masks must be binary')
    truth,predicted=truth.astype(bool),predicted.astype(bool)
    threshold=(np.full((len(target),1),options['img_threshold']) if modality=='img' else
               np.maximum(options['aud_absolute_threshold'],options['aud_relative_threshold']*target.max(1,keepdims=True).astype(np.float64)))
    # Match Torch's scalar comparison: compute the audio threshold in Python
    # precision, then cast it to the target dtype before the strict comparison.
    threshold=threshold.astype(target.dtype)
    fg=target>threshold
    regions={'global':np.ones_like(truth),'corrupt':truth,'clean':~truth,
             'foreground':fg,'corrupt_foreground':truth&fg,'corrupt_background':truth&~fg,
             'clean_foreground':~truth&fg,'clean_background':~truth&~fg}
    errors={'input':(cue.astype(np.float64)-target.astype(np.float64))**2,
            'recovered':(recovered.astype(np.float64)-target.astype(np.float64))**2}
    result={}
    for name,mask in regions.items():
        counts=mask.sum(1);result[name+'_pixels']=counts
        for kind,err in errors.items():
            sse=np.sum(err*mask,axis=1,dtype=np.float64)
            result[f'{name}_{kind}_sse']=sse
            result[f'{name}_{kind}_mse']=[float(x/n) if n else None for x,n in zip(sse,counts)]
    for k,m in {'tp':predicted&truth,'fp':predicted&~truth,'fn':~predicted&truth,'tn':~predicted&~truth}.items():
        result['detector_'+k]=m.sum(1)
    return result


class Accumulator:
    def __init__(self):
        self.n=0;self.correct=0
        self.values={m:{'regions':{r:{'pixels':0,'input_sse':0.,'recovered_sse':0.,'valid_samples':0} for r in REGIONS},
            'detector':{k:0 for k in ('tp','fp','fn','tn')},
            'placement':{'active_samples':0,'foreground_evaluable':0,'constraint_applicable':0,
                         'constraint_met':0,'constraint_not_met':0,'constraint_infeasible':0,'no_foreground':0,'coverage_sum':0.,
                         'coverage_count':0,'zero_overlap_active':0}} for m in ('img','aud')}

    def add(self,metrics,metadata,correct):
        self.n+=len(correct);self.correct+=int(np.sum(correct))
        for mod in ('img','aud'):
            dst=self.values[mod];src=metrics[mod]
            for region in REGIONS:
                d=dst['regions'][region]
                d['pixels']+=int(np.sum(src[region+'_pixels']))
                d['valid_samples']+=int(np.count_nonzero(src[region+'_pixels']))
                for kind in ('input','recovered'):
                    d[kind+'_sse']+=float(np.sum(src[f'{region}_{kind}_sse']))
            for k in ('tp','fp','fn','tn'):dst['detector'][k]+=int(np.sum(src['detector_'+k]))
            p=dst['placement']
            for row in metadata['samples']:
                m=row['modalities'][mod]
                if not m['active']:continue
                p['active_samples']+=1
                p['foreground_evaluable']+=int(m['evaluable'])
                p['constraint_applicable']+=int(m['constraint_applicable'])
                p['constraint_met']+=int(m['constraint_met'] is True)
                p['constraint_not_met']+=int(m['constraint_met'] is False)
                p['constraint_infeasible']+=int(m['status']=='constraint_infeasible')
                p['no_foreground']+=int(m['status']=='no_foreground')
                p['zero_overlap_active']+=int(m['covered_foreground_pixels']==0)
                if m['foreground_coverage'] is not None:
                    p['coverage_sum']+=m['foreground_coverage'];p['coverage_count']+=1

    def finish(self):
        if not self.n:raise ValueError('No observations to summarize')
        result={'n':self.n,'index_accuracy':self.correct/self.n,'modalities':deepcopy(self.values)}
        for mod,d in result['modalities'].items():
            for r in d['regions'].values():
                for kind in ('input','recovered'):
                    r[kind+'_mse']=r[kind+'_sse']/r['pixels'] if r['pixels'] else None
            d['detector'].update(detection_rates(**d['detector']))
            p=d['placement'];p['mean_foreground_coverage']=p['coverage_sum']/p['coverage_count'] if p['coverage_count'] else None
            p['constraint_met_rate']=p['constraint_met']/p['constraint_applicable'] if p['constraint_applicable'] else None
            p['constraint_not_met_rate']=p['constraint_not_met']/p['constraint_applicable'] if p['constraint_applicable'] else None
            p['constraint_infeasible_rate']=p['constraint_infeasible']/p['constraint_applicable'] if p['constraint_applicable'] else None
            p['no_foreground_rate']=p['no_foreground']/p['active_samples'] if p['active_samples'] else None
            p['zero_overlap_rate']=p['zero_overlap_active']/p['active_samples'] if p['active_samples'] else None
        return result


def verified_condition(folder,identity):
    marker=folder/'complete.json'
    if not marker.exists():return None
    rec=json.loads(marker.read_text(encoding='utf-8'))
    if rec['identity']!=identity:raise ValueError(f'Completed evaluation settings differ: {folder}')
    for rel,digest in rec['artifacts'].items():
        if sha(folder/rel)!=digest:raise ValueError(f'Changed result: {folder/rel}')
    return json.loads((folder/'summary.json').read_text(encoding='utf-8'))


def run_seed(args,asset_manifest,seed):
    asset=asset_manifest['seeds'][str(seed)];base=Path(args.assets)
    cp=base/asset['checkpoint'];dataset_info=asset['datasets'][args.split]
    npz=base/dataset_info['file']
    if sha(cp)!=asset['checkpoint_sha256'] or sha(npz)!=dataset_info['file_sha256']:
        raise ValueError('Packaged checkpoint/data hash mismatch')
    state=torch.load(cp,map_location='cpu',weights_only=True)
    if not state.get('asset_packaging',{}).get('inference_only'):
        raise ValueError('Expected documented inference-only packaged weights')
    if state['asset_packaging']['original_checkpoint_sha256']!=asset['original_checkpoint_sha256']:
        raise ValueError('Original checkpoint provenance mismatch')
    cfg=deepcopy(state['cfg']);cfg['device']=args.device
    verify_checkpoint_config(cfg,state['cfg'])
    model=CrossModalSNN(cfg).to(args.device);model.load_state_dict(state['model'],strict=True)
    del state
    model.eval();model.noise_mask_source='predicted'
    with np.load(npz,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    n=len(arrays['labels']);n=min(n,args.max_samples) if args.max_samples else n
    images=torch.from_numpy(arrays['images'][:n]);audio=torch.from_numpy(arrays['unique_audio'])
    ids=arrays['sample_ids'][:n];labels=arrays['labels'][:n]
    options=dict(DEFAULT_OPTIONS)
    severity=float(cfg['noise_evaluation'].get('severity',.25));noise_seed=int(cfg['noise_evaluation']['seed'])
    source_digest=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'tools/v15_foreground_protocol.py').read_bytes()).hexdigest()
    identity_base=dict(schema='v15-foreground-eval.v1',training_seed=seed,original_checkpoint_sha256=asset['original_checkpoint_sha256'],
        packaged_checkpoint_sha256=asset['checkpoint_sha256'],data_sha256=dataset_info['file_sha256'],split=args.split,
        samples=n,total_split_samples=dataset_info['total_split_samples'],partial_evaluation=n<dataset_info['total_split_samples'],
        options=options,area_coverage=severity,evaluation_seed=noise_seed,batch_size=args.batch_size,examples=args.examples,
        device=args.device,torch_version=torch.__version__,source_sha256=source_digest,runtime_sha256=runtime_source_fingerprint())
    results=[]
    for policy in args.policies:
        for family,mode in _conditions(cfg):
            # The unmodified clean condition is shared by both placement protocols.
            if policy=='foreground' and family=='clean':continue
            identity=dict(identity_base,policy=policy,family=family,cue_mode=mode)
            folder=Path(args.output)/f'seed_{seed}'/policy/f'{family}__{mode}'
            old=verified_condition(folder,identity)
            if old is not None:
                print(f'[verified] seed={seed} {policy} {family} {mode}',flush=True);results.append(old);continue
            folder.mkdir(parents=True,exist_ok=True)
            save_json(folder/'in_progress.json',identity)
            accumulator=Accumulator();examples=[];start=time.monotonic()
            csv_path=folder/'per_sample.csv.gz';tmp=folder/'per_sample.csv.gz.tmp'
            with gzip.open(tmp,'wt',encoding='utf-8',newline='') as f,torch.inference_mode():
                writer=None
                for lo in range(0,n,args.batch_size):
                    hi=min(n,lo+args.batch_size)
                    image=images[lo:hi];sound=audio[arrays['audio_indices'][lo:hi]]
                    ci,ca,truth,metadata=make_cues(image,sound,ids[lo:hi].tolist(),mode,family,cfg,severity,noise_seed,policy,options)
                    # Truth and foreground masks NEVER enter predicted-mode forward.
                    out=model(x_img_cue=ci.to(args.device),x_aud_cue=ca.to(args.device),training_mode=False)
                    recovered={'img':out['recovered_img'].sigmoid().cpu().numpy(),'aud':out['recovered_aud'].cpu().numpy()}
                    predicted={m:out[m+'_predicted_mask'].cpu().numpy() for m in ('img','aud')}
                    repair={m:out[m+'_repair_mask'].cpu().numpy() for m in ('img','aud')}
                    cls=out['logits'].argmax(1).cpu().numpy();correct=cls==labels[lo:hi]
                    targets={'img':image.numpy(),'aud':sound.numpy()};cues={'img':ci.numpy(),'aud':ca.numpy()}
                    metrics={m:batch_metrics(cues[m],recovered[m],targets[m],truth[m].numpy(),predicted[m],m,options) for m in ('img','aud')}
                    accumulator.add(metrics,metadata,correct)
                    for j,position in enumerate(range(lo,hi)):
                        row={'sample_index':position,'sample_id':int(ids[position]),'label':int(labels[position]),
                             'index_prediction':int(cls[j]),'correct':int(correct[j]),'training_seed':seed,
                             'policy':policy,'family':family,'cue_mode':mode,'split':args.split}
                        for key in ('image_ids','audio_ids','speakers'):
                            if key in arrays:row[key]=str(arrays[key][position])
                        for mod in ('img','aud'):
                            for key,val in metrics[mod].items():
                                v=val[j];row[mod+'_'+key]=v.item() if hasattr(v,'item') else v
                            for key,v in metadata['samples'][j]['modalities'][mod].items():
                                row[mod+'_placement_'+key]=json.dumps(v) if isinstance(v,list) else v
                        if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                        if position<args.examples:
                            stem=f'sample_{position:05d}';edir=folder/'examples';edir.mkdir(exist_ok=True)
                            package={}
                            for mod in ('img','aud'):
                                for key,value in {'target':targets[mod][j],'cue':cues[mod][j],'true_mask':truth[mod][j].numpy(),
                                    'predicted_mask':predicted[mod][j],'repair_mask':repair[mod][j],'recovered':recovered[mod][j]}.items():
                                    package[mod+'_'+key]=value
                            np.savez_compressed(edir/(stem+'.npz'),**package)
                            md={**row,'severity':severity if family!='clean' else 0.,'noise_std':cfg['corruption'].get('noise_std',.2),
                                'coverage':metadata['samples'][j]['modalities'],'sample_selection':'first N indices; no quality filtering',
                                'original_checkpoint_sha256':asset['original_checkpoint_sha256'],'truth_usage':'generation and metrics only; never model input'}
                            save_json(edir/(stem+'.json'),md)
                            examples.append({'npz':'examples/'+stem+'.npz','metadata':'examples/'+stem+'.json','sample_id':int(ids[position])})
                    if lo==0 or hi==n or (lo//args.batch_size)%25==0:
                        print(f'[eval] seed={seed} {policy} {family} {mode} {hi}/{n} elapsed={time.monotonic()-start:.1f}s',flush=True)
                    del out
            tmp.replace(csv_path)
            summary={**identity,'statistics':accumulator.finish(),'elapsed_seconds':time.monotonic()-start,'examples':examples}
            save_json(folder/'summary.json',summary)
            artifact_paths=[csv_path,folder/'summary.json']+[folder/e[k] for e in examples for k in ('npz','metadata')]
            save_json(folder/'complete.json',{'identity':identity,'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
                'artifacts':{p.relative_to(folder).as_posix():sha(p) for p in artifact_paths}})
            (folder/'in_progress.json').unlink(missing_ok=True)
            results.append(summary)
    save_json(Path(args.output)/f'seed_{seed}'/'summary.json',{'seed':seed,'conditions':results})
    del model
    if args.device=='cuda':torch.cuda.empty_cache()
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1234,2345,3456])
    p.add_argument('--policies',choices=['uniform','foreground'],nargs='+',default=['uniform','foreground'])
    p.add_argument('--split',choices=['test','val'],default='test')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-samples',type=int,default=0);p.add_argument('--examples',type=int,default=8)
    p.add_argument('--cuda-memory-fraction',type=float,default=.20)
    args=p.parse_args(argv)
    if min(args.batch_size,args.threads)<=0 or min(args.max_samples,args.examples)<0: p.error('Invalid count')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.policies))!=len(args.policies):p.error('Duplicate seeds/policies')
    if not 0<args.cuda_memory_fraction<=1:p.error('Invalid CUDA memory fraction')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    manifest=json.loads((Path(args.assets)/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete':raise ValueError('Incomplete asset export')
    if manifest['runtime']['source_sha256']!=runtime_source_fingerprint():
        raise ValueError('Original training/inference runtime sources differ')
    records=[]
    for seed in args.seeds:records.extend(run_seed(args,manifest,seed))
    save_json(Path(args.output)/'summary.json',{'schema':'v15-foreground-suite.v1','status':'complete',
        'seeds':args.seeds,'policies':args.policies,'split':args.split,'conditions':records,
        'condition_count':len(records),'partial_evaluation':any(x['partial_evaluation'] for x in records),
        'notes':'No retraining. Clean condition shared; compare placement policies within the same family/cue mode.'})
    print('[complete] supplementary evaluation',flush=True)

if __name__=='__main__':main()
