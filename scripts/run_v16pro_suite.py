"""Plan 13-cue v16pro experiments; --run trains then exports all evaluations.

Seven training jobs per seed: detector10, two parents100, three adaptations30,
and clean recognizers30. No CNN recovery or single-scale ablation is scheduled.
Without --run the command is read-only. Repeating the same run resumes jobs.
"""
import bootstrap  # noqa: F401
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

from common import load_config
from models.frozen_base import file_sha256
from paths import PROJECT_ROOT
from scripts.job_runner import run_job
from scripts.run_v15pro_suite import (payload_fingerprint, job_fingerprint,
                                     verified_complete, write_json_atomic)

STAGES = ('detector', 'parent', 'parent_no_recurrence', 'main', 'no_value',
          'no_recurrence', 'recognizer')
RECOVERY = ('main', 'no_value', 'no_recurrence')


def source_fingerprint():
    files = [PROJECT_ROOT / 'common.py', PROJECT_ROOT / 'paths.py']
    for directory in ('models', 'data', 'scripts', 'tools'):
        files.extend((PROJECT_ROOT / directory).glob('*.py'))
    return payload_fingerprint({str(p.relative_to(PROJECT_ROOT).as_posix()):
                                file_sha256(p) for p in sorted(files)})


def build_plan(base, root, seeds=(1234, 2345, 3456), device=None,
               detector_epochs=10, parent_epochs=100, epochs=30,
               recognizer_epochs=30, data_root=None, fsdd_root=None):
    root = Path(root).resolve()
    configs, jobs = {}, []
    for seed in seeds:
        for stage in STAGES:
            cfg = deepcopy(base)
            cfg.update(seed=seed, version='v16pro', paper_template=False)
            cfg['device'] = device or cfg.get('device', 'cuda')
            cfg['train']['require_cuda'] = cfg['device'].startswith('cuda')
            if data_root:
                cfg['data']['root'] = str(Path(data_root).resolve())
            if fsdd_root:
                cfg['audio']['fsdd_root'] = str(Path(fsdd_root).resolve())
            directory = root / f'seed_{seed}' / stage
            detector = root / f'seed_{seed}' / 'detector'
            cfg['audio']['norm_stats_path'] = str(root / 'train_audio_norm.pt')
            cfg['noise_detector'].update(checkpoint=str(detector / 'best.pt'),
                last_checkpoint=str(detector / 'last.pt'), pretrain_epochs=detector_epochs)
            parent = stage.startswith('parent')
            cfg['v16'].update(no_value=stage == 'no_value',
                local_scales=['64'] if parent else ['16', '32', '64'])
            cfg['ablation']['use_recurrent'] = 'no_recurrence' not in stage
            budget = (detector_epochs if stage == 'detector' else
                      parent_epochs if parent else
                      recognizer_epochs if stage == 'recognizer' else epochs)
            cfg['train'].update(stage=stage, epochs=budget,
                lr=0.001 if stage == 'recognizer' else 0.0002 if parent else 0.0001,
                ckpt_path=str(directory / 'last.pt'), best_ckpt_path=str(directory / 'best.pt'),
                eval_ckpt_path=str(directory / 'best.pt'), start_epoch=0,
                output_version=f'v16pro_{seed}_{stage}', init_ckpt_path='',
                init_required=False, init_load_optimizer=False, init_strict=True,
                trainable_prefixes=[], parent_sha256='')
            dependencies = []
            if parent or stage in RECOVERY:
                dependencies.append(str(detector / 'best.pt'))
            if stage in RECOVERY:
                parent_name = 'parent_no_recurrence' if stage == 'no_recurrence' else 'parent'
                parent_path = root / f'seed_{seed}' / parent_name / 'best.pt'
                cfg['train'].update(init_ckpt_path=str(parent_path), init_required=True,
                    trainable_prefixes=['image_decoder.', 'image_refiner.', 'audio_decoder.',
                                        'audio_local_cue_multiscale.'])
                dependencies.append(str(parent_path))
            config_path = str(directory / 'config.yaml')
            configs[config_path] = cfg
            script = ('scripts/train_detector_v16pro.py' if stage == 'detector' else
                      'scripts/paper_baseline.py' if stage == 'recognizer' else
                      'scripts/train_v16pro.py')
            command = [script, '--config', config_path]
            if stage == 'recognizer':
                command += ['--kind', 'recognizer']
            jobs.append(dict(id=f'{seed}_{stage}', kind=stage, epochs=budget, command=command,
                outputs=[str(directory / 'best.pt'), str(directory / 'last.pt')],
                last=str(directory / 'last.pt'), dependencies=dependencies))
    return configs, jobs


def execute_plan(root, configs, jobs):
    root = Path(root).resolve()
    description = dict(version='v16pro', source_sha256=source_fingerprint(),
                       configs=configs, jobs=jobs)
    fingerprint = payload_fingerprint(description)
    record = dict(fingerprint=fingerprint, plan=description)
    manifest = root / 'plan.json'
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf-8')) != record:
            raise RuntimeError('Plan/code changed; use a new output directory')
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError('Nonempty directory has no plan.json; use a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest, record)
    for name, cfg in configs.items():
        path = Path(name)
        if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != cfg:
            raise RuntimeError(f'Generated configuration was edited: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    for job in jobs:
        marker = root / f"{job['id']}.done.json"
        task_hash = job_fingerprint(job, fingerprint)
        if verified_complete(marker, job, task_hash):
            print(f"[v16pro] verified complete: {job['id']}", flush=True)
            continue
        command = list(job['command'])
        if Path(job['last']).is_file():
            command += ['--resume']
        run_job(command, root / 'logs' / f"{job['id']}.log")
        for name in job['outputs']:
            if not Path(name).is_file():
                raise RuntimeError(f'Training did not produce {name}')
        write_json_atomic(marker, dict(fingerprint=task_hash,
            outputs={name: file_sha256(name) for name in job['outputs']}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v16pro.yaml')
    parser.add_argument('--output', default='outputs/v16pro')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 2345, 3456])
    parser.add_argument('--device', default=None)
    parser.add_argument('--data-root')
    parser.add_argument('--fsdd-root')
    parser.add_argument('--detector-epochs', type=int, default=10)
    parser.add_argument('--parent-epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--recognizer-epochs', type=int, default=30)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-mask-diagnostics', action='store_true',
                        help='Skip main-only oracle/all inference diagnostics')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**31 for s in args.seeds):
        parser.error('Seeds must be unique integers in [0, 2**31)')
    if min(args.detector_epochs, args.parent_epochs, args.epochs, args.recognizer_epochs) < 1:
        parser.error('Epoch budgets must be positive')
    if args.examples < 0:
        parser.error('--examples must be non-negative')
    if args.device and not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device):
        parser.error('--device must be cpu, cuda, or cuda:N')
    root = Path(args.output).resolve()
    base = load_config(args.config)
    if not base.get('v16', {}).get('enabled'):
        parser.error('v16.enabled=true required')
    if not base.get('data', {}).get('paper_split', {}).get('enabled'):
        parser.error('Independent train/validation/test splits required')
    configs, jobs = build_plan(base, root, args.seeds, args.device,
        args.detector_epochs, args.parent_epochs, args.epochs, args.recognizer_epochs,
        args.data_root, args.fsdd_root)
    export = ['tools/export_v16pro.py', '--root', str(root), '--seeds',
              *map(str, args.seeds), '--examples', str(args.examples)]
    if args.device:
        export += ['--device', args.device]
    exports = [export]
    if not args.skip_mask_diagnostics:
        exports += [export + ['--models', 'main', '--skip-scans', '--mask-source', source]
                    for source in ('oracle', 'all')]
    print(f'v16pro: {len(jobs)} training jobs; {sum(j["epochs"] for j in jobs)} job-epochs; '
          '13 cues, main / no_value / no_recurrence; no CNN recovery baseline.')
    if not args.run:
        for job in jobs:
            print(job['id'], subprocess.list2cmdline([sys.executable, '-u', *job['command']]))
        if not args.skip_eval:
            for command in exports:
                print('evaluation', subprocess.list2cmdline([sys.executable, '-u', *command]))
        print('Dry run only: no files written. Add --run to train and then evaluate.')
        return
    execute_plan(root, configs, jobs)
    if not args.skip_eval:
        # Export verifies every child artifact itself, including on repeated runs.
        for index, command in enumerate(exports):
            name = ('export', 'export_oracle', 'export_all')[index]
            run_job(command, root / 'logs' / (name + '.log'))
    print('[v16pro] requested stages complete.', flush=True)


if __name__ == '__main__':
    main()
