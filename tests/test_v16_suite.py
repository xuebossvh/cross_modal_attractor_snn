"""Budget, lineage, persistence, and automatic-evaluation contracts."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import load_config
from scripts import run_v16pro_suite as suite


class V16SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.base = load_config(ROOT / 'configs/v16pro.yaml')

    def plan(self, seeds=(1234,)):
        return suite.build_plan(self.base, self.root, seeds=seeds, device='cpu')

    def runner(self, jobs, calls, fail=None):
        def run(command, logfile):
            job = next(j for j in jobs if command[:len(j['command'])] == j['command'])
            calls.append((job['kind'], command))
            if fail == job['kind']:
                Path(job['last']).write_bytes(b'interrupted')
                raise RuntimeError('interrupted')
            for filename in job['outputs']:
                Path(filename).write_bytes(f"{job['id']}:{filename}".encode())
        return run

    def test_budget_lineage_and_no_unrequested_models(self):
        original = deepcopy(self.base)
        configs, jobs = self.plan((1234, 2345, 3456))
        self.assertEqual(self.base, original)
        self.assertFalse(self.root.exists())
        self.assertEqual(len(jobs), 21)
        self.assertEqual(sum(j['epochs'] for j in jobs), 990)
        self.assertEqual(set(j['kind'] for j in jobs), set(suite.STAGES))
        for path, cfg in configs.items():
            stage = cfg['train']['stage']
            self.assertFalse(cfg['cross_key_conditioning']['enabled'])
            self.assertFalse(cfg['cross_key_conditioning']['build_modules'])
            self.assertEqual(cfg['v16']['no_value'], stage == 'no_value')
            self.assertEqual(cfg['ablation']['use_recurrent'], 'no_recurrence' not in stage)
            self.assertAlmostEqual(sum(cfg['v16']['cue_probabilities'].values()), 1.)
            if stage in suite.RECOVERY:
                self.assertEqual(cfg['v16']['local_scales'], ['16', '32', '64'])
                expected = 'parent_no_recurrence' if stage == 'no_recurrence' else 'parent'
                self.assertEqual(Path(cfg['train']['init_ckpt_path']).parent.name, expected)
                self.assertEqual(cfg['train']['epochs'], 30)

    def test_resume_and_verify_completed_outputs(self):
        configs, jobs = self.plan()
        calls = []
        with patch.object(suite, 'source_fingerprint', return_value='source'), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch.object(suite, 'run_job', side_effect=self.runner(jobs, calls, 'main')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    suite.execute_plan(self.root, configs, jobs)
            calls.clear()
            with patch.object(suite, 'run_job', side_effect=self.runner(jobs, calls)):
                suite.execute_plan(self.root, configs, jobs)
                self.assertIn('--resume', calls[0][1])
                self.assertEqual(calls[0][0], 'main')
                calls.clear()
                suite.execute_plan(self.root, configs, jobs)
                self.assertEqual(calls, [])
                Path(jobs[-1]['outputs'][0]).write_bytes(b'tampered')
                with self.assertRaisesRegex(RuntimeError, 'artifact missing or changed'):
                    suite.execute_plan(self.root, configs, jobs)

    def test_changed_plan_rejected_and_dry_run_has_no_files(self):
        configs, jobs = self.plan()
        with contextlib.redirect_stdout(io.StringIO()):
            suite.main(['--config', str(ROOT / 'configs/v16pro.yaml'),
                        '--output', str(self.root), '--device', 'cpu'])
        self.assertFalse(self.root.exists())
        with patch.object(suite, 'run_job', side_effect=self.runner(jobs, [])), \
                patch.object(suite, 'source_fingerprint', return_value='source'), \
                contextlib.redirect_stdout(io.StringIO()):
            suite.execute_plan(self.root, configs, jobs)
            changed = deepcopy(configs)
            next(iter(changed.values()))['seed'] = 42
            with self.assertRaisesRegex(RuntimeError, 'Plan/code changed'):
                suite.execute_plan(self.root, changed, jobs)

    def test_run_trains_before_export_and_stops_on_failure(self):
        args = ['--config', str(ROOT / 'configs/v16pro.yaml'), '--output', str(self.root),
                '--device', 'cpu', '--seeds', '1234', '--run']
        sequence = []
        with patch.object(suite, 'execute_plan', side_effect=lambda *a: sequence.append('train')), \
                patch.object(suite, 'run_job', side_effect=lambda command, log: sequence.append(command)), \
                contextlib.redirect_stdout(io.StringIO()):
            suite.main(args)
        self.assertEqual(sequence[0], 'train')
        self.assertEqual(sequence[1][0], 'tools/export_v16pro.py')
        self.assertEqual(len(sequence), 4)
        self.assertEqual(sequence[2][-1], 'oracle')
        self.assertEqual(sequence[3][-1], 'all')
        self.assertIn('--skip-scans', sequence[2])
        with patch.object(suite, 'execute_plan', side_effect=RuntimeError('training failed')), \
                patch.object(suite, 'run_job') as export, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                suite.main(args)
            export.assert_not_called()


if __name__ == '__main__':
    unittest.main()
