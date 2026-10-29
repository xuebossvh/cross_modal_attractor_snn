"""Exercise suite planning/resume/provenance without training or datasets."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CLI modules use `import bootstrap`; support both discovery and direct execution.
for directory in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import load_config  # noqa: E402
from scripts import run_v15pro_suite as suite  # noqa: E402


class V15SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="v15pro-suite-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = load_config(PROJECT_ROOT / "configs" / "v15pro.yaml")
        source = patch.object(suite, "source_fingerprint", return_value="test-source-sha")
        source.start()
        self.addCleanup(source.stop)

    def plan(self, name="run", seeds=(1234,)):
        root = self.directory / name
        configs, jobs = suite.build_plan(
            self.base, root, seeds=seeds, device="cpu", detector_epochs=1, epochs=1)
        return root, configs, jobs

    def fake_runner(self, jobs, calls, interrupt_stage=None):
        def run(command, _logfile):
            matches = [job for job in jobs
                       if command[:len(job["command"])] == job["command"]]
            self.assertEqual(len(matches), 1)
            job = matches[0]
            calls.append((job["id"], list(command)))
            if job["kind"] == interrupt_stage:
                last = Path(job["last"])
                last.parent.mkdir(parents=True, exist_ok=True)
                last.write_bytes(b"partial checkpoint")
                raise RuntimeError("simulated interruption")
            for name in job["outputs"]:
                output = Path(name)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"{job['id']}:{output.name}".encode())
        return run

    def execute(self, root, configs, jobs, **kwargs):
        with redirect_stdout(io.StringIO()):
            suite.execute_plan(root, configs, jobs, **kwargs)

    def test_build_plan_has_no_writes_and_seeds_have_independent_artifacts(self):
        original = deepcopy(self.base)
        root, configs, jobs = self.plan(seeds=(1234, 2345, 3456))
        self.assertFalse(root.exists())
        self.assertEqual(self.base, original)
        self.assertEqual(len(configs), 3)
        self.assertEqual(len(jobs), 15)
        artifacts = [name for job in jobs for name in job["outputs"]]
        self.assertEqual(len(artifacts), len(set(artifacts)))
        for config_path, cfg in configs.items():
            seed_directory = root / f"seed_{cfg['seed']}"
            self.assertEqual(Path(config_path).parent, seed_directory)
            self.assertEqual(Path(cfg["noise_detector"]["checkpoint"]),
                             seed_directory / "detector" / "best.pt")
            self.assertEqual(Path(cfg["train"]["ckpt_path"]),
                             seed_directory / "recovery" / "last.pt")
            self.assertFalse(cfg["train"]["require_cuda"])
        for job in jobs:
            if job["kind"] == "evaluation":
                self.assertEqual({Path(p).suffix for p in job["outputs"]}, {".json", ".csv"})

    def test_cli_defaults_to_dry_run_without_creating_output(self):
        root = self.directory / "dry-run"
        argv = ["run_v15pro_suite.py", "--config",
                str(PROJECT_ROOT / "configs" / "v15pro.yaml"),
                "--output", str(root), "--device", "cpu", "--seeds", "1234"]
        captured = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(suite, "run_job") as run:
            with redirect_stdout(captured):
                suite.main()
        run.assert_not_called()
        self.assertFalse(root.exists())
        self.assertIn("Dry run only", captured.getvalue())
        self.assertIn("evaluation_predicted", captured.getvalue())

    def test_skip_eval_then_complete_and_skip_verified_jobs(self):
        root, configs, jobs = self.plan()
        calls = []
        with patch.object(suite, "run_job", side_effect=self.fake_runner(jobs, calls)):
            self.execute(root, configs, jobs, skip_eval=True)
            self.assertEqual([name for name, _ in calls], [jobs[0]["id"], jobs[1]["id"]])
            self.assertFalse((root / "seed_1234" / "evaluation").exists())
            self.execute(root, configs, jobs)
            self.assertEqual(len(calls), 5)
            self.execute(root, configs, jobs)
            self.assertEqual(len(calls), 5)
        for job in jobs:
            marker = json.loads((root / f"{job['id']}.done.json").read_text(encoding="utf-8"))
            self.assertEqual(set(marker["outputs"]), set(job["outputs"]))
            for artifact in job["outputs"]:
                self.assertEqual(marker["outputs"][artifact], suite.file_sha256(artifact))

    def test_complete_markers_reject_tampering_and_missing_artifacts(self):
        root, configs, jobs = self.plan()
        calls = []
        with patch.object(suite, "run_job", side_effect=self.fake_runner(jobs, calls)):
            self.execute(root, configs, jobs)
            artifacts = [Path(name) for job in jobs for name in job["outputs"]]
            for artifact in artifacts:
                original = artifact.read_bytes()
                for operation in ("tamper", "delete"):
                    with self.subTest(artifact=artifact.name, parent=artifact.parent.name,
                                      operation=operation):
                        try:
                            if operation == "tamper":
                                artifact.write_bytes(b"different contents")
                            else:
                                artifact.unlink()
                            with self.assertRaisesRegex(RuntimeError, "artifact missing or changed"):
                                self.execute(root, configs, jobs)
                            self.assertEqual(len(calls), 5)
                        finally:
                            artifact.write_bytes(original)

    def test_incomplete_detector_and_recovery_resume_from_last_checkpoint(self):
        for stage in ("detector", "recovery"):
            with self.subTest(stage=stage):
                root, configs, jobs = self.plan(name=f"interrupted-{stage}")
                calls = []
                interrupted = self.fake_runner(jobs, calls, interrupt_stage=stage)
                with patch.object(suite, "run_job", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        self.execute(root, configs, jobs, skip_eval=True)
                job = next(job for job in jobs if job["kind"] == stage)
                self.assertTrue(Path(job["last"]).is_file())
                self.assertFalse((root / f"{job['id']}.done.json").exists())
                calls.clear()
                with patch.object(suite, "run_job", side_effect=self.fake_runner(jobs, calls)):
                    self.execute(root, configs, jobs, skip_eval=True)
                resumed = [command for name, command in calls if name == job["id"]]
                self.assertEqual(len(resumed), 1)
                self.assertEqual(resumed[0][-1], "--resume")
                if stage == "recovery":
                    self.assertNotIn(jobs[0]["id"], [name for name, _ in calls])

    def test_changed_plan_cannot_overwrite_existing_run(self):
        root, configs, jobs = self.plan()
        calls = []
        with patch.object(suite, "run_job", side_effect=self.fake_runner(jobs, calls)):
            self.execute(root, configs, jobs, skip_eval=True)
            changed = deepcopy(configs)
            next(iter(changed.values()))["train"]["epochs"] = 2
            with self.assertRaisesRegex(RuntimeError, "Plan/code changed"):
                self.execute(root, changed, jobs)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
