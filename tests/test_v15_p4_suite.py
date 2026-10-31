"""Suite routing/provenance/locking tests; subprocess training is deliberately stubbed."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from tools import run_v15_p4_suite as suite


def fake_assets(root):
    folder = root / "assets"
    folder.mkdir()
    paths = {"checkpoint": "seed_1234/last.pt", "config": "seed_1234/resolved_config.yaml",
             "norm": "seed_1234/norm_stats.pt", "train": "datasets/train60.npz",
             "val": "datasets/val60.npz", "best": "seed_1234/best_reference.pt"}
    for role, relative in paths.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(role, encoding="utf-8")
    hashes = {relative: suite.file_sha256(folder / relative) for relative in paths.values()}
    source_sha = hashes[paths["checkpoint"]]
    manifest = dict(schema="v15pro-p4-assets.v1", status="complete", seed=1234,
        diagnostic_contract=dict(permitted_splits=["train", "val"], per_class=6,
                                 selection_uses_model_outputs=False),
        split_validation=dict(train_val_disjoint=True),
        source_schema=dict(epoch=99, inference_only=False, resume_supported=True),
        artifacts=hashes, checkpoint=paths["checkpoint"], checkpoint_sha256=source_sha,
        resolved_config=paths["config"], normalization=dict(file=paths["norm"], file_sha256=hashes[paths["norm"]]),
        datasets={name: dict(file=paths[name], samples=60, file_sha256=hashes[paths[name]]) for name in ("train", "val")},
        source_last_checkpoint=dict(sha256=source_sha, epoch=99),
        source_best_checkpoint=dict(file=paths["best"]),
        runtime=dict(source_sha256=suite.ORIGINAL_RUNTIME_SHA256))
    suite.atomic_json(folder / "manifest.json", manifest)
    return folder, manifest


def args_for(root, assets, *, run=False, resume=False):
    return Namespace(assets=str(assets), output=str(root / "suite"), data_root=str(root / "rawdata"),
        fsdd_root=str(root / "rawdata/fsdd/recordings"), device="cuda", threads=2,
        cuda_memory_fraction=.6, run=run, resume=resume)


def write_fake_stage(stage, output, source_sha):
    folder = Path(output) / stage
    folder.mkdir(parents=True, exist_ok=True)
    if stage == "tiny":
        checkpoint = folder / "checkpoints/last.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"tiny")
        (folder / "metrics.json").write_text("{}", encoding="utf-8")
        manifest = dict(status="complete", completed_updates=1000,
            identity=dict(source_checkpoint_sha256=source_sha, updates=1000, seed=1234),
            last_checkpoint_sha256=suite.file_sha256(checkpoint),
            completed_artifacts_sha256={"metrics.json": suite.file_sha256(folder / "metrics.json")})
    elif stage in ("keep_lr", "restart_lr"):
        manifest = dict(identity=dict(branch=stage, source_sha256=source_sha, epochs=30, seed=1234))
        for name in ("last.pt", "best.pt"):
            (folder / name).write_bytes(stage.encode())
        suite.atomic_json(folder / "epochs.json", [])
        suite.atomic_json(folder / "status.json", dict(complete=True, last_epoch=129, smoke_only=False))
    else:
        (folder / "report.html").write_text("report", encoding="utf-8")
        suite.atomic_json(folder / "summary.json", {})
        manifest = dict(status="complete_awaiting_scientific_review", artifacts_sha256={
            name: suite.file_sha256(folder / name) for name in ("report.html", "summary.json")})
    suite.atomic_json(folder / "manifest.json", manifest)


class SuiteTests(unittest.TestCase):
    def test_asset_hash_mismatch_and_manifest_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256):
            root = Path(temp)
            assets, manifest = fake_assets(root)
            suite.verify_assets(assets)
            checkpoint = assets / manifest["checkpoint"]
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                suite.verify_assets(assets)
            checkpoint.write_bytes(b"checkpoint")
            manifest["artifacts"]["../outside.pt"] = "0" * 64
            suite.atomic_json(assets / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "escapes"):
                suite.verify_assets(assets)
            for path in ("C:/outside.pt", "/outside.pt", "a/../../outside.pt", "..\\outside.pt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    suite.checked_child(assets, path)

    def test_commands_keep_same_original_source_and_required_order(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256):
            root = Path(temp)
            assets, manifest = fake_assets(root)
            _, paths = suite.verify_assets(assets)
            args = args_for(root, assets)
            commands = suite.commands(args, paths)
            self.assertEqual(tuple(commands), suite.STAGES)
            for name in ("keep_lr", "restart_lr"):
                command = commands[name]
                self.assertEqual(command[command.index("--source-checkpoint") + 1], str(paths["checkpoint"]))
                self.assertNotIn(str(root / "suite/tiny/checkpoints/last.pt"), command)
                self.assertEqual(command[command.index("--epochs") + 1], "30")
                self.assertEqual(command[command.index("--branch") + 1], name)
            command = commands["tiny"]
            for flag, value in (("--seed", "1234"), ("--updates", "1000"), ("--batch-size", "60"),
                                ("--eval-every", "100"), ("--lr", "5e-5")):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_plan_does_not_launch_or_read_raw_training_data(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256):
            root = Path(temp)
            assets, _ = fake_assets(root)
            args = args_for(root, assets)
            with patch.object(suite, "invoke") as invoke, patch.object(suite, "verify_training_paths") as verify, redirect_stdout(io.StringIO()):
                result = suite.run(args)
            self.assertEqual(result["status"], "planned")
            self.assertTrue((Path(args.output) / "plan.json").is_file())
            invoke.assert_not_called()
            verify.assert_not_called()

    def test_sequence_aborts_on_failure_and_preserves_prior_completion(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256))
            stack.enter_context(patch.object(suite, "verify_training_paths", return_value={"verified": True}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            root = Path(temp)
            assets, manifest = fake_assets(root)
            args = args_for(root, assets, run=True)
            calls = []
            def invoke(stage, command, log, args, lock):
                calls.append(stage)
                self.assertEqual(Path(log).parent, root / "suite/logs")
                if stage == "keep_lr":
                    raise RuntimeError("simulated failed stage")
                write_fake_stage(stage, args.output, manifest["checkpoint_sha256"])
            stack.enter_context(patch.object(suite, "invoke", side_effect=invoke))
            with self.assertRaisesRegex(RuntimeError, "simulated failed stage"):
                suite.run(args)
            self.assertEqual(calls, ["tiny", "keep_lr"])
            saved = suite.read_json(root / "suite/suite_manifest.json")
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["stages"]["tiny"]["status"], "complete")
            self.assertEqual(saved["stages"]["keep_lr"]["status"], "failed")
            self.assertEqual(saved["stages"]["restart_lr"]["status"], "pending")
            self.assertFalse((root / "suite/.p4_suite.lock").exists())

    def test_completed_resume_invokes_each_stage_validator_and_changed_identity_fails(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256))
            stack.enter_context(patch.object(suite, "verify_training_paths", return_value={"verified": True}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            root = Path(temp)
            assets, manifest = fake_assets(root)
            args = args_for(root, assets, run=True)
            calls = []
            def invoke(stage, command, log, args, lock):
                calls.append((stage, command))
                if not (Path(args.output) / stage).exists():
                    write_fake_stage(stage, args.output, manifest["checkpoint_sha256"])
            stack.enter_context(patch.object(suite, "invoke", side_effect=invoke))
            result = suite.run(args)
            self.assertEqual(result["status"], "complete_awaiting_scientific_review")
            self.assertEqual([stage for stage, _ in calls], list(suite.STAGES))
            calls.clear()
            args.resume = True
            suite.run(args)
            self.assertEqual([stage for stage, _ in calls], list(suite.STAGES))
            for stage, command in calls:
                self.assertEqual("--resume" in command, stage != "report")
            args.threads = 3
            calls.clear()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                suite.run(args)
            self.assertEqual(calls, [])

    def test_partial_stage_without_checkpoint_and_missing_manifest_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "keep_lr"
            folder.mkdir()
            suite.atomic_json(folder / "manifest.json", {})
            with self.assertRaisesRegex(ValueError, "partial initialization"):
                suite.command_for_state("keep_lr", ["python"], root, True)
            (folder / "manifest.json").unlink()
            (folder / "last.pt").touch()
            with self.assertRaisesRegex(ValueError, "no provenance manifest"):
                suite.command_for_state("keep_lr", ["python"], root, True)

    def test_lock_rejects_live_parent_or_orphan_child_and_reclaims_dead_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with suite.running_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with suite.running_lock(root):
                        pass
            lock = root / ".p4_suite.lock"
            suite.atomic_json(lock, dict(hostname=socket.gethostname(), pid=88888, child_pid=77777))
            with patch.object(suite, "pid_alive", side_effect=lambda pid: pid == 77777):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with suite.running_lock(root):
                        pass
            with patch.object(suite, "pid_alive", return_value=False):
                with suite.running_lock(root):
                    self.assertEqual(suite.read_json(lock)["pid"], os.getpid())
            self.assertFalse(lock.exists())
            self.assertEqual(len(list(root.glob(".p4_suite.stale_*.json"))), 1)

    def test_final_stage_validation_rejects_smoke_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fake_stage("keep_lr", root, "a" * 64)
            status = root / "keep_lr/status.json"
            record = suite.read_json(status)
            record["smoke_only"] = True
            suite.atomic_json(status, record)
            with self.assertRaisesRegex(ValueError, "30 formal epochs"):
                suite.validate_completed("keep_lr", root, "a" * 64)

    def test_gpu_pid_query_is_fresh_and_failures_are_never_idle(self):
        completed = Namespace(returncode=0, stdout=f"{os.getpid()}\n777\n777\n", stderr="")
        with patch.object(suite.subprocess, "run", return_value=completed) as execute:
            self.assertEqual(suite.foreign_gpu_pids(), [777])
            self.assertEqual(execute.call_args.args[0],
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        for result in (Namespace(returncode=1, stdout="", stderr="driver failure"),
                       Namespace(returncode=0, stdout="N/A\n", stderr="")):
            with self.subTest(result=result), patch.object(suite.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
                    suite.foreign_gpu_pids()
        with patch.object(suite.subprocess, "run", side_effect=FileNotFoundError("missing executable")):
            with self.assertRaisesRegex(RuntimeError, "Unable to query GPU"):
                suite.foreign_gpu_pids()

    def test_gpu_barrier_only_applies_to_enabled_cuda_continuation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, "foreign_gpu_pids") as query, patch.object(suite, "cuda_available") as available:
            args = Namespace(wait_for_gpu_idle_before_continuation=True, device="cuda")
            update = lambda pids: None
            for stage in ("tiny", "report"):
                suite.wait_for_continuation_gpu(stage, args, update, Path(temp) / "wait.log")
            args.device = "cpu"
            suite.wait_for_continuation_gpu("keep_lr", args, update, Path(temp) / "wait.log")
            args.device = "cuda:0"
            args.wait_for_gpu_idle_before_continuation = False
            suite.wait_for_continuation_gpu("keep_lr", args, update, Path(temp) / "wait.log")
            query.assert_not_called()
            available.assert_not_called()
            args.wait_for_gpu_idle_before_continuation = True
            available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                suite.wait_for_continuation_gpu("restart_lr", args, update, Path(temp) / "wait.log")

    def test_gpu_wait_rechecks_exiting_pids_preserves_child_order_and_records_status(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(suite, "current_runtime_sha256", return_value=suite.ORIGINAL_RUNTIME_SHA256))
            stack.enter_context(patch.object(suite, "verify_training_paths", return_value={"verified": True}))
            stack.enter_context(patch.object(suite, "cuda_available", return_value=True))
            stack.enter_context(redirect_stdout(io.StringIO()))
            root = Path(temp)
            assets, manifest = fake_assets(root)
            args = args_for(root, assets, run=True)
            args.wait_for_gpu_idle_before_continuation = True
            args.cuda_memory_fraction = .95
            events, snapshots = [], []
            replies = iter([[466887], [], [888], []])
            def query():
                pids = next(replies)
                events.append(("query", pids))
                return pids
            def sleep(seconds):
                self.assertEqual(seconds, 10)
                snapshots.append(suite.read_json(root / "suite/suite_manifest.json"))
            def invoke(stage, command, log, args, lock):
                events.append(("child", stage))
                write_fake_stage(stage, args.output, manifest["checkpoint_sha256"])
            stack.enter_context(patch.object(suite, "foreign_gpu_pids", side_effect=query))
            stack.enter_context(patch.object(suite.time, "sleep", side_effect=sleep))
            stack.enter_context(patch.object(suite, "invoke", side_effect=invoke))
            result = suite.run(args)
            self.assertEqual(events, [("child", "tiny"), ("query", [466887]), ("query", []),
                ("child", "keep_lr"), ("query", [888]), ("query", []),
                ("child", "restart_lr"), ("child", "report")])
            self.assertEqual([row["status"] for row in snapshots], ["waiting_for_gpu"] * 2)
            self.assertEqual([row["active_stage"] for row in snapshots], ["keep_lr", "restart_lr"])
            self.assertEqual([row["gpu_foreign_pids"] for row in snapshots], [[466887], [888]])
            self.assertTrue(result["identity"]["wait_for_gpu_idle_before_continuation"])
            self.assertEqual(result["identity"]["cuda_memory_fraction"], .95)
            self.assertEqual(result["status"], "complete_awaiting_scientific_review")


if __name__ == "__main__":
    unittest.main()
