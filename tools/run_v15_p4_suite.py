"""Plan, execute and strictly resume the isolated seed1234 P4 experiment suite.

Execution is opt-in via --run. The tiny diagnostic is followed by two separate
30-epoch branches from the same original last.pt, then the audited report.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

ORIGINAL_RUNTIME_SHA256 = "b9b072bdfc1aa26fe1b3c95976dde86364a1b131b3845cb39e5be1325f8244d7"
STAGES = ("tiny", "keep_lr", "restart_lr", "report")
SOURCE_FILES = ("run_v15_p4_suite.py", "run_v15_p4_tiny.py", "v15_p4_diagnostics.py",
                "train_v15_p4.py", "report_v15_p4.py", "package_v15_p4_assets.py",
                "package_v15_eval_assets.py", "v15_foreground_protocol.py",
                "evaluate_v15_foreground.py", "export_v15pro.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_child(root, relative):
    """Manifest paths must be portable relatives and cannot escape via symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid relative artifact path: {relative!r}")
    if (PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or ".." in PurePosixPath(relative).parts):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def verify_hashes(root, artifacts):
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("A nonempty artifact hash manifest is required")
    for relative, expected in artifacts.items():
        path = checked_child(root, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact missing or SHA256 mismatch: {path}")


def current_runtime_sha256():
    from tools.package_v15_eval_assets import runtime_fingerprint
    return runtime_fingerprint()["source_sha256"]


def verify_assets(assets):
    assets = Path(assets).resolve()
    manifest = read_json(assets / "manifest.json")
    if (manifest.get("schema") != "v15pro-p4-assets.v1" or manifest.get("status") != "complete"
            or manifest.get("seed") != 1234):
        raise ValueError("Expected complete seed1234 P4 assets")
    contract = manifest.get("diagnostic_contract", {})
    if (contract.get("permitted_splits") != ["train", "val"] or contract.get("per_class") != 6
            or contract.get("selection_uses_model_outputs") is not False
            or not manifest.get("split_validation", {}).get("train_val_disjoint")):
        raise ValueError("Expected fixed six-per-class train60/val60 with disjoint membership")
    schema = manifest.get("source_schema", {})
    if (schema.get("epoch") != 99 or schema.get("inference_only") is not False
            or schema.get("resume_supported") is not True):
        raise ValueError("P4 requires the full original epoch-100 training checkpoint")
    verify_hashes(assets, manifest.get("artifacts"))
    roles = {"checkpoint": manifest["checkpoint"], "config": manifest["resolved_config"],
             "norm": manifest["normalization"]["file"],
             "train": manifest["datasets"]["train"]["file"],
             "val": manifest["datasets"]["val"]["file"]}
    best = manifest.get("source_best_checkpoint", {})
    if not best.get("file"):
        raise ValueError("P4 reporting requires the packaged original best_reference checkpoint")
    roles["best_reference"] = best["file"]
    for role, relative in roles.items():
        checked_child(assets, relative)
        if relative not in manifest["artifacts"]:
            raise ValueError(f"Asset role is not protected by the artifact manifest: {role}")
    if (manifest["artifacts"][roles["checkpoint"]] != manifest["checkpoint_sha256"]
            or manifest["checkpoint_sha256"] != manifest["source_last_checkpoint"]["sha256"]
            or manifest["source_last_checkpoint"]["epoch"] != 99):
        raise ValueError("Packaged checkpoint provenance disagrees with original last.pt")
    if manifest["artifacts"][roles["norm"]] != manifest["normalization"]["file_sha256"]:
        raise ValueError("Normalization provenance mismatch")
    for split in ("train", "val"):
        record = manifest["datasets"][split]
        if (record.get("samples") != 60
                or record["file_sha256"] != manifest["artifacts"][roles[split]]):
            raise ValueError(f"Invalid {split}60 data provenance")
    runtime = current_runtime_sha256()
    if runtime != ORIGINAL_RUNTIME_SHA256 or manifest["runtime"]["source_sha256"] != runtime:
        raise ValueError("Original v15 runtime differs from the packaged source")
    return manifest, {role: checked_child(assets, relative) for role, relative in roles.items()}


def verify_training_paths(args, manifest, paths):
    """Raw-data verification reads train bytes and existing normalization only."""
    from common import load_config
    from tools.package_v15_p4_assets import verify_mapped_training_data
    cfg = load_config(paths["config"])
    cfg = deepcopy(cfg)
    cfg["data"]["root"] = str(Path(args.data_root).resolve())
    cfg["audio"]["fsdd_root"] = str(Path(args.fsdd_root).resolve())
    cfg["audio"]["norm_stats_path"] = str(paths["norm"])
    cfg["audio"]["auto_download"] = False
    return verify_mapped_training_data(cfg, manifest, paths["norm"])


def commands(args, paths):
    output = Path(args.output).resolve()
    python = sys.executable
    common = ["--device", args.device, "--cuda-memory-fraction", str(args.cuda_memory_fraction)]
    tiny = [python, "-u", str(ROOT / "tools/run_v15_p4_tiny.py"),
            "--checkpoint", str(paths["checkpoint"]), "--train-npz", str(paths["train"]),
            "--val-npz", str(paths["val"]), "--output-dir", str(checked_child(output, "tiny")),
            "--updates", "1000", "--eval-every", "100", "--batch-size", "60",
            "--lr", "5e-5", "--seed", "1234", *common]
    result = {"tiny": tiny}
    for branch in STAGES[1:3]:
        result[branch] = [python, "-u", str(ROOT / "tools/train_v15_p4.py"),
            "--source-checkpoint", str(paths["checkpoint"]),
            "--data-root", str(Path(args.data_root).resolve()),
            "--fsdd-root", str(Path(args.fsdd_root).resolve()), "--norm-stats", str(paths["norm"]),
            "--output", str(checked_child(output, branch)), "--branch", branch,
            "--epochs", "30", "--threads", str(args.threads), *common]
    result["report"] = [python, "-u", str(ROOT / "tools/report_v15_p4.py"),
        "--root", str(output), "--assets", str(Path(args.assets).resolve()),
        "--threads", str(args.threads), *common]
    return result


def suite_identity(args, manifest, paths, plan):
    return dict(schema="v15-p4-suite.v1", assets=str(Path(args.assets).resolve()),
        asset_manifest_sha256=file_sha256(Path(args.assets) / "manifest.json"),
        original_checkpoint_sha256=manifest["checkpoint_sha256"],
        runtime_sha256=manifest["runtime"]["source_sha256"],
        source_files_sha256={name: file_sha256(ROOT / "tools" / name) for name in SOURCE_FILES},
        output=str(Path(args.output).resolve()), data_root=str(Path(args.data_root).resolve()),
        fsdd_root=str(Path(args.fsdd_root).resolve()), seed=1234,
        device=args.device, threads=args.threads,
        cuda_memory_fraction=args.cuda_memory_fraction,
        wait_for_gpu_idle_before_continuation=bool(getattr(args, "wait_for_gpu_idle_before_continuation", False)),
        tiny=dict(updates=1000, batch_size=60, eval_every=100, base_lr=5e-5),
        branches=dict(keep_lr_epochs=30, restart_lr_epochs=30,
                      both_source_checkpoint=str(paths["checkpoint"]), optimizer="original source state"),
        commands=plan, python_executable=sys.executable)


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a safe liveness probe on Windows.
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return kernel.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def running_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".p4_suite.lock"
    token = uuid.uuid4().hex
    record = dict(pid=os.getpid(), child_pid=None, hostname=socket.gethostname(),
                  token=token, created_utc=now())
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            break
        except FileExistsError:
            old = read_json(path)
            if old.get("hostname") != socket.gethostname():
                raise RuntimeError("Suite lock belongs to another host; liveness cannot be verified")
            if (not isinstance(old.get("pid"), int) or old["pid"] <= 0
                    or (old.get("child_pid") is not None and
                        (not isinstance(old["child_pid"], int) or old["child_pid"] <= 0))):
                raise RuntimeError("Malformed suite lock has no verifiable process owner")
            if pid_alive(old.get("pid")) or pid_alive(old.get("child_pid")):
                raise RuntimeError(f"P4 suite already running: parent={old.get('pid')} child={old.get('child_pid')}")
            if attempt or read_json(path) != old:
                raise RuntimeError("P4 suite lock changed while checking its stale owner")
            path.rename(output / (".p4_suite.stale_" + uuid.uuid4().hex + ".json"))
    else:
        raise RuntimeError("Could not acquire P4 suite lock")
    def update(child_pid=None, stage=None):
        if not path.is_file() or read_json(path).get("token") != token:
            raise RuntimeError("P4 suite lost ownership of its running lock")
        record.update(child_pid=child_pid, stage=stage)
        atomic_json(path, record)
    try:
        yield update
    finally:
        # If this launcher is interrupted while a child continues, retain its PID
        # so a later launcher cannot mistake the dead parent for an idle suite.
        if path.exists() and read_json(path).get("token") == token:
            if not pid_alive(record.get("child_pid")):
                path.unlink()


def cuda_available():
    import torch
    return torch.cuda.is_available()


def foreign_gpu_pids():
    """Read current compute users anew; a query failure never means an idle GPU."""
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Unable to query GPU compute processes with nvidia-smi: {error}") from error
    if result.returncode:
        raise RuntimeError(f"nvidia-smi compute-process query failed ({result.returncode}): {result.stderr.strip()}")
    pids = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.isdecimal() or int(line) <= 0:
            raise RuntimeError(f"Unexpected nvidia-smi process PID output: {line!r}")
        pids.add(int(line))
    return sorted(pids - {os.getpid()})


def wait_for_continuation_gpu(stage, args, state_update, log_path):
    """Manual suite barrier: tiny proceeds now; full A/B waits for current users."""
    if (stage not in ("keep_lr", "restart_lr")
            or not getattr(args, "wait_for_gpu_idle_before_continuation", False)
            or str(args.device).split(":", 1)[0] != "cuda"):
        return
    if not cuda_available():
        raise RuntimeError("CUDA is unavailable while waiting for continuation; no CPU fallback")
    previous, logged_at = None, float("-inf")
    while True:
        pids = foreign_gpu_pids()
        state_update(pids)
        current = time.monotonic()
        if pids != previous or current - logged_at >= 60:
            message = (f"[P4 suite] {stage}: waiting for GPU compute PIDs {pids}"
                       if pids else f"[P4 suite] {stage}: GPU has no foreign compute process; continuing")
            print(message, flush=True)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(f"{now()} {message}\n")
            previous, logged_at = pids, current
        if not pids:
            return
        time.sleep(10)


def invoke(stage, command, log_path, args, lock_update):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps(dict(time=now(), stage=stage, argv=command), ensure_ascii=False) + "\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, shell=False)
        lock_update(child_pid=child.pid, stage=stage)
        code = child.wait()
        lock_update(child_pid=None, stage=stage)
    if code:
        raise RuntimeError(f"P4 stage {stage} exited with code {code}; see {log_path}")


def command_for_state(stage, base_command, output, resume):
    folder = checked_child(output, stage)
    if not folder.exists() or not any(folder.iterdir()):
        return list(base_command)
    if not resume:
        raise FileExistsError(f"Existing stage {stage} requires suite --resume")
    if not (folder / "manifest.json").is_file():
        raise ValueError(f"Existing {stage} directory has no provenance manifest; refusing overwrite")
    if stage == "report":
        return list(base_command)  # Report validates its own manifest and caches.
    last = folder / ("checkpoints/last.pt" if stage == "tiny" else "last.pt")
    if not last.is_file():
        raise ValueError(f"{stage} has partial initialization but no resumable checkpoint; use a new suite output")
    return [*base_command, "--resume"]


def validate_completed(stage, output, source_sha256):
    folder = checked_child(output, stage)
    manifest = read_json(folder / "manifest.json")
    if stage == "tiny":
        identity = manifest["identity"]
        if (manifest.get("status") != "complete" or manifest.get("completed_updates") != 1000
                or identity.get("source_checkpoint_sha256") != source_sha256
                or identity.get("updates") != 1000 or identity.get("seed") != 1234):
            raise ValueError("Tiny process returned without completing the prescribed diagnostic")
        if file_sha256(folder / "checkpoints/last.pt") != manifest["last_checkpoint_sha256"]:
            raise ValueError("Tiny final checkpoint hash mismatch")
        verify_hashes(folder, manifest["completed_artifacts_sha256"])
    elif stage in STAGES[1:3]:
        status = read_json(folder / "status.json")
        identity = manifest["identity"]
        if (status.get("complete") is not True or status.get("last_epoch") != 129
                or status.get("smoke_only") is not False
                or identity.get("branch") != stage or identity.get("source_sha256") != source_sha256
                or identity.get("epochs") != 30 or identity.get("seed") != 1234):
            raise ValueError(f"{stage} process returned without completing all 30 formal epochs")
        for name in ("best.pt", "last.pt", "epochs.json"):
            if not (folder / name).is_file():
                raise ValueError(f"{stage} missing completed artifact {name}")
    else:
        if manifest.get("status") != "complete_awaiting_scientific_review":
            raise ValueError("Report process returned without its complete audit artifacts")
        verify_hashes(folder, manifest["artifacts_sha256"])
    return dict(validated_utc=now(), manifest_sha256=file_sha256(folder / "manifest.json"))


def run(args):
    if args.threads < 1 or not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("threads must be positive and cuda-memory-fraction must lie in (0,1]")
    output, assets = Path(args.output).resolve(), Path(args.assets).resolve()
    if output == assets or output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Suite output and source assets must be separate directory trees")
    for raw_root in (Path(args.data_root).resolve(), Path(args.fsdd_root).resolve()):
        if output == raw_root or raw_root.is_relative_to(output) or output.is_relative_to(raw_root):
            raise ValueError("Suite output must be separate from raw data directories")
    manifest, paths = verify_assets(assets)
    plan = commands(args, paths)
    identity = suite_identity(args, manifest, paths, plan)
    suite_path = output / "suite_manifest.json"

    def prepare():
        if suite_path.exists():
            suite = read_json(suite_path)
            if suite.get("identity") != identity:
                raise ValueError("P4 suite resume identity mismatch: sources, parameters or script hashes changed")
            if suite.get("status") != "planned" and not args.resume:
                raise ValueError("Existing executed P4 suite requires --resume")
            return suite
        if args.resume:
            raise ValueError("--resume requires an existing matching suite_manifest.json")
        allowed = {".p4_suite.lock"}
        if output.exists() and any(p.name not in allowed and not p.name.startswith(".p4_suite.stale_")
                                   for p in output.iterdir()):
            raise FileExistsError("Existing output is not a recognized P4 suite; refusing overwrite")
        suite = dict(identity=identity, status="planned", created_utc=now(),
                     training_data_verified=False,
                     stages={stage: dict(status="pending") for stage in STAGES})
        atomic_json(suite_path, suite)
        return suite

    if not args.run:
        suite = prepare()
        atomic_json(output / "plan.json", dict(identity=identity, execution_requested=False,
            stage_order=list(STAGES), commands=plan,
            training_data_verification="required before --run; no training started"))
        print(json.dumps(dict(plan=str(output / "plan.json"), status=suite["status"],
                              commands=plan), ensure_ascii=False, indent=2), flush=True)
        return suite

    with running_lock(output) as lock_update:
        suite = prepare()
        try:
            suite["training_data_verification"] = verify_training_paths(args, manifest, paths)
            suite["training_data_verified"] = True
            suite.update(status="running", updated_utc=now())
            atomic_json(suite_path, suite)
            for stage in STAGES:
                suite["active_stage"] = stage
                if suite_identity(args, manifest, paths, plan) != identity:
                    raise ValueError("P4 source scripts or parameters changed during execution")
                if file_sha256(paths["checkpoint"]) != manifest["checkpoint_sha256"]:
                    raise ValueError("Original source checkpoint changed before the next P4 stage")
                def gpu_state(pids):
                    lock_update(child_pid=None, stage=stage)
                    if read_json(suite_path).get("identity") != identity:
                        raise ValueError("P4 suite identity changed during GPU resource waiting")
                    suite.update(status="waiting_for_gpu" if pids else "running",
                                 active_stage=stage, gpu_foreign_pids=pids, gpu_checked_utc=now())
                    suite["stages"][stage].update(status="waiting_for_gpu" if pids else "pending",
                                                  gpu_foreign_pids=pids)
                    atomic_json(suite_path, suite)
                wait_for_continuation_gpu(stage, args, gpu_state, output / "logs/gpu_wait.log")
                # Waiting can span another job's completion; ownership and provenance
                # are checked again before this process launches a training child.
                lock_update(child_pid=None, stage=stage)
                if (read_json(suite_path).get("identity") != identity
                        or suite_identity(args, manifest, paths, plan) != identity):
                    raise ValueError("P4 suite identity changed while waiting for GPU resources")
                if file_sha256(paths["checkpoint"]) != manifest["checkpoint_sha256"]:
                    raise ValueError("Original source checkpoint changed while waiting for GPU resources")
                command = command_for_state(stage, plan[stage], output, args.resume)
                suite["status"] = "running"
                suite["stages"][stage].update(status="running", started_utc=now(), command=command)
                suite["active_stage"] = stage
                atomic_json(suite_path, suite)
                print(f"[P4 suite] {stage}: {output / 'logs' / (stage + '.log')}", flush=True)
                invoke(stage, command, output / "logs" / (stage + ".log"), args, lock_update)
                evidence = validate_completed(stage, output, manifest["checkpoint_sha256"])
                suite["stages"][stage].update(status="complete", finished_utc=now(), **evidence)
                atomic_json(suite_path, suite)
            # Revalidate source bytes after all child processes before declaring completion.
            verify_assets(assets)
            if suite_identity(args, manifest, paths, plan) != identity:
                raise ValueError("P4 source scripts changed during execution")
            suite.update(status="complete_awaiting_scientific_review", active_stage=None,
                         completed_utc=now(), report="report/report.html")
            atomic_json(suite_path, suite)
            return suite
        except BaseException as error:
            stage = suite.get("active_stage")
            if stage:
                suite["stages"][stage].update(status="failed", failed_utc=now(), error=str(error))
            suite.update(status="failed", updated_utc=now(), error=str(error))
            atomic_json(suite_path, suite)
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--assets", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--data-root", required=True, help="Original MNIST root; verified before execution")
    result.add_argument("--fsdd-root", required=True, help="FSDD recordings directory or its parent")
    result.add_argument("--device", default="cuda")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--cuda-memory-fraction", type=float, default=.6)
    result.add_argument("--wait-for-gpu-idle-before-continuation", action="store_true",
                        help="Run tiny immediately; poll every 10s for no foreign GPU compute PIDs before each A/B branch")
    result.add_argument("--run", action="store_true", help="Execute the plan; omitted means plan only")
    result.add_argument("--resume", action="store_true", help="Strictly resume an existing identical suite")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
