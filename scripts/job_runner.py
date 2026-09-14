"""Run one logged child process and stop the suite on failure."""

import os
import subprocess
import sys

from paths import PROJECT_ROOT


def run_job(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f"[suite] start: {' '.join(command)}\n[suite] log: {logfile}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u"] + command,
            cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)
    print(f"[suite] done: {command[0]}", flush=True)
