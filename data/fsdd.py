"""FSDD（Free Spoken Digit Dataset）下载与路径工具。"""

import glob
import io
import os
import shutil
import subprocess
import zipfile

import urllib.request

FSDD_REPO = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
FSDD_ZIP = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"


def fsdd_recordings_dir(cfg):
    """返回 FSDD wav 目录（含末尾 recordings）。"""
    root = cfg["audio"]["fsdd_root"]
    if root.rstrip("/\\").endswith("recordings"):
        return root
    return os.path.join(root, "recordings")


def count_wav_files(recordings_dir):
    return len(glob.glob(os.path.join(recordings_dir, "*.wav")))


def _clone_fsdd(parent, verbose=True):
    if verbose:
        print(f"[FSDD] 尝试 git clone 到 {parent} …", flush=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", FSDD_REPO, parent],
            check=True, capture_output=True, text=True, timeout=300,
        )
        return True
    except FileNotFoundError:
        if verbose:
            print("[FSDD] 未安装 git，改用 zip 下载。", flush=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if verbose:
            print(f"[FSDD] git clone 失败: {e}", flush=True)
    return False


def _download_zip_fsdd(parent, verbose=True):
    """无 git 时从 GitHub 下载 zip 并解压。"""
    if verbose:
        print(f"[FSDD] 正在下载 zip …", flush=True)
    try:
        with urllib.request.urlopen(FSDD_ZIP, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            top = zf.namelist()[0].split("/")[0]
            extract_tmp = parent + "_zip_tmp"
            if os.path.isdir(extract_tmp):
                shutil.rmtree(extract_tmp)
            zf.extractall(extract_tmp)
            src = os.path.join(extract_tmp, top)
            if os.path.isdir(parent):
                shutil.rmtree(parent, ignore_errors=True)
            shutil.move(src, parent)
            shutil.rmtree(extract_tmp, ignore_errors=True)
        if verbose:
            print(f"[FSDD] zip 解压完成 -> {parent}", flush=True)
        return True
    except Exception as e:
        if verbose:
            print(f"[FSDD] zip 下载失败: {e}", flush=True)
        return False


def ensure_fsdd(cfg, verbose=True):
    """若本地无 wav，自动下载 FSDD。成功返回 recordings 目录，否则 None。"""
    rec = fsdd_recordings_dir(cfg)
    if count_wav_files(rec) > 0:
        return rec

    parent = os.path.dirname(rec.rstrip("/\\"))
    os.makedirs(parent, exist_ok=True)

    if verbose:
        print(f"[FSDD] 未找到 wav（{rec}），开始自动下载 …", flush=True)

    ok = _clone_fsdd(parent, verbose) or _download_zip_fsdd(parent, verbose)
    if not ok:
        if verbose:
            print("[FSDD] 自动下载失败。请手动下载并解压到：", flush=True)
            print(f"  {FSDD_ZIP}", flush=True)
            print(f"  目标目录: {parent}（需含 recordings/*.wav）", flush=True)
        return None

    # zip/git 解压后 recordings 可能在子目录，向上搜索一次
    if count_wav_files(rec) == 0:
        for root, _dirs, files in os.walk(parent):
            if any(f.lower().endswith(".wav") for f in files):
                candidate = root if any(
                    f.lower().endswith(".wav") for f in os.listdir(root)) else None
                if candidate and count_wav_files(candidate) > 0:
                    rec = candidate
                    if verbose:
                        print(f"[FSDD] 在 {rec} 找到 wav。", flush=True)
                    break

    if count_wav_files(rec) > 0:
        if verbose:
            print(f"[FSDD] 就绪，共 {count_wav_files(rec)} 条 wav。", flush=True)
        return rec
    if verbose:
        print(f"[FSDD] 下载完成但未在 {rec} 找到 wav。", flush=True)
    return None
