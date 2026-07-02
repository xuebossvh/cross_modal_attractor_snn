"""将评估结果转为表格图 + CSV。

支持两种输入（自动识别）：
  1. demo_eval_table.txt  — 【汇总】段（8 列）
  2. eval_*.log           — evaluate.py 全量评估日志（9 列）
  3. 上述日志若含 [音频塌缩诊断] 段，自动生成 aud_diag 表格（PNG+CSV）

用法：
    python scripts/plot_eval_summary.py outputs/outputs_v9b/tables/demo_eval_table.txt
    python scripts/plot_eval_summary.py outputs/outputs_v9b/logs/eval_v9b_fixed_mask.log
    python scripts/plot_eval_summary.py eval_v9b_full.log --title "v9b full eval"
    python scripts/plot_eval_summary.py eval_v9b_fixed_mask.log --diag-only
"""

import argparse
import csv
import re
from pathlib import Path

import bootstrap  # noqa: F401

import matplotlib.pyplot as plt

from common import fix_console_encoding, log, setup_matplotlib_chinese

_EVAL_MODES = (
    "corrupt_img_only", "corrupt_aud_only", "corrupt_both",
    "clean_img_only", "clean_aud_only", "clean_both",
)
_FULL_ROW_RE = re.compile(
    r"(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"(\d+\.\d+)%\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"(\S+)",
)
_AUD_DIAG_ROW_RE = re.compile(
    r"^(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)%\s*$",
)


def parse_eval_summary(path):
    """解析 demo_eval_table.txt【汇总】段。"""
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    in_block = False
    past_sep = False
    for line in text.splitlines():
        if "【汇总】" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip().startswith("-"):
            past_sep = True
            continue
        if not past_sep or not line.strip():
            continue
        if line.strip().startswith("【"):
            break
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) < 6:
            continue
        rows.append({
            "mode": cols[0],
            "acc": float(cols[1].rstrip("%")) / 100.0,
            "img_ssim": float(cols[2]),
            "img_mse": float(cols[3]),
            "aud_ssim": float(cols[4]),
            "aud_mse": float(cols[5]),
            "img_tgt": cols[6] if len(cols) > 6 else "",
            "aud_tgt": cols[7] if len(cols) > 7 else "",
        })
    if not rows:
        raise ValueError(f"未在 {path} 中找到【汇总】数据。")
    return rows, "demo"


def parse_eval_full_log(path):
    """解析 evaluate.py 全量评估日志中的 6 行结果。"""
    text = Path(path).read_text(encoding="utf-8")
    by_mode = {}
    severity = None
    n_test = None

    for line in text.splitlines():
        m = re.search(r"corrupt severity=([\d.]+)", line)
        if m:
            severity = float(m.group(1))
        m = re.search(r"\[dataset\] test.*n=(\d+)", line)
        if m:
            n_test = int(m.group(1))
        m = _FULL_ROW_RE.search(line)
        if m:
            by_mode[m.group(1)] = {
                "mode": m.group(1),
                "acc": float(m.group(2)) / 100.0,
                "img_mse": float(m.group(3)),
                "psnr": float(m.group(4)),
                "img_ssim": float(m.group(5)),
                "aud_mse": float(m.group(6)),
                "pix_var": float(m.group(7)),
                "pair_l2": float(m.group(8)),
                "tgt": m.group(9),
            }

    rows = [by_mode[m] for m in _EVAL_MODES if m in by_mode]
    if not rows:
        raise ValueError(f"未在 {path} 中找到 evaluate.py 结果行。")
    meta = {"severity": severity, "n_test": n_test}
    return rows, "full", meta


def parse_aud_collapse_diag(path):
    """解析 evaluate.py / demo 日志末尾的 [音频塌缩诊断] 段。"""
    rows = []
    in_block = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "音频塌缩诊断" in line:
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("[") or stripped.startswith("【"):
            if rows:
                break
            continue
        if stripped.startswith("-") or "rec均值" in stripped or "rec_std" in stripped:
            continue
        m = _AUD_DIAG_ROW_RE.match(stripped)
        if m:
            rows.append({
                "mode": m.group(1),
                "rec_mean": float(m.group(2)),
                "rec_std": float(m.group(3)),
                "rec_max": float(m.group(4)),
                "tgt_mean": float(m.group(5)),
                "tgt_std": float(m.group(6)),
                "tgt_max": float(m.group(7)),
                "topk_recall": float(m.group(8)) / 100.0,
            })
    if not rows:
        raise ValueError(f"未在 {path} 中找到 [音频塌缩诊断] 数据行。")
    return rows


def detect_input(path):
    text = Path(path).read_text(encoding="utf-8")
    if "【汇总】" in text:
        return parse_eval_summary(path)
    if any(m in text for m in _EVAL_MODES) and "imgMSE" in text:
        return parse_eval_full_log(path)
    # 尝试 full log（无表头时仅靠结果行）
    try:
        return parse_eval_full_log(path)
    except ValueError:
        return parse_eval_summary(path)


def _save_table_figure(fig, path, *, pad_inches=0.08):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad_inches,
                facecolor="white")
    plt.close(fig)


def _style_metric_table(tbl, ncols, col_widths, font_size=9):
    for j, w in enumerate(col_widths[:ncols]):
        for row in range(len(tbl.get_celld()) // ncols):
            key = (row, j)
            if key in tbl.get_celld():
                tbl.get_celld()[key].set_width(w)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(weight="bold", fontsize=font_size, color="#222222")
        else:
            cell.set_facecolor("#ffffff")
            if col == 0:
                cell.set_text_props(ha="left", fontsize=font_size, color="#222222")
                cell.PAD = 0.06
            else:
                cell.set_text_props(ha="center", fontsize=font_size, color="#222222")


def _render_table(headers, cell, col_widths, title, path, font_size=9):
    nrows = len(cell) + 1
    ncols = len(headers)
    fig_w = max(11.2, 0.95 * ncols)
    fig_h = 1.05 + 0.22 * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.78])
    ax.axis("off")
    fig.text(
        0.5, 0.995, title, ha="center", va="top",
        fontsize=12, fontweight="medium", color="#222222",
    )

    tbl = ax.table(
        cellText=cell, colLabels=headers, cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0], colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.35)
    _style_metric_table(tbl, ncols, col_widths, font_size=font_size)
    _save_table_figure(fig, path, pad_inches=0.15)

    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(cell)
    return path, csv_path


def render_demo_table(rows, title, path):
    headers = ["", "acc", "img SSIM", "img MSE", "aud SSIM", "aud MSE",
               "img target", "aud target"]
    cell = [[
        r["mode"], f"{r['acc']:.3f}",
        f"{r['img_ssim']:.3f}", f"{r['img_mse']:.4f}",
        f"{r['aud_ssim']:.3f}", f"{r['aud_mse']:.4f}",
        r["img_tgt"], r["aud_tgt"],
    ] for r in rows]
    col_w = [0.13, 0.09, 0.11, 0.11, 0.11, 0.11, 0.165, 0.165]
    return _render_table(headers, cell, col_w, title, path)


_KIND_ABBR = {"sam": "sample", "cat": "category"}


def _expand_target(tgt):
    """sam/cat -> sample/category（兼容旧日志缩写）。"""
    return "/".join(_KIND_ABBR.get(p.strip(), p.strip()) for p in tgt.split("/"))


def render_aud_diag_table(rows, title, path):
    headers = ["", "rec mean", "rec std", "rec max",
               "tgt mean", "tgt std", "tgt max", "top15% recall"]
    cell = [[
        r["mode"],
        f"{r['rec_mean']:.4f}", f"{r['rec_std']:.4f}", f"{r['rec_max']:.4f}",
        f"{r['tgt_mean']:.4f}", f"{r['tgt_std']:.4f}", f"{r['tgt_max']:.4f}",
        f"{r['topk_recall']:.3f}",
    ] for r in rows]
    col_w = [0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12]
    return _render_table(headers, cell, col_w, title, path, font_size=8.5)


def _default_aud_diag_out(path, main_out=None):
    if main_out is not None:
        p = Path(main_out)
        return p.with_name(f"{p.stem}_aud_diag{p.suffix}")
    p = Path(path)
    return p.parent.parent / "tables" / "aud_collapse_table.png"


def render_full_eval_table(rows, title, path):
    headers = ["", "acc", "img MSE", "PSNR", "img SSIM", "aud MSE",
               "target(image/audio)"]
    cell = [[
        r["mode"], f"{r['acc']:.3f}",
        f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['img_ssim']:.3f}",
        f"{r['aud_mse']:.4f}", _expand_target(r["tgt"]),
    ] for r in rows]
    col_w = [0.14, 0.08, 0.10, 0.08, 0.10, 0.10, 0.18]
    return _render_table(headers, cell, col_w, title, path, font_size=8.5)


def _parse_n_samples(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"样本数:\s*(\d+)", line.strip())
        if m:
            return int(m.group(1))
    return None


def _default_out(path, kind):
    p = Path(path)
    if kind == "full":
        return p.parent.parent / "tables" / "full_eval_table.png"
    return p.parent.parent / "figures" / "demo_eval_summary_table.png"


def main():
    fix_console_encoding()
    setup_matplotlib_chinese()

    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?",
                    default="outputs/outputs_v9b/tables/demo_eval_table.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--diag-out", default=None, help="音频塌缩诊断表输出路径")
    ap.add_argument("--diag-title", default=None, help="音频塌缩诊断表标题")
    ap.add_argument("--diag-only", action="store_true",
                    help="仅生成音频塌缩诊断表（跳过主表）")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out) if args.out else None

    if not args.diag_only:
        result = detect_input(src)
        if len(result) == 2:
            rows, kind = result
            meta = None
        else:
            rows, kind, meta = result

        if out is None:
            out = _default_out(src, kind)

        if kind == "full":
            title = args.title or "Full Test Evaluation(n=1000)"
            png, csv = render_full_eval_table(rows, title, out)
        else:
            title = args.title or "Demo Evaluation(n=8)"
            png, csv = render_demo_table(rows, title, out)

        log(f"[plot] 表格图 -> {png}")
        log(f"[plot] CSV -> {csv}")

    diag_out = Path(args.diag_out) if args.diag_out else _default_aud_diag_out(
        src, out)
    try:
        diag_rows = parse_aud_collapse_diag(src)
        if args.diag_title:
            diag_title = args.diag_title
        elif args.diag_only and args.title:
            diag_title = args.title
        else:
            diag_title = "Audio Collapse Diagnostics"
        d_png, d_csv = render_aud_diag_table(diag_rows, diag_title, diag_out)
        log(f"[plot] 音频塌缩诊断表 -> {d_png}")
        log(f"[plot] 音频塌缩诊断 CSV -> {d_csv}")
    except ValueError as e:
        if args.diag_only:
            raise
        log(f"[plot] 跳过音频塌缩诊断表：{e}")


if __name__ == "__main__":
    main()
