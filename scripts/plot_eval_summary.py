"""将评估结果转为表格图 + CSV。

支持以下输入（自动识别）：
  1. demo_eval_table.txt  — 【汇总】段（10 列，含 img/aud mask MSE）
  2. eval_*.log           — evaluate.py 全量评估日志；多 family 时写入 tables/<family_slug>/
  3. 上述日志若含 [音频塌缩诊断] 段，自动生成 aud_diag 表格（PNG+CSV）
  4. 上述日志若含 [Cross-Key归因] 段，按 family 生成独立 PNG+CSV

用法：
    python scripts/plot_eval_summary.py outputs/outputs_v11c/tables/demo_eval_table.txt
    python scripts/plot_eval_summary.py outputs/outputs_v11c/logs/eval_v11c_cross_key_sweep_sev04.log
    python scripts/plot_eval_summary.py eval_v11c_full.log --title "v11c full eval"
    python scripts/plot_eval_summary.py eval_v11c_fixed_mask.log --diag-only
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
    "clean_img_corrupt_aud", "corrupt_img_clean_aud",
    "clean_img_only", "clean_aud_only", "clean_both",
)
_FULL_ROW_RE = re.compile(
    r"(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"(\d+\.\d+)%\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"(\S+)",
)
# v10b+ 全量评估日志含 audSSIM / maskMSE 两列（在 audMSE 与像素方差之间）
_FULL_ROW_V10B_RE = re.compile(
    r"(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"(\d+\.\d+)%\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"   # imgMSE PSNR imgSSIM
    r"([\d.]+)\s+([\d.]+)\s+"              # audMSE audSSIM
    r"([\d.]+|nan)\s+"                     # maskMSE
    r"([\d.]+)\s+([\d.]+)\s+"              # pix_var sampleL2
    r"(\S+)",
)
# v10e+ 全量评估日志同时含 imgMaskMSE / audMaskMSE
_FULL_ROW_V10E_RE = re.compile(
    r"(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"(\d+\.\d+)%\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"   # imgMSE PSNR imgSSIM
    r"([\d.]+|nan)\s+"                     # imgMaskMSE
    r"([\d.]+)\s+([\d.]+|nan)\s+"          # audMSE audSSIM
    r"([\d.]+|nan)\s+"                     # audMaskMSE
    r"([\d.]+)\s+([\d.]+)\s+"              # pix_var sampleL2
    r"(\S+)",
)
_AUD_DIAG_ROW_RE = re.compile(
    r"^(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)%\s*$",
)
_ATTR_ROW_RE = re.compile(
    r"^(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s*$",
)
_ATTR_ROW_RE_LEGACY = re.compile(
    r"^(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s*$",
)
_CROSS_KEY_ROW_RE = re.compile(
    r"^(corrupt_img_only|corrupt_aud_only|corrupt_both|"
    r"clean_img_corrupt_aud|corrupt_img_clean_aud|"
    r"clean_img_only|clean_aud_only|clean_both)\s+"
    r"(img->aud|aud->img)\s+"
    r"(-?[\d.]+|N/A|nan)\s+(-?[\d.]+|N/A|nan)\s+"
    r"(-?[\d.]+|N/A|nan)\s+(-?[\d.]+|N/A|nan)\s+"
    r"(-?[\d.]+|N/A|nan)\s+(-?[\d.]+|N/A|nan)"
    r"(?:\s+(-?[\d.]+|N/A|nan)\s+(-?[\d.]+|N/A|nan))?\s*$",
)
_FAMILY_HDR_RE = re.compile(
    r"\[评估 family (\d+)/(\d+)\] img=(\S+)\s+aud=(\S+)",
)


def _maybe_float(text):
    return None if text in ("nan", "N/A") else float(text)


def _order_eval_rows(by_mode):
    return [by_mode[m] for m in _EVAL_MODES if m in by_mode]


def _parse_full_rows_from_text(text):
    """从日志片段解析当前 cue 模式主评估行。"""
    by_mode = {}
    for line in text.splitlines():
        m = _FULL_ROW_V10E_RE.search(line)
        if m:
            img_mask_mse = _maybe_float(m.group(6))
            aud_mask_mse = _maybe_float(m.group(9))
            by_mode[m.group(1)] = {
                "mode": m.group(1),
                "acc": float(m.group(2)) / 100.0,
                "img_mse": float(m.group(3)),
                "psnr": float(m.group(4)),
                "img_ssim": float(m.group(5)),
                "img_mask_mse": img_mask_mse,
                "aud_mse": float(m.group(7)),
                "aud_ssim": _maybe_float(m.group(8)),
                "aud_mask_mse": aud_mask_mse,
                "mask_mse": aud_mask_mse,
                "pix_var": float(m.group(10)),
                "pair_l2": float(m.group(11)),
                "tgt": m.group(12),
            }
            continue
        m = _FULL_ROW_V10B_RE.search(line)
        if m:
            mask_mse = m.group(8)
            by_mode[m.group(1)] = {
                "mode": m.group(1),
                "acc": float(m.group(2)) / 100.0,
                "img_mse": float(m.group(3)),
                "psnr": float(m.group(4)),
                "img_ssim": float(m.group(5)),
                "img_mask_mse": None,
                "aud_mse": float(m.group(6)),
                "aud_ssim": float(m.group(7)),
                "aud_mask_mse": float(mask_mse) if mask_mse != "nan" else None,
                "mask_mse": float(mask_mse) if mask_mse != "nan" else None,
                "pix_var": float(m.group(9)),
                "pair_l2": float(m.group(10)),
                "tgt": m.group(11),
            }
            continue
        m = _FULL_ROW_RE.search(line)
        if m:
            by_mode[m.group(1)] = {
                "mode": m.group(1),
                "acc": float(m.group(2)) / 100.0,
                "img_mse": float(m.group(3)),
                "psnr": float(m.group(4)),
                "img_ssim": float(m.group(5)),
                "img_mask_mse": None,
                "aud_mse": float(m.group(6)),
                "aud_ssim": None,
                "aud_mask_mse": None,
                "mask_mse": None,
                "pix_var": float(m.group(7)),
                "pair_l2": float(m.group(8)),
                "tgt": m.group(9),
            }
    return _order_eval_rows(by_mode)


def _parse_aud_diag_from_text(text):
    """从日志片段解析 [音频塌缩诊断] 段。"""
    rows = []
    in_block = False
    for line in text.splitlines():
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
    return rows


def _parse_attribution_from_text(text):
    """从日志片段解析 [归因] 段。"""
    rows = []
    in_block = False
    past_hdr = False
    for line in text.splitlines():
        if "[归因]" in line:
            in_block = True
            past_hdr = False
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("=") or stripped.startswith("-"):
            continue
        if "imgCmask" in stripped or "imgCoarse" in stripped or "cue模式" in stripped:
            past_hdr = True
            continue
        m = _ATTR_ROW_RE.match(stripped)
        if m and past_hdr:
            rows.append({
                "mode": m.group(1),
                "img_coarse_masked_mse": _maybe_float(m.group(2)),
                "img_final_masked_mse": _maybe_float(m.group(3)),
                "img_coarse_visible_mse": _maybe_float(m.group(4)),
                "img_final_visible_mse": _maybe_float(m.group(5)),
                "aud_coarse_masked_mse": _maybe_float(m.group(6)),
                "aud_final_masked_mse": _maybe_float(m.group(7)),
                "aud_coarse_visible_mse": _maybe_float(m.group(8)),
                "aud_final_visible_mse": _maybe_float(m.group(9)),
            })
            continue
        m = _ATTR_ROW_RE_LEGACY.match(stripped)
        if m and past_hdr:
            rows.append({
                "mode": m.group(1),
                "img_coarse_masked_mse": _maybe_float(m.group(2)),
                "img_final_masked_mse": _maybe_float(m.group(3)),
                "img_coarse_visible_mse": None,
                "img_final_visible_mse": None,
                "aud_coarse_masked_mse": _maybe_float(m.group(4)),
                "aud_final_masked_mse": _maybe_float(m.group(5)),
                "aud_coarse_visible_mse": None,
                "aud_final_visible_mse": None,
            })
    return rows


def _parse_cross_key_attribution_from_text(text):
    """Parse paired normal/zero/wrong/same-class Cross-Key attribution."""
    rows = []
    in_block = False
    past_hdr = False
    for line in text.splitlines():
        if "[Cross-Key归因]" in line:
            in_block = True
            past_hdr = False
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("=") or stripped.startswith("-"):
            continue
        if "Cgain" in stripped and ("Fdamage" in stripped or "Fsame" in stripped):
            past_hdr = True
            continue
        match = _CROSS_KEY_ROW_RE.match(stripped)
        if match and past_hdr:
            rows.append({
                "mode": match.group(1),
                "direction": match.group(2),
                "gate": _maybe_float(match.group(3)),
                "residual_ratio": _maybe_float(match.group(4)),
                "coarse_gain": _maybe_float(match.group(5)),
                "final_gain": _maybe_float(match.group(6)),
                "coarse_damage": _maybe_float(match.group(7)),
                "final_damage": _maybe_float(match.group(8)),
                "coarse_same_damage": (
                    _maybe_float(match.group(9)) if match.group(9) else None),
                "final_same_damage": (
                    _maybe_float(match.group(10)) if match.group(10) else None),
            })
    return rows


def _split_family_blocks(text):
    """按 [评估 family i/n] 切分多 family 日志。"""
    parts = re.split(r"(?=\[评估 family \d+/\d+\])", text)
    blocks = []
    for part in parts:
        m = _FAMILY_HDR_RE.search(part)
        if not m:
            continue
        blocks.append({
            "family_idx": int(m.group(1)),
            "family_total": int(m.group(2)),
            "img_mode": m.group(3),
            "aud_mode": m.group(4),
            "text": part,
        })
    return blocks


def _family_slug(family):
    if family.get("family_idx") is None:
        return ""
    return (
        f"family{family['family_idx']:02d}_"
        f"{family['img_mode']}_{family['aud_mode']}"
    )


def _family_dir(base_out, family):
    """多 family 评估表写入 tables/<family_slug>/ 子目录。"""
    p = Path(base_out)
    slug = _family_slug(family)
    if not slug:
        return p.parent
    return p.parent / slug


def _family_artifact_path(base_out, family, stem):
    return _family_dir(base_out, family) / f"{stem}.png"


def _family_title(base_title, family):
    slug = _family_slug(family)
    if not slug:
        return base_title
    return (
        f"{base_title} — img={family['img_mode']} / aud={family['aud_mode']} "
        f"({family['family_idx']}/{family['family_total']})"
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
        row = {
            "mode": cols[0],
            "acc": float(cols[1].rstrip("%")) / 100.0,
            "img_ssim": float(cols[2]),
            "img_mse": float(cols[3]),
            "aud_ssim": float(cols[4]),
            "aud_mse": float(cols[5]),
        }
        if len(cols) >= 10:
            row["img_masked_mse"] = _maybe_float(cols[6])
            row["aud_masked_mse"] = _maybe_float(cols[7])
            row["img_tgt"] = cols[8]
            row["aud_tgt"] = cols[9]
        else:
            row["img_masked_mse"] = None
            row["aud_masked_mse"] = None
            row["img_tgt"] = cols[6] if len(cols) > 6 else ""
            row["aud_tgt"] = cols[7] if len(cols) > 7 else ""
        rows.append(row)
    if not rows:
        raise ValueError(f"未在 {path} 中找到【汇总】数据。")
    return rows, "demo"


def parse_eval_full_log(path):
    """解析 evaluate.py 全量评估日志；多 family 时返回分 family 列表。"""
    text = Path(path).read_text(encoding="utf-8")
    severity = None
    n_test = None
    for line in text.splitlines():
        m = re.search(r"corrupt severity=([\d.]+)", line)
        if m:
            severity = float(m.group(1))
        m = re.search(r"\[dataset\] test.*n=(\d+)", line)
        if m:
            n_test = int(m.group(1))

    blocks = _split_family_blocks(text)
    meta = {"severity": severity, "n_test": n_test}
    if blocks:
        families = []
        for block in blocks:
            rows = _parse_full_rows_from_text(block["text"])
            if not rows:
                continue
            families.append({
                **block,
                "rows": rows,
                "aud_diag": _parse_aud_diag_from_text(block["text"]),
                "attribution": _parse_attribution_from_text(block["text"]),
                "cross_key": _parse_cross_key_attribution_from_text(block["text"]),
            })
        if not families:
            raise ValueError(f"未在 {path} 中找到 evaluate.py 结果行。")
        return families, "full_families", meta

    rows = _parse_full_rows_from_text(text)
    if not rows:
        raise ValueError(f"未在 {path} 中找到 evaluate.py 结果行。")
    family = {
        "family_idx": None,
        "family_total": None,
        "img_mode": None,
        "aud_mode": None,
        "text": text,
        "rows": rows,
        "aud_diag": _parse_aud_diag_from_text(text),
        "attribution": _parse_attribution_from_text(text),
        "cross_key": _parse_cross_key_attribution_from_text(text),
    }
    return [family], "full_families", meta


def parse_aud_collapse_diag(path):
    """解析 evaluate.py / demo 日志末尾的 [音频塌缩诊断] 段。"""
    rows = _parse_aud_diag_from_text(Path(path).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"未在 {path} 中找到 [音频塌缩诊断] 数据行。")
    return rows


def parse_attribution_table(path):
    """解析 evaluate.py [归因] coarse/final masked/visible MSE 段。"""
    rows = _parse_attribution_from_text(Path(path).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"未在 {path} 中找到 [归因] 数据行。")
    return rows


def detect_input(path):
    text = Path(path).read_text(encoding="utf-8")
    if "【汇总】" in text:
        return parse_eval_summary(path)
    if any(m in text for m in _EVAL_MODES) and "imgMSE" in text:
        return parse_eval_full_log(path)
    try:
        return parse_eval_full_log(path)
    except ValueError:
        return parse_eval_summary(path)


def _render_full_family_tables(families, base_out, base_title, meta=None):
    """为每个 family 生成主表 + 归因 + 音频塌缩诊断表。"""
    if base_out is None:
        base_out = Path("full_eval_table.png")
    base_out = Path(base_out)
    sev = meta.get("severity") if meta else None
    n_test = meta.get("n_test") if meta else None
    if base_title is None:
        parts = ["Full Test Evaluation"]
        if n_test is not None:
            parts.append(f"(n={n_test})")
        if sev is not None:
            parts.append(f"sev={sev}")
        base_title = " ".join(parts)

    outputs = []
    for family in families:
        main_out = _family_artifact_path(base_out, family, "full_eval")
        title = _family_title(base_title, family)
        png, csv = render_full_eval_table(family["rows"], title, main_out)
        log(f"[plot] 主评估表 -> {png}")
        log(f"[plot] CSV -> {csv}")
        outputs.append((png, csv))

        if family.get("aud_diag"):
            diag_out = _family_artifact_path(base_out, family, "aud_diag")
            d_png, d_csv = render_aud_diag_table(
                family["aud_diag"], "Audio Collapse Diagnostics", diag_out)
            log(f"[plot] 音频塌缩诊断表 -> {d_png}")
            log(f"[plot] 音频塌缩诊断 CSV -> {d_csv}")

        if family.get("attribution"):
            attr_out = _family_artifact_path(base_out, family, "attribution")
            a_png, a_csv = render_attribution_table(
                family["attribution"], "Coarse vs Final Masked/Visible MSE",
                attr_out)
            log(f"[plot] 归因表 -> {a_png}")
            log(f"[plot] 归因 CSV -> {a_csv}")

        if family.get("cross_key"):
            cross_out = _family_artifact_path(
                base_out, family, "cross_key_attribution")
            c_png, c_csv = render_cross_key_attribution_table(
                family["cross_key"], "Cross-Key Paired Attribution", cross_out)
            log(f"[plot] Cross-Key attribution -> {c_png}")
            log(f"[plot] Cross-Key attribution CSV -> {c_csv}")

    return outputs


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
    has_img_mask = any(r.get("img_masked_mse") is not None for r in rows)
    has_aud_mask = any(r.get("aud_masked_mse") is not None for r in rows)
    if has_img_mask or has_aud_mask:
        headers = ["", "acc", "img SSIM", "img MSE", "aud SSIM", "aud MSE",
                   "img mask MSE", "aud mask MSE", "img target", "aud target"]
        cell = [[
            r["mode"], f"{r['acc']:.3f}",
            f"{r['img_ssim']:.3f}", f"{r['img_mse']:.4f}",
            f"{r['aud_ssim']:.3f}", f"{r['aud_mse']:.4f}",
            f"{r['img_masked_mse']:.4f}"
            if r.get("img_masked_mse") is not None else "nan",
            f"{r['aud_masked_mse']:.4f}"
            if r.get("aud_masked_mse") is not None else "nan",
            r["img_tgt"], r["aud_tgt"],
        ] for r in rows]
        col_w = [0.11, 0.07, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.13, 0.13]
    else:
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
    has_img_mask = any(r.get("img_mask_mse") is not None for r in rows)
    has_aud_ssim = any(r.get("aud_ssim") is not None for r in rows)
    if has_img_mask:
        headers = ["", "acc", "img MSE", "PSNR", "img SSIM", "img mask MSE",
                   "aud MSE", "aud SSIM", "aud mask MSE",
                   "target(image/audio)"]
        cell = [[
            r["mode"], f"{r['acc']:.3f}",
            f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['img_ssim']:.3f}",
            f"{r['img_mask_mse']:.4f}" if r.get("img_mask_mse") is not None else "nan",
            f"{r['aud_mse']:.4f}",
            f"{r['aud_ssim']:.3f}" if r.get("aud_ssim") is not None else "nan",
            f"{r['aud_mask_mse']:.4f}" if r.get("aud_mask_mse") is not None else "nan",
            _expand_target(r["tgt"]),
        ] for r in rows]
        col_w = [0.11, 0.06, 0.08, 0.06, 0.08, 0.10, 0.08, 0.08, 0.10, 0.15]
    elif has_aud_ssim:
        headers = ["", "acc", "img MSE", "PSNR", "img SSIM", "aud MSE",
                   "aud SSIM", "mask MSE", "target(image/audio)"]
        cell = [[
            r["mode"], f"{r['acc']:.3f}",
            f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['img_ssim']:.3f}",
            f"{r['aud_mse']:.4f}",
            f"{r['aud_ssim']:.3f}" if r.get("aud_ssim") is not None else "",
            f"{r['mask_mse']:.4f}" if r.get("mask_mse") is not None else "nan",
            _expand_target(r["tgt"]),
        ] for r in rows]
        col_w = [0.12, 0.07, 0.09, 0.07, 0.09, 0.09, 0.09, 0.09, 0.16]
    else:
        headers = ["", "acc", "img MSE", "PSNR", "img SSIM", "aud MSE",
                   "target(image/audio)"]
        cell = [[
            r["mode"], f"{r['acc']:.3f}",
            f"{r['img_mse']:.4f}", f"{r['psnr']:.2f}", f"{r['img_ssim']:.3f}",
            f"{r['aud_mse']:.4f}", _expand_target(r["tgt"]),
        ] for r in rows]
        col_w = [0.14, 0.08, 0.10, 0.08, 0.10, 0.10, 0.18]
    return _render_table(headers, cell, col_w, title, path, font_size=8.5)


def render_attribution_table(rows, title, path):
    headers = ["", "img C mask", "img F mask", "img C vis", "img F vis",
               "aud C mask", "aud F mask", "aud C vis", "aud F vis"]
    cell = [[
        r["mode"],
        f"{r['img_coarse_masked_mse']:.4f}"
        if r.get("img_coarse_masked_mse") is not None else "nan",
        f"{r['img_final_masked_mse']:.4f}"
        if r.get("img_final_masked_mse") is not None else "nan",
        f"{r['img_coarse_visible_mse']:.4f}"
        if r.get("img_coarse_visible_mse") is not None else "nan",
        f"{r['img_final_visible_mse']:.4f}"
        if r.get("img_final_visible_mse") is not None else "nan",
        f"{r['aud_coarse_masked_mse']:.4f}"
        if r.get("aud_coarse_masked_mse") is not None else "nan",
        f"{r['aud_final_masked_mse']:.4f}"
        if r.get("aud_final_masked_mse") is not None else "nan",
        f"{r['aud_coarse_visible_mse']:.4f}"
        if r.get("aud_coarse_visible_mse") is not None else "nan",
        f"{r['aud_final_visible_mse']:.4f}"
        if r.get("aud_final_visible_mse") is not None else "nan",
    ] for r in rows]
    col_w = [0.14] + [0.095] * 8
    return _render_table(headers, cell, col_w, title, path, font_size=8.5)


def render_cross_key_attribution_table(rows, title, path):
    headers = ["cue mode", "direction", "gate", "res/V", "coarse gain",
               "final gain", "coarse wrong", "final wrong",
               "coarse same", "final same"]

    def fmt(value):
        return "N/A" if value is None else f"{value:.5f}"

    cell = [[
        r["mode"], r["direction"], fmt(r.get("gate")),
        fmt(r.get("residual_ratio")), fmt(r.get("coarse_gain")),
        fmt(r.get("final_gain")), fmt(r.get("coarse_damage")),
        fmt(r.get("final_damage")), fmt(r.get("coarse_same_damage")),
        fmt(r.get("final_same_damage")),
    ] for r in rows]
    col_w = [0.16, 0.09] + [0.085] * 8
    return _render_table(headers, cell, col_w, title, path, font_size=7.8)


def _parse_n_samples(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"样本数:\s*(\d+)", line.strip())
        if m:
            return int(m.group(1))
    return None


def _default_out(path, kind):
    p = Path(path)
    if kind in ("full", "full_families"):
        return p.parent.parent / "tables" / "full_eval_table.png"
    return p.parent.parent / "figures" / "demo_eval_summary_table.png"


def main():
    fix_console_encoding()
    setup_matplotlib_chinese()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input", nargs="?",
        default="outputs/outputs_v11c/tables/demo_eval_table.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--diag-out", default=None, help="音频塌缩诊断表输出路径")
    ap.add_argument("--diag-title", default=None, help="音频塌缩诊断表标题")
    ap.add_argument("--diag-only", action="store_true",
                    help="仅生成音频塌缩诊断表（跳过主表）")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out) if args.out else None

    result = detect_input(src)
    if len(result) == 2:
        rows, kind = result
        meta = None
        families = None
    else:
        families, kind, meta = result
        rows = None

    if out is None:
        out = _default_out(src, "full" if kind == "full_families" else kind)

    if kind == "full_families":
        if args.diag_only:
            for family in families:
                if not family.get("aud_diag"):
                    continue
                diag_out = (
                    Path(args.diag_out)
                    if args.diag_out
                    else _family_artifact_path(out, family, "aud_diag")
                )
                diag_title = args.diag_title or _family_title(
                    "Audio Collapse Diagnostics", family)
                d_png, d_csv = render_aud_diag_table(
                    family["aud_diag"], diag_title, diag_out)
                log(f"[plot] 音频塌缩诊断表 -> {d_png}")
                log(f"[plot] 音频塌缩诊断 CSV -> {d_csv}")
        else:
            _render_full_family_tables(families, out, args.title, meta)
    elif not args.diag_only:
        title = args.title or "Demo Evaluation(n=8)"
        png, csv = render_demo_table(rows, title, out)
        log(f"[plot] 表格图 -> {png}")
        log(f"[plot] CSV -> {csv}")

        diag_out = Path(args.diag_out) if args.diag_out else _default_aud_diag_out(
            src, out)
        try:
            diag_rows = parse_aud_collapse_diag(src)
            diag_title = args.diag_title or "Audio Collapse Diagnostics"
            d_png, d_csv = render_aud_diag_table(diag_rows, diag_title, diag_out)
            log(f"[plot] 音频塌缩诊断表 -> {d_png}")
            log(f"[plot] 音频塌缩诊断 CSV -> {d_csv}")
        except ValueError as e:
            log(f"[plot] 跳过音频塌缩诊断表：{e}")

        attr_out = out.with_name(f"{out.stem}_attribution{out.suffix}")
        try:
            attr_rows = parse_attribution_table(src)
            a_png, a_csv = render_attribution_table(
                attr_rows, "Coarse vs Final Masked/Visible MSE", attr_out)
            log(f"[plot] 归因表 -> {a_png}")
            log(f"[plot] 归因 CSV -> {a_csv}")
        except ValueError as e:
            log(f"[plot] 跳过归因表：{e}")
    else:
        diag_out = Path(args.diag_out) if args.diag_out else _default_aud_diag_out(
            src, out)
        diag_rows = parse_aud_collapse_diag(src)
        diag_title = args.diag_title or args.title or "Audio Collapse Diagnostics"
        d_png, d_csv = render_aud_diag_table(diag_rows, diag_title, diag_out)
        log(f"[plot] 音频塌缩诊断表 -> {d_png}")
        log(f"[plot] 音频塌缩诊断 CSV -> {d_csv}")


if __name__ == "__main__":
    main()
