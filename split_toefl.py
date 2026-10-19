#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Optional, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image

# ----------------------------- 正则 -----------------------------
DATE_RE = re.compile(r"20\d{2}\s*-\s*\d{2}\s*-\s*\d{2}")
PASSAGE_RE = re.compile(r"^Passage\s*[-–—]", re.IGNORECASE)
KPO_RE = re.compile(r"^KPO\s*\d+\s*[-–—]", re.IGNORECASE)


def safe_filename(s: str, max_len: int = 100) -> str:
    s = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] or "untitled"


def has_english_letters(s: str) -> bool:
    return re.search(r"[A-Za-z]{3,}", s) is not None


def clean_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[#\s:：\-–—]+", "", s)
    s = re.sub(r"^Passage\s*[-–—]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^KPO\s*\d+\s*[-–—]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"来源[:：]\s*KPO.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"原文[（(]Passage[)）].*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"题目[（(]Questions[)）].*$", "", s, flags=re.IGNORECASE)
    s = s.strip(" -–—:#")
    return s


def dedupe_anchors(anchors: List[Dict]) -> List[Dict]:
    result = []
    for a in anchors:
        if any(b["page"] == a["page"] and abs(b["y"] - a["y"]) < 15 for b in result):
            continue
        result.append(a)
    return result


# ----------------------------- 文本层提取 -----------------------------
def text_layer_lines(page, dpi: int) -> List[Dict]:
    scale = dpi / 72.0
    d = page.get_text("dict")
    lines = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue

            # 【新增】过滤水印
            if "闲鱼" in text or "留学考试资料库" in text:
                continue

            x0, y0, x1, y1 = line["bbox"]
            lines.append({
                "text": text,
                "x0": x0 * scale,
                "y0": y0 * scale,
                "x1": x1 * scale,
                "y1": y1 * scale,
            })

    lines.sort(key=lambda d: (d["y0"], d["x0"]))
    return lines


def scan_pages(pdf_path: str, dpi: int) -> List[Dict]:
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = []

    for i, page in enumerate(doc):
        lines = text_layer_lines(page, dpi)
        scale = dpi / 72.0
        w_px = int(round(page.rect.width * scale))
        h_px = int(round(page.rect.height * scale))
        print(f"[文本层提取] {i + 1}/{total} 页", flush=True)

        pages.append({
            "page_no": i + 1,
            "lines": lines,
            "w_px": w_px,
            "h_px": h_px,
        })

    doc.close()
    return pages


# ----------------------------- 锚点识别 -----------------------------
def detect_anchors(pages: List[Dict]) -> Tuple[List[Dict], Optional[Tuple[int, float]]]:
    anchors = []
    current_date = "undated"
    boundary = None

    for pi, page in enumerate(pages):
        if boundary is not None:
            break

        lines = page["lines"]
        n = len(lines)

        for li, line in enumerate(lines):
            t = line["text"].strip()
            if not t:
                continue

            # 1. 判断答案区开始（排除第一页的说明文字）
            if ("参考答案" in t or "Correct Answers" in t):
                if "移至文末" not in t and "本题库" not in t and len(t) < 40:
                    boundary = (pi, line["y0"])
                    break

            # 2. 过滤第一页的 "日期 2026-01-21 ~ 2026-09-19"
            if "日期" in t and "~" in t:
                continue

            # 3. 判断带日期的文章锚点
            m = DATE_RE.search(t)
            if m:
                date_str = m.group(0).replace(" ", "")
                current_date = date_str

                rest = DATE_RE.sub("", t).strip(" -:：#（）()")

                # 如果只是 "(3 套)" 这种分组，跳过
                if re.match(r"^\d+套$", rest) or not rest:
                    continue

                # 如果后面跟着英文标题
                if has_english_letters(rest):
                    title = clean_title(rest)
                    anchors.append({
                        "page": pi,
                        "y": line["y0"],
                        "date": date_str,
                        "title": title,
                    })
                continue

            # 4. 判断无日期的独立标题 (Passage - Expert Systems)
            t_clean = re.sub(r"^[#\s]+", "", t)
            if PASSAGE_RE.match(t_clean) or KPO_RE.match(t_clean):
                title = clean_title(t_clean)
                if not title or not has_english_letters(title):
                    next_text = lines[li + 1]["text"].strip() if li + 1 < n else ""
                    if has_english_letters(next_text):
                        title = clean_title(next_text)

                if title:
                    anchors.append({
                        "page": pi,
                        "y": line["y0"],
                        "date": current_date,
                        "title": title,
                    })
                continue

    return dedupe_anchors(anchors), boundary


# ----------------------------- 构建范围 -----------------------------
def build_units(pages: List[Dict], anchors: List[Dict], boundary: Optional[Tuple[int, float]]) -> List[Dict]:
    units = []
    total_pages = len(pages)

    for i, a in enumerate(anchors):
        start_page = a["page"]
        start_y = max(0.0, a["y"] - 5.0)

        if i + 1 < len(anchors):
            end_page = anchors[i + 1]["page"]
            end_y = max(0.0, anchors[i + 1]["y"] - 5.0)
        elif boundary is not None:
            end_page, end_y = boundary
            end_y = max(0.0, end_y - 5.0)
        else:
            end_page = total_pages - 1
            end_y = pages[-1]["h_px"]

        units.append({
            "date": a["date"],
            "title": a["title"],
            "start_page": start_page,
            "start_y": start_y,
            "end_page": end_page,
            "end_y": end_y,
        })

    return units


# ----------------------------- 裁剪与输出 -----------------------------
def render_page(doc, page_idx: int, dpi: int) -> Image.Image:
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# 【修改】加入动态寻找文字边界，去除大量空白
def crop_range(doc, start_page: int, start_y: float, end_page: int, end_y: float, dpi: int, pages: List[Dict]) -> List[
    Image.Image]:
    imgs = []

    for pi in range(start_page, end_page + 1):
        img = render_page(doc, pi, dpi)
        W, H = img.size

        # 原始的切分限制
        y0_limit = int(max(0, start_y)) if pi == start_page else 0
        y1_limit = int(min(H, end_y)) if pi == end_page else H

        # 找到这一页在当前范围内，实际有文字的上下边界
        valid_lines = []
        for ln in pages[pi]["lines"]:
            if ln["y1"] > y0_limit and ln["y0"] < y1_limit:
                valid_lines.append(ln)

        if valid_lines:
            # 找到最高点的顶部和最低点的底部
            min_y0 = min(ln["y0"] for ln in valid_lines)
            max_y1 = max(ln["y1"] for ln in valid_lines)

            # 上下各留 15 像素的呼吸空间
            margin = 15
            y0 = max(0, int(min_y0 - margin))
            y1 = min(H, int(max_y1 + margin))
        else:
            # 如果这一页在这篇文章范围内没有任何文字，就跳过
            continue

        if y1 - y0 < 8:
            continue

        imgs.append(img.crop((0, y0, W, y1)))

    return imgs


def vconcat(imgs: List[Image.Image]) -> Optional[Image.Image]:
    if not imgs:
        return None
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height
    return out


def save_as_a4_pdf(img: Image.Image, path: str, dpi: int):
    a4_w = img.width
    a4_h = int(a4_w * 297 / 210)
    pages = []
    y = 0
    while y < img.height:
        pages.append(img.crop((0, y, img.width, min(y + a4_h, img.height))))
        y += a4_h

    if not pages:
        return
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:] if len(pages) > 1 else [])


# ----------------------------- 主流程 -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="输入 PDF 路径")
    parser.add_argument("-o", "--outdir", default="output_articles", help="输出目录")
    parser.add_argument("--dpi", type=int, default=200, help="渲染 DPI")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"找不到 PDF：{args.pdf}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    answers_dir = os.path.join(args.outdir, "answers")
    os.makedirs(answers_dir, exist_ok=True)

    print(f"==> 扫描页面 (dpi={args.dpi}) ...", flush=True)
    pages = scan_pages(args.pdf, args.dpi)

    print("\n==> 识别文章锚点 ...", flush=True)
    anchors, boundary = detect_anchors(pages)

    print(f"\n共识别到 {len(anchors)} 个文章锚点：")
    for a in anchors[:10]:  # 只打印前10个看看
        print(f"  page {a['page'] + 1:>3}  y={int(a['y']):>5}  {a['date']:<12}  {a['title']}")
    if len(anchors) > 10:
        print(f"  ... 还有 {len(anchors) - 10} 个")

    if boundary is not None:
        print(f"  参考答案起始位置：page {boundary[0] + 1}  y={int(boundary[1])}")
    else:
        print("  未识别到参考答案起始位置。")

    if not anchors:
        print("没有识别到文章。", file=sys.stderr)
        return

    units = build_units(pages, anchors, boundary)
    doc = fitz.open(args.pdf)

    used = {}
    index = []

    print("\n==> 开始裁剪并输出文章 ...", flush=True)
    for u in units:
        # 【修改】调用时多传一个 pages 参数
        imgs = crop_range(doc, u["start_page"], u["start_y"], u["end_page"], u["end_y"], args.dpi, pages)
        if not imgs:
            continue
        long_img = vconcat(imgs)
        if long_img is None:
            continue

        key = (u["date"], safe_filename(u["title"]))
        used[key] = used.get(key, 0) + 1
        seq = used[key]

        base = f"{u['date']}_{seq:02d}_{safe_filename(u['title'])}"
        out_pdf = os.path.join(args.outdir, base + ".pdf")
        save_as_a4_pdf(long_img, out_pdf, args.dpi)

        index.append({
            "date": u["date"],
            "title": u["title"],
            "file": out_pdf,
            "start_page": u["start_page"] + 1,
            "end_page": u["end_page"] + 1,
        })
        print(f"[保存] {out_pdf}")

    # 处理答案区
    if boundary is not None:
        print("\n==> 整理参考答案 ...", flush=True)
        last_page = len(pages) - 1
        last_y = pages[-1]["h_px"]

        # 【修改】调用时多传一个 pages 参数
        full_imgs = crop_range(doc, boundary[0], max(0.0, boundary[1] - 5), last_page, last_y, args.dpi, pages)
        full_img = vconcat(full_imgs)
        if full_img is not None:
            full_pdf = os.path.join(answers_dir, "answers_full.pdf")
            save_as_a4_pdf(full_img, full_pdf, args.dpi)
            print(f"[保存] {full_pdf}")

    doc.close()

    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n完成，共输出 {len(index)} 篇文章。")


if __name__ == "__main__":
    main()