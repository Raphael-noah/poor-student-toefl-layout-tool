import os
import re
import shutil
import difflib
import fitz
from collections import defaultdict

# ================= 配置区 =================
TARGET_DIR = "output_articles3"
DUPLICATE_DIR = os.path.join(TARGET_DIR, "_duplicates")
SIMILARITY_THRESHOLD = 0.5  # 降维打击：50% 相似度就算重复


def extract_clean_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        doc.close()

        lines = full_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            if "闲鱼" in line or "留学考试资料库" in line: continue
            if re.search(r"20\d{2}-\d{2}-\d{2}", line): continue
            if line.startswith("#"): line = line.lstrip("#").strip()

            clean_lines.append(line)

        # 把所有内容拼成一行，去除非字母字符
        raw_content = " ".join(clean_lines)  # 保留空格，避免单词粘连
        clean_content = re.sub(r'[^a-zA-Z]', '', raw_content).lower()
        return clean_content
    except Exception as e:
        return ""


def extract_title(filename):
    # 提取 2026-06-15_01_Free Will Debate.pdf 里的 "Free Will Debate"
    match = re.search(r"_\d{2}_(.*)\.pdf", filename)
    if match:
        return match.group(1).strip().lower()
    return filename.strip().lower()


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"找不到文件夹: {TARGET_DIR}")
        return

    os.makedirs(DUPLICATE_DIR, exist_ok=True)
    pdf_files = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith(".pdf")])

    print(f"==> 开始分析 {len(pdf_files)} 个 PDF 文件 ...\n")

    # 读取所有文件数据
    fingerprints = {}
    titles_map = defaultdict(list)  # 按标题分组

    for filename in pdf_files:
        file_path = os.path.join(TARGET_DIR, filename)
        text = extract_clean_text(file_path)
        title = extract_title(filename)

        fingerprints[filename] = text
        titles_map[title].append(filename)

    to_keep = set(pdf_files)
    to_move = set()

    # 第一重过滤：按标题去重（标题相同直接干掉后面的）
    for title, files in titles_map.items():
        if len(files) > 1:
            print(f"🎯 发现同名标题: [{title}] 共 {len(files)} 份")
            for f in files[1:]:  # 保留第一份
                print(f"   [剔除同名] {f}")
                to_move.add(f)
                to_keep.discard(f)
    print("-" * 40)

    # 第二重过滤：按正文内容相似度去重
    file_list = list(to_keep)  # 只比对第一轮存活下来的文件
    for i in range(len(file_list)):
        file_a = file_list[i]
        if file_a in to_move: continue

        text_a = fingerprints[file_a]
        if len(text_a) < 50: continue  # 跳过无文本的废文件

        for j in range(i + 1, len(file_list)):
            file_b = file_list[j]
            if file_b in to_move: continue

            text_b = fingerprints[file_b]
            if len(text_b) < 50: continue

            # 快速长度判断
            if abs(len(text_a) - len(text_b)) / max(len(text_a), len(text_b)) > 0.4:
                continue

            # 取前 1000 个字符比对，速度更快
            similarity = difflib.SequenceMatcher(None, text_a[:1000], text_b[:1000]).ratio()

            if similarity >= SIMILARITY_THRESHOLD:
                print(f"⚠️ 发现内容雷同 ({similarity:.1%}):")
                print(f"   [保留] {file_a}")
                print(f"   [剔除] {file_b}")
                to_move.add(file_b)
                to_keep.discard(file_b)

    # 执行移动
    for filename in to_move:
        src = os.path.join(TARGET_DIR, filename)
        dst = os.path.join(DUPLICATE_DIR, filename)
        shutil.move(src, dst)

    print(f"\n✅ 双重去重完成！剔除 {len(to_move)} 个，剩余 {len(to_keep)} 个。")


if __name__ == "__main__":
    main()