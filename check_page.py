import sys
try:
    import pymupdf as fitz
except ImportError:
    import fitz

pdf = sys.argv[1] if len(sys.argv) > 1 else "托福阅读.pdf"
doc = fitz.open(pdf)
page = doc[0]

print("=== get_text('text') ===")
print(page.get_text("text")[:3000])

print("\n=== 前 30 行带坐标 ===")
d = page.get_text("dict")
count = 0
for block in d.get("blocks", []):
    if block.get("type") != 0:
        continue
    for line in block.get("lines", []):
        text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
        if not text:
            continue
        print(line["bbox"], repr(text))
        count += 1
        if count >= 30:
            break
    if count >= 30:
        break

doc.close()
