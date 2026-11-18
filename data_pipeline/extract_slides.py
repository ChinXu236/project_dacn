import sys, json, re
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd

def parse_pages(s):
    s = re.sub(r"\(.*?\)", "", str(s))  # bỏ phần "(Sách: ...)"
    pages = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages += list(range(int(a), int(b) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return pages

def main(src, out_json, concepts_csv):
    src = Path(src)
    index, counts = [], {}
    for f in sorted(src.glob("*.pdf")):
        doc = fitz.open(f)
        counts[f.name] = len(doc)
        for i, p in enumerate(doc, 1):
            index.append({"file": f.name, "page": i, "text": p.get_text().strip()})
    Path(out_json).write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Số trang:", counts)

    for _, r in pd.read_csv(concepts_csv).iterrows():
        for fcol, pcol in (("slide_file", "slide_pages"), ("textbook_file", "textbook_pages")):
            f, pages = r[fcol], parse_pages(r[pcol])
            if f not in counts:
                print(f"{r.concept_id}: thiếu file {f}")
            elif pages and max(pages) > counts[f]:
                print(f"{r.concept_id}: {f} trang {max(pages)} > {counts[f]}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])