from pathlib import Path
import fitz
import json
import re

PDF_DIR = Path("papers")
DATA_DIR = Path("data")
TARGET_WORDS = 700
OVERLAP_WORDS = 100

def clean_text(text):
    replacements = {"\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl","\u00ad":""}
    for old,new in replacements.items(): text = text.replace(old,new)
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def make_chunks(pages, target_words=TARGET_WORDS, overlap_words=OVERLAP_WORDS):
    chunks, buffer, page_start, chunk_id = [], [], None, 0
    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]
        for paragraph in paragraphs:
            if page_start is None: page_start = page["page"]
            words = paragraph.split()
            if buffer and len(buffer)+len(words) > target_words:
                chunks.append({"paper":page["paper"],"chunk_id":chunk_id,"page_start":page_start,"page_end":page["page"],"text":" ".join(buffer)})
                chunk_id += 1
                buffer = buffer[-overlap_words:] if overlap_words else []
                page_start = page["page"]
            buffer.extend(words)
    if buffer:
        chunks.append({"paper":pages[0]["paper"],"chunk_id":chunk_id,"page_start":page_start,"page_end":pages[-1]["page"],"text":" ".join(buffer)})
    return chunks

DATA_DIR.mkdir(exist_ok=True)
pages, chunks = [], []

for pdf in sorted(PDF_DIR.glob("*.pdf")):
    doc = fitz.open(pdf)
    paper_pages = []
    print(f"\n{pdf.name}: {len(doc)} pages")

    for i,page in enumerate(doc):
        text = clean_text(page.get_text("text"))
        record = {"paper":pdf.stem,"page":i+1,"text":text}
        pages.append(record)
        paper_pages.append(record)
        print(f"  Page {i+1}: {len(text)} characters")

    paper_chunks = make_chunks(paper_pages)
    chunks.extend(paper_chunks)
    print(f"  Created {len(paper_chunks)} chunks")

with open(DATA_DIR/"pages.json","w",encoding="utf-8") as f: json.dump(pages,f,indent=2)
with open(DATA_DIR/"chunks.json","w",encoding="utf-8") as f: json.dump(chunks,f,indent=2)

print(f"\nSaved {len(pages)} pages to data/pages.json")
print(f"Saved {len(chunks)} chunks to data/chunks.json")
