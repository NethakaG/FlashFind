import os, json, hashlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

nltk.download('punkt', quiet=True)

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def _txt_to_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    return _txt_to_text(path)

def file_fingerprint(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(str(p.stat().st_mtime).encode())
    return h.hexdigest()[:16]

def build_corpus() -> Tuple[list, list]:
    files = [p for p in DATA_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in [".pdf",".txt",".md"]]
    texts, meta = [], []
    for f in files:
        try:
            raw = read_file(f)
            for sent in sent_tokenize(raw):
                sent = sent.strip()
                if len(sent) > 20:
                    texts.append(sent)
                    meta.append({"source": f.name})
        except Exception:
            continue
    return texts, meta

def load_or_build_index(model):
    files = [p for p in DATA_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in [".pdf",".txt",".md"]]
    fp = file_fingerprint(files)
    idx_path = INDEX_DIR / f"index_{fp}.npz"
    txt_path = INDEX_DIR / f"texts_{fp}.json"
    meta_path = INDEX_DIR / f"meta_{fp}.json"

    if idx_path.exists() and txt_path.exists() and meta_path.exists():
        texts = json.loads(txt_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeds = np.load(idx_path)["embeds"]
        return texts, embeds, meta

    texts, meta = build_corpus()
    embeds = model.encode(texts, normalize_embeddings=True)
    np.savez_compressed(idx_path, embeds=embeds)
    txt_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return texts, embeds, meta
