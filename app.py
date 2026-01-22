# app.py
# pip install flask chromadb numpy sentence-transformers torch python-docx pypdf requests
#
# Optional (for LLM generation + OpenAI embeddings):
#   export OPENAI_API_KEY="..."
#   export GEN_PROVIDER="openai"                 # or leave unset for extractive fallback
#   export GEN_MODEL="gpt-4.1-mini"              # better generation model
#   export EMBED_PROVIDER="local"                # "local" (default) or "openai"
#   export EMBED_MODEL="all-MiniLM-L6-v2"        # cheap/local embeddings (default)
#   export OPENAI_EMBED_MODEL="text-embedding-3-small"  # cheap OpenAI embeddings
#
# Codespaces: python app.py, then open forwarded port 5000

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, redirect, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# -------------------- App --------------------
app = Flask(__name__)

# -------------------- Config --------------------
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
INDEX_DIR = os.path.join(BASE_DIR, "index_data")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx"}

# Gemini (Generation)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Chunking (document -> chunks)
CHUNK_MAX_WORDS = int(os.environ.get("CHUNK_MAX_WORDS", "260"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "50"))

# Cleaning (line/paragraph merge)
CLEAN_MIN_CHARS = int(os.environ.get("CLEAN_MIN_CHARS", "120"))
CLEAN_MIN_WORDS = int(os.environ.get("CLEAN_MIN_WORDS", "6"))

# Retrieval
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "20"))          # UI search results
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "18"))              # retrieved chunks
RAG_PER_FILE_CAP = int(os.environ.get("RAG_PER_FILE_CAP", "3")) # cap evidence per file
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "18000"))

# Embeddings provider
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local").strip().lower()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2").strip()
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

# Generation provider (RAG "Generation" step)
GEN_PROVIDER = os.environ.get("GEN_PROVIDER", "").strip().lower()  # "openai" or empty for fallback
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4.1-mini").strip()    # default "better" than nano
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

FILES_DB_PATH = os.path.join(INDEX_DIR, "files.json")  # tracks files + tags + chunk counts

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)

# -------------------- Vector DB (Chroma) --------------------
try:
    import chromadb
except Exception as e:
    raise RuntimeError("chromadb is required. Install: pip install chromadb") from e

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_COLLECTION_NAME = "chunks"


_collection = None

def get_collection():
    global _collection
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def reset_collection():
    global _collection
    try:
        _chroma_client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    return get_collection()



# -------------------- Embeddings --------------------
_local_model = None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        denom = np.linalg.norm(mat) or 1.0
        return mat / denom
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def _embed_local(texts: List[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(EMBED_MODEL)
    vecs = _local_model.encode(texts, normalize_embeddings=True)  # already normalized
    return np.asarray(vecs, dtype=np.float32)


def _embed_openai(texts: List[str]) -> np.ndarray:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set, but EMBED_PROVIDER=openai")

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_EMBED_MODEL, "input": texts, "encoding_format": "float"}

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI embeddings error ({r.status_code}): {r.text[:4000]}")
    data = r.json()
    vecs = [row["embedding"] for row in data["data"]]
    mat = np.asarray(vecs, dtype=np.float32)
    return _l2_normalize(mat)  # normalize to make cosine/dot consistent


def embed_texts(texts: List[str]) -> np.ndarray:
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if EMBED_PROVIDER == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


# -------------------- Text extraction --------------------
def _clean_paragraphs(
    paragraphs: List[str],
    min_chars: int = CLEAN_MIN_CHARS,
    min_words: int = CLEAN_MIN_WORDS,
) -> List[str]:
    """
    Merge short lines into more paragraph-like chunks; drop very short leftovers.
    Helps with title pages that are split line-by-line.
    """
    merged: List[str] = []
    buffer = ""

    for p in paragraphs:
        p = re.sub(r"\s+", " ", (p or "")).strip()
        if not p:
            continue

        buffer = (buffer + " " + p).strip() if buffer else p

        if len(buffer) >= min_chars or buffer.endswith((".", "?", "!")):
            if len(buffer.split()) >= min_words:
                merged.append(buffer)
            buffer = ""

    if buffer and len(buffer.split()) >= min_words:
        merged.append(buffer)

    return merged


def docx_to_paragraphs(path: str) -> List[str]:
    from docx import Document

    doc = Document(path)
    return _clean_paragraphs([para.text for para in doc.paragraphs])


def pdf_to_paragraphs(path: str) -> List[str]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    full_text = "\n".join(chunks)

    # Convert single newlines to spaces, keep blank lines as paragraph separators
    full_text = re.sub(r"(?<!\n)\n(?!\n)", " ", full_text)

    raw_paras = re.split(r"\n\s*\n+", full_text)
    return _clean_paragraphs(raw_paras)


def extract_paragraphs(file_path: str) -> List[str]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return docx_to_paragraphs(file_path)
    if ext == ".pdf":
        return pdf_to_paragraphs(file_path)
    raise ValueError("Unsupported file type")


def chunk_paragraphs(paragraphs: List[str], max_words: int, overlap_words: int) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    def flush():
        nonlocal cur, cur_words
        if not cur:
            return
        text = " ".join(cur).strip()
        if text:
            chunks.append(text)
        if overlap_words > 0 and text:
            tail = " ".join(text.split()[-overlap_words:])
            cur = [tail] if tail else []
            cur_words = len(tail.split()) if tail else 0
        else:
            cur, cur_words = [], 0

    for p in paragraphs:
        w = len((p or "").split())
        if w == 0:
            continue
        if cur and (cur_words + w > max_words):
            flush()
        cur.append(p)
        cur_words += w

    if cur:
        text = " ".join(cur).strip()
        if text:
            chunks.append(text)

    return chunks


# -------------------- Auto-tagging --------------------
@dataclass(frozen=True)
class TagDef:
    name: str
    description: str
    keywords: Tuple[str, ...]


TAG_DEFS: List[TagDef] = [
    TagDef("buyout manager", "Private equity buyout fund manager, LBO strategy, acquisitions, leverage, EBITDA.", ("lbo", "buyout", "leveraged", "acquisition", "ebitda", "sponsor", "private equity")),
    TagDef("venture capital", "Venture capital manager investing in startups, seed/series rounds, growth equity.", ("venture", "seed", "series a", "series b", "term sheet", "startup")),
    TagDef("private credit", "Private credit or direct lending manager, loans, yields, covenants, credit.", ("direct lending", "credit", "loan", "covenant", "yield", "spread", "amortization")),
    TagDef("real assets", "Infrastructure, real estate, energy, real assets strategy.", ("infrastructure", "real estate", "renewables", "asset", "project finance")),
    TagDef("hedge fund", "Hedge fund strategies, trading, long/short, derivatives, macro.", ("hedge", "alpha", "long/short", "derivative", "macro", "volatility")),
    TagDef("fundraising / LP", "Fundraising materials, LP communications, investor relations, DDQ.", ("lp", "limited partner", "fundraising", "ddq", "diligence", "track record", "ir")),
    TagDef("fees / economics", "Fees, carry, waterfalls, management fees, fund terms.", ("carry", "carried interest", "waterfall", "management fee", "hurdle", "gp", "lp")),
    TagDef("risk", "Risk factors, drawdowns, limits, compliance, controls.", ("risk", "drawdown", "limit", "compliance", "control", "stress")),
    TagDef("ESG", "ESG / sustainability policy, emissions, governance, social impact.", ("esg", "sustainability", "emissions", "governance", "diversity")),
]

_tag_embs: Optional[np.ndarray] = None


def get_tag_embeddings() -> np.ndarray:
    global _tag_embs
    if _tag_embs is None:
        _tag_embs = embed_texts([t.description for t in TAG_DEFS])
        _tag_embs = _l2_normalize(_tag_embs)
    return _tag_embs


def auto_tags(filename: str, chunks: List[str], chunk_embs: Optional[np.ndarray] = None, top_n: int = 4) -> List[str]:
    """
    Free/cheap tagging:
      - embed document (mean of first few chunk embeddings) and match against tag descriptions
      - add small keyword boosts
    """
    if not chunks:
        return []

    # doc vector from first N chunks
    n = min(len(chunks), 8)
    if chunk_embs is None:
        chunk_embs = embed_texts(chunks[:n])
    docv = _l2_normalize(np.mean(chunk_embs[:n], axis=0))

    tag_embs = get_tag_embeddings()
    sims = tag_embs @ docv  # (T,)

    fname = (filename or "").lower()
    text_head = " ".join(chunks[: min(len(chunks), 4)]).lower()

    scored: List[Tuple[str, float]] = []
    for i, td in enumerate(TAG_DEFS):
        score = float(sims[i])

        # keyword boost
        kw_hits = 0
        for kw in td.keywords:
            kw_l = kw.lower()
            if kw_l in fname:
                kw_hits += 2
            if kw_l in text_head:
                kw_hits += 1
        score += 0.03 * kw_hits

        scored.append((td.name, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # keep top_n with a mild threshold
    out = []
    for name, sc in scored[:top_n]:
        if sc >= 0.18 or len(out) < 2:  # ensure at least a couple tags
            out.append(name)
    return out


# -------------------- Files DB --------------------
def load_files_db() -> Dict:
    if not os.path.exists(FILES_DB_PATH):
        return {"version": 1, "files": {}}
    try:
        with open(FILES_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "files": {}}


def save_files_db(db: Dict) -> None:
    tmp = FILES_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FILES_DB_PATH)


def compute_file_id(dataroom: str, relpath: str) -> str:
    h = hashlib.sha1(f"{dataroom}::{relpath}".encode("utf-8")).hexdigest()
    return h[:16]


def list_files(db: Dict) -> List[Dict]:
    files = list(db.get("files", {}).values())
    # sort by dataroom then relpath
    files.sort(key=lambda r: (r.get("dataroom", "").lower(), r.get("relpath", "").lower()))
    return files


def datarooms(db: Dict) -> List[str]:
    rooms = sorted({r.get("dataroom", "default") for r in db.get("files", {}).values()}, key=lambda s: s.lower())
    return rooms


# -------------------- Safe path handling --------------------
def sanitize_relpath(relpath: str) -> str:
    relpath = (relpath or "").replace("\\", "/").strip()
    relpath = relpath.lstrip("/")

    parts = []
    for part in relpath.split("/"):
        part = part.strip()
        if not part or part in (".", ".."):
            continue
        parts.append(secure_filename(part) or "file")
    return "/".join(parts) if parts else "file"


def ensure_unique_path(base_dir: str, relpath: str) -> str:
    """
    If base_dir/relpath exists, append _N before extension.
    """
    relpath = sanitize_relpath(relpath)
    abs_path = os.path.join(base_dir, relpath)

    if not os.path.exists(abs_path):
        return relpath

    root, ext = os.path.splitext(relpath)
    n = 1
    while True:
        cand = f"{root}_{n}{ext}"
        if not os.path.exists(os.path.join(base_dir, cand)):
            return cand
        n += 1


# -------------------- Indexing --------------------
def allowed_ext(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTS


def write_per_file_csv(csv_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["dataroom", "file", "chunk_id", "tags", "text"])
        for r in rows:
            w.writerow([r["dataroom"], r["file"], r["chunk_id"], r["tags"], r["text"]])


def index_one_file(dataroom: str, relpath: str, abs_path: str, manual_tags: str = "") -> Tuple[str, int]:
    """
    Extract -> chunk -> embed -> store in Chroma -> write CSV -> update files DB.
    Returns (file_id, chunks_indexed)
    """
    dataroom = (dataroom or "default").strip() or "default"
    relpath = sanitize_relpath(relpath)

    paragraphs = extract_paragraphs(abs_path)
    chunks = chunk_paragraphs(paragraphs, CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)
    if not chunks:
        return (compute_file_id(dataroom, relpath), 0)

    # embeddings in batches
    chunk_vecs = []
    batch = 32
    for i in range(0, len(chunks), batch):
        chunk_vecs.append(embed_texts(chunks[i : i + batch]))
    chunk_vecs = np.vstack(chunk_vecs)

    # tags
    tags_auto = auto_tags(os.path.basename(relpath), chunks, chunk_embs=chunk_vecs)
    tags_manual = []
    if manual_tags.strip():
        tags_manual = [t.strip() for t in manual_tags.split(",") if t.strip()]
    tags = []
    seen = set()
    for t in tags_manual + tags_auto:
        tl = t.lower()
        if tl not in seen:
            tags.append(t)
            seen.add(tl)
    tags_str = ", ".join(tags) if tags else ""

    file_id = compute_file_id(dataroom, relpath)
    csv_name = f"{file_id}.csv"
    csv_path = os.path.join(INDEX_DIR, csv_name)

    # store in chroma
    col = get_collection()
    ids = [f"{file_id}::{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "file_id": file_id,
            "dataroom": dataroom,
            "relpath": relpath,
            "chunk_id": int(i),
            "tags": tags_str,
        }
        for i in range(len(chunks))
    ]

    # Chroma expects plain python types
    col.add(
        ids=ids,
        embeddings=[v.tolist() for v in chunk_vecs],
        documents=chunks,
        metadatas=metadatas,
    )

    # write CSV
    csv_rows = [
        {"dataroom": dataroom, "file": relpath, "chunk_id": i, "tags": tags_str, "text": chunks[i]}
        for i in range(len(chunks))
    ]
    write_per_file_csv(csv_path, csv_rows)

    # update files DB
    db = load_files_db()
    db.setdefault("files", {})
    db["files"][file_id] = {
        "file_id": file_id,
        "dataroom": dataroom,
        "relpath": relpath,
        "abs_path": abs_path,
        "tags": tags,
        "tags_str": tags_str,
        "chunks": len(chunks),
        "csv_name": csv_name,
        "added_at": int(time.time()),
    }
    save_files_db(db)

    return (file_id, len(chunks))


def delete_file_from_index(file_id: str) -> None:
    db = load_files_db()
    rec = db.get("files", {}).get(file_id)
    if not rec:
        return

    # delete from chroma
    col = get_collection()
    try:
        col.delete(where={"file_id": file_id})
    except Exception:
        # if where delete isn't supported in some environments, fall back to ids delete
        try:
            # best-effort: delete by enumerating known chunks
            n = int(rec.get("chunks", 0) or 0)
            ids = [f"{file_id}::{i}" for i in range(n)]
            col.delete(ids=ids)
        except Exception:
            pass

    # delete CSV
    csv_name = rec.get("csv_name") or ""
    if csv_name:
        csv_path = os.path.join(INDEX_DIR, os.path.basename(csv_name))
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except Exception:
                pass

    # delete upload file
    abs_path = rec.get("abs_path") or ""
    if abs_path and os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except Exception:
            pass

    # remove empty directories under uploads (best-effort)
    try:
        parent = os.path.dirname(abs_path)
        while parent and os.path.abspath(parent).startswith(os.path.abspath(UPLOAD_DIR)):
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
                parent = os.path.dirname(parent)
            else:
                break
    except Exception:
        pass

    # update DB
    del db["files"][file_id]
    save_files_db(db)


def scan_uploads_for_reindex() -> List[Tuple[str, str, str]]:
    """
    Returns list of (dataroom, relpath, abs_path) for all allowed files under uploads/.
    Expected structure: uploads/<dataroom>/<any nested>/<file>
    Files at uploads/ root are assigned dataroom="default".
    """
    out = []
    for root, _, files in os.walk(UPLOAD_DIR):
        for name in files:
            if not allowed_ext(name):
                continue
            abs_path = os.path.join(root, name)
            rel_from_uploads = os.path.relpath(abs_path, UPLOAD_DIR).replace("\\", "/")
            parts = [p for p in rel_from_uploads.split("/") if p]
            if len(parts) == 1:
                dataroom = "default"
                relpath = parts[0]
            else:
                dataroom = parts[0] or "default"
                relpath = "/".join(parts[1:])
            out.append((dataroom, sanitize_relpath(relpath), abs_path))
    out.sort(key=lambda x: (x[0].lower(), x[1].lower()))
    return out


def reindex_all_uploads(manual_tags: str = "") -> Tuple[int, int]:
    """
    Rebuild Chroma + files DB + per-file CSVs from uploads/ recursively.
    Returns (files_indexed, chunks_indexed).
    """
    reset_collection()

    # clear files DB + old CSVs
    db = {"version": 1, "files": {}}
    save_files_db(db)
    for name in os.listdir(INDEX_DIR):
        if name.lower().endswith(".csv"):
            try:
                os.remove(os.path.join(INDEX_DIR, name))
            except Exception:
                pass

    items = scan_uploads_for_reindex()
    files_indexed = 0
    chunks_indexed = 0

    for dataroom, relpath, abs_path in items:
        try:
            _, n = index_one_file(dataroom, relpath, abs_path, manual_tags=manual_tags)
            if n > 0:
                files_indexed += 1
                chunks_indexed += n
        except Exception:
            # keep going; errors will surface via UI message if needed
            continue

    return (files_indexed, chunks_indexed)


# -------------------- Retrieval (Search) --------------------
def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def highlight_escaped(escaped_text: str, query: str) -> str:
    toks = sorted(set(_WORD_RE.findall((query or "").lower())), key=len, reverse=True)
    out = escaped_text
    for t in toks:
        if not t:
            continue
        out = re.sub(rf"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", out)
    return out


def vector_search(query: str, k: int, dataroom: str = "", tag: str = "") -> List[Dict]:
    query = (query or "").strip()
    if not query:
        return []

    col = get_collection()
    qv = embed_texts([query])[0].tolist()

    where = {}
    if dataroom.strip():
        where["dataroom"] = dataroom.strip()

    # If tag filter is set, Chroma can't do "contains" on metadata, so we over-retrieve and filter in Python.
    over_k = max(k, 10)
    if tag.strip():
        over_k = max(over_k, k * 6)

    res = col.query(
        query_embeddings=[qv],
        n_results=over_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []
    tag_l = tag.strip().lower()
    for doc, meta, dist in zip(docs, metas, dists):
        tags_str = (meta.get("tags") or "")
        if tag_l and (tag_l not in tags_str.lower()):
            continue

        # cosine distance -> similarity approx (1 - distance)
        sim = 1.0 - float(dist)
        out.append(
            {
                "file_id": meta.get("file_id", ""),
                "dataroom": meta.get("dataroom", ""),
                "file": meta.get("relpath", ""),
                "chunk_id": int(meta.get("chunk_id", 0) or 0),
                "tags": tags_str,
                "score": sim,
                "text": doc or "",
            }
        )
        if len(out) >= k:
            break

    return out

def _gemini_generate(question: str, evidence_blocks: List[Tuple[str, str]]) -> str:
    """
    Calls Gemini generateContent.
    evidence_blocks: [(label, text), ...] where label already includes [D#]
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    instructions = (
        "You are a retrieval-augmented assistant. Use ONLY the provided evidence.\n"
        "If the evidence is insufficient, say so explicitly.\n"
        "Cite claims with bracket citations like [D1], [D2] matching the evidence labels.\n"
        "Provide:\n"
        "1) A short answer (3-8 sentences)\n"
        "2) A 'Consensus' section that reconciles agreements/disagreements across sources\n"
        "3) An 'Evidence' section with bullet points keyed to citations\n"
    )

    evidence_text = "\n\n".join([f"{lbl} {txt}" for lbl, txt in evidence_blocks])
    prompt = f"{instructions}\n\nQUESTION:\n{question}\n\nEVIDENCE:\n{evidence_text}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini generation error ({r.status_code}): {r.text[:4000]}")

    data = r.json()
    # Typical response: candidates[0].content.parts[0].text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text.strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)[:4000]

# -------------------- RAG Generation --------------------
def _openai_generate(question: str, evidence_blocks: List[Tuple[str, str]]) -> str:
    """
    evidence_blocks: [(citation, text), ...]
    Uses OpenAI Responses API if OPENAI_API_KEY is set.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    instructions = (
        "You are a retrieval-augmented assistant. Use ONLY the provided evidence.\n"
        "If the evidence is insufficient, say so explicitly.\n"
        "Cite claims with bracket citations like [D1], [D2] matching the evidence labels.\n"
        "Provide:\n"
        "1) A short answer (3-8 sentences)\n"
        "2) A 'Consensus' section that reconciles agreements/disagreements across sources\n"
        "3) An 'Evidence' section with bullet points keyed to citations\n"
    )

    evidence_text = "\n\n".join([f"{c} {t}" for c, t in evidence_blocks])
    user_input = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_text}"

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GEN_MODEL,
        "instructions": instructions,
        "input": user_input,
        "temperature": 0.2,
        "max_output_tokens": 900,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI generation error ({r.status_code}): {r.text[:4000]}")
    data = r.json()

    # Parse output_text items
    texts = []
    for item in data.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                texts.append(c["text"])
    return "\n".join(texts).strip()


def _extractive_consensus(question: str, evidence_blocks: List[Tuple[str, str]]) -> str:
    """
    Free fallback: not an LLM, but keeps the pipeline functional.
    Produces a conservative synthesis + citations + excerpts.
    """
    q_words = set(_WORD_RE.findall(question.lower()))
    scored = []
    for cit, txt in evidence_blocks:
        words = _WORD_RE.findall((txt or "").lower())
        overlap = sum(1 for w in words if w in q_words)
        scored.append((overlap, cit, txt))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[: min(len(scored), 10)]
    bullets = []
    for _, cit, txt in top:
        # take first ~2 sentences worth
        snippet = txt.strip()
        if len(snippet) > 420:
            snippet = snippet[:420].rsplit(" ", 1)[0] + "…"
        bullets.append(f"- {cit} {snippet}")

    answer_lines = []
    answer_lines.append("LLM is not configured; showing an extractive, evidence-only synthesis.")
    answer_lines.append("")
    answer_lines.append("Evidence excerpts:")
    answer_lines.extend(bullets)
    answer_lines.append("")
    answer_lines.append("Consensus (heuristic):")
    answer_lines.append(
        "Across the retrieved excerpts above, the most relevant points are those with the highest keyword overlap "
        "to the question. If you want a true generated answer, set OPENAI_API_KEY and GEN_PROVIDER=openai."
    )
    return "\n".join(answer_lines)

def rag_answer(question: str, dataroom: str = "", tag: str = "") -> Dict:
    raw = vector_search(question, k=max(RAG_TOP_K, 8), dataroom=dataroom, tag=tag)

    by_file: Dict[str, int] = {}
    evidence: List[Dict] = []
    for r in raw:
        key = f"{r['dataroom']}/{r['file']}"
        by_file[key] = by_file.get(key, 0) + 1
        if by_file[key] > RAG_PER_FILE_CAP:
            continue
        evidence.append(r)

    evidence_blocks: List[Tuple[str, str]] = []
    total_chars = 0
    for i, r in enumerate(evidence, start=1):
        cit = f"[D{i}]"
        label = f"{cit} ({r['dataroom']}/{r['file']}#chunk{r['chunk_id']})"
        txt = (r["text"] or "").strip()
        if not txt:
            continue
        add = len(label) + 1 + len(txt) + 2
        if total_chars + add > RAG_MAX_CONTEXT_CHARS:
            break
        evidence_blocks.append((label, txt))
        total_chars += add

    # generate (AFTER building evidence_blocks)
    if GEN_PROVIDER == "gemini":
        try:
            answer = _gemini_generate(question, evidence_blocks)
        except Exception as e:
            answer = _extractive_consensus(question, evidence_blocks) + f"\n\n(Generation error: {e})"
    elif GEN_PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            answer = _openai_generate(question, evidence_blocks)
        except Exception as e:
            answer = _extractive_consensus(question, evidence_blocks) + f"\n\n(Generation error: {e})"
    else:
        answer = _extractive_consensus(question, evidence_blocks)

    return {"question": question, "answer": answer, "evidence": evidence}


# -------------------- HTML (Dark Mode + RAG) --------------------
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="color-scheme" content="dark" />
    <title>RAG Search</title>
    <style>
      :root {{
        --bg: #0b0f14;
        --panel: #111826;
        --panel-2: #0e1522;
        --border: #223045;
        --text: #e6edf3;
        --muted: #9fb0c0;
        --muted-2: #7f93a8;
        --accent: #7aa2ff;
        --accent-2: #5eead4;
        --danger: #ff6b6b;
        --shadow: 0 8px 24px rgba(0,0,0,.35);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
        margin: 2rem;
        max-width: 1040px;
        background:
          radial-gradient(1000px 600px at 20% -10%, rgba(122,162,255,.20), transparent 60%),
          radial-gradient(900px 600px at 100% 0%, rgba(94,234,212,.12), transparent 55%),
          var(--bg);
        color: var(--text);
      }}
      h1 {{ margin: 0 0 1rem 0; }}
      h2 {{ margin: 0 0 .75rem 0; font-size: 1.1rem; }}
      h3 {{ margin: 1rem 0 .5rem 0; font-size: 1rem; }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .grid {{ display: grid; grid-template-columns: 1.15fr .85fr; gap: 1.25rem; }}
      @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
      .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00)), var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: var(--shadow);
      }}
      .msg {{
        padding: .7rem .9rem;
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: .75rem;
        margin: 1rem 0;
        color: var(--muted);
        white-space: pre-wrap;
      }}
      .hint {{ color: var(--muted-2); margin-top: .5rem; }}
      .small {{ font-size: .92rem; color: var(--muted); }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}

      form {{ display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; }}
      .stack {{ display: grid; gap: .6rem; }}
      label {{ color: var(--muted); font-size: .9rem; }}
      input[type="text"], select {{
        padding: .6rem .7rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,.03);
        color: var(--text);
        outline: none;
      }}
      input[type="text"] {{ width: min(620px, 100%); }}
      input[type="text"]::placeholder {{ color: rgba(159,176,192,.65); }}
      input[type="text"]:focus, select:focus {{
        border-color: rgba(122,162,255,.75);
        box-shadow: 0 0 0 4px rgba(122,162,255,.12);
      }}
      input[type="file"] {{ color: var(--muted); max-width: 100%; }}
      button {{
        padding: .62rem .9rem;
        border-radius: 10px;
        border: 1px solid rgba(122,162,255,.35);
        background: rgba(122,162,255,.14);
        color: var(--text);
        cursor: pointer;
      }}
      button:hover {{ background: rgba(122,162,255,.20); }}
      button:active {{ transform: translateY(1px); }}
      .danger {{
        border-color: rgba(255,107,107,.35);
        background: rgba(255,107,107,.14);
      }}
      .danger:hover {{ background: rgba(255,107,107,.20); }}

      ul {{ padding-left: 1.25rem; }}
      li {{ margin: .55rem 0; }}
      code {{
        background: rgba(255,255,255,.06);
        padding: .12rem .28rem;
        border-radius: .3rem;
        border: 1px solid rgba(255,255,255,.08);
      }}
      mark {{
        background: rgba(94,234,212,.22);
        color: var(--text);
        padding: 0 .14rem;
        border-radius: .25rem;
      }}
      .row {{ display: flex; gap: .6rem; flex-wrap: wrap; align-items: baseline; }}
      .pill {{
        display: inline-block;
        padding: .15rem .45rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.04);
        color: var(--muted);
        font-size: .82rem;
      }}
      .fileline {{
        display: flex;
        gap: .6rem;
        flex-wrap: wrap;
        align-items: center;
      }}
      .fileline form {{ display: inline-flex; }}
      .answer {{
        margin-top: .75rem;
        padding: .75rem .9rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.03);
        white-space: pre-wrap;
      }}
    </style>
  </head>
  <body>
    <h1>RAG Search</h1>
    {msg}

    <div class="grid">
      <div class="card">
        <h2>Search / Ask (RAG)</h2>
        <form method="get" action="/">
          <input name="q" type="text" placeholder="Search or ask a question across your dataroom..." value="{q}" />

          <select name="dataroom" title="Filter by dataroom">
            <option value="">All datarooms</option>
            {dataroom_options}
          </select>

          <input name="tag" type="text" placeholder="Optional tag filter (e.g., buyout manager)" value="{tag}" />

          <button type="submit" name="mode" value="search">Search</button>
          <button type="submit" name="mode" value="ask">Ask (RAG)</button>
        </form>

        <div class="hint">
          Endpoints: <code>/api/search</code>, <code>/api/ask</code>, <code>/api/files</code>.
          {llm_hint}
        </div>

        <div class="small">
          Indexed chunks: <span class="mono">{total_chunks}</span> · Files: <span class="mono">{total_files}</span>
        </div>

        {results_block}
      </div>

      <div class="card">
        <h2>Upload</h2>
        <div class="small">
          Upload individual files or a whole folder (Chrome/Edge: folder picker). Files are stored under <span class="mono">uploads/&lt;dataroom&gt;/...</span>
        </div>

        <form method="post" action="/upload" enctype="multipart/form-data" class="stack" style="margin-top:.75rem;">
          <div class="row">
            <label>Dataroom</label>
            <input name="dataroom" type="text" placeholder="e.g., Manager_A" value="default" />
          </div>

          <div class="row">
            <label>Optional manual tags (comma-separated)</label>
            <input name="manual_tags" type="text" placeholder="e.g., buyout manager, healthcare" />
          </div>

          <div class="stack">
            <label>Select files</label>
            <input name="files" type="file" accept=".pdf,.docx" multiple />
          </div>

          <div class="stack">
            <label>Select folder (Chrome/Edge)</label>
            <input name="folder" type="file" webkitdirectory directory multiple />
          </div>

          <div class="row">
            <button type="submit">Upload & Index</button>
          </div>
        </form>

        <div class="row" style="margin-top:.9rem;">
          <form method="post" action="/reindex">
            <button type="submit">Reindex all uploads</button>
          </form>
          <span class="small">Rebuilds Chroma + CSVs from everything currently under <span class="mono">uploads/</span>.</span>
        </div>

        <h3>Files</h3>
        {files_block}
      </div>
    </div>
  </body>
</html>
"""


def render_dataroom_options(selected: str, db: Dict) -> str:
    selected = (selected or "").strip()
    opts = []
    for room in datarooms(db):
        sel = " selected" if room == selected else ""
        opts.append(f"<option value='{escape_html(room)}'{sel}>{escape_html(room)}</option>")
    return "\n".join(opts)


def render_files_block(db: Dict) -> str:
    files = list_files(db)
    if not files:
        return "<p class='small'>No files indexed yet.</p>"

    lis = []
    for r in files:
        file_id = r.get("file_id", "")
        room = r.get("dataroom", "")
        relpath = r.get("relpath", "")
        tags_str = r.get("tags_str", "")
        chunks = int(r.get("chunks", 0) or 0)
        csv_name = r.get("csv_name", "")

        lis.append(
            "<li>"
            "<div class='fileline'>"
            f"<span class='mono'>{escape_html(room)}/{escape_html(relpath)}</span>"
            f"<span class='pill'>{chunks} chunks</span>"
            + (f"<span class='pill'>{escape_html(tags_str)}</span>" if tags_str else "")
            + (f"<a href='/download/{escape_html(csv_name)}'>CSV</a>" if csv_name else "")
            + (
                f"<form method='post' action='/remove' onsubmit='return confirm(\"Remove {escape_html(room)}/{escape_html(relpath)}?\")'>"
                f"<input type='hidden' name='file_id' value='{escape_html(file_id)}' />"
                f"<button class='danger' type='submit'>Remove</button>"
                "</form>"
            )
            + "</div></li>"
        )
    return "<ul>" + "".join(lis) + "</ul>"


def render_search_results(q: str, results: List[Dict]) -> str:
    if not q.strip():
        return "<p class='small'>Enter a query to search.</p>"
    if not results:
        return "<p>No results.</p>"

    lis = []
    for r in results:
        esc_text = escape_html(r["text"])
        shown = highlight_escaped(esc_text, q)
        score = f"{r['score']:.4f}"
        lis.append(
            "<li>"
            f"<div class='row'><strong>{escape_html(r['dataroom'])}/{escape_html(r['file'])}</strong>"
            f"<span class='pill'>chunk {r['chunk_id']}</span>"
            f"<span class='pill'>score {score}</span>"
            + (f"<span class='pill'>{escape_html(r.get('tags',''))}</span>" if r.get("tags") else "")
            + "</div>"
            f"<div class='small'>{shown}</div>"
            "</li>"
        )
    return f"<p>Top results ({len(results)}):</p><ul>{''.join(lis)}</ul>"


def render_ask_block(q: str, rag: Dict) -> str:
    if not q.strip():
        return "<p class='small'>Enter a question, then click Ask (RAG).</p>"

    ans = rag.get("answer", "")
    evidence = rag.get("evidence", []) or []

    ev_lis = []
    for i, r in enumerate(evidence[:24], start=1):
        esc_text = escape_html(r["text"])
        shown = highlight_escaped(esc_text, q)
        score = f"{r['score']:.4f}"
        ev_lis.append(
            "<li>"
            f"<div class='row'><strong>[D{i}] {escape_html(r['dataroom'])}/{escape_html(r['file'])}</strong>"
            f"<span class='pill'>chunk {r['chunk_id']}</span>"
            f"<span class='pill'>score {score}</span></div>"
            f"<div class='small'>{shown}</div>"
            "</li>"
        )

    evidence_html = "<p class='small'>No evidence retrieved. Upload and index files first.</p>" if not ev_lis else "<ul>" + "".join(ev_lis) + "</ul>"
    return (
        "<h3>Answer</h3>"
        f"<div class='answer'>{escape_html(ans)}</div>"
        "<h3>Evidence (retrieved chunks)</h3>"
        + evidence_html
    )


# -------------------- Routes --------------------
@app.get("/")
def index():
    db = load_files_db()

    q = request.args.get("q", "")
    mode = request.args.get("mode", "search").strip().lower()
    room = request.args.get("dataroom", "").strip()
    tag = request.args.get("tag", "").strip()
    msg = request.args.get("msg", "")

    # counts
    total_files = len(db.get("files", {}))
    total_chunks = int(sum(int(r.get("chunks", 0) or 0) for r in db.get("files", {}).values()))

    msg_html = f"<div class='msg'>{escape_html(msg)}</div>" if msg else ""

    llm_hint = ""
    if GEN_PROVIDER == "openai":
        if OPENAI_API_KEY:
            llm_hint = f" Using OpenAI generation model: <code>{escape_html(GEN_MODEL)}</code>."
        else:
            llm_hint = " <span class='pill'>GEN_PROVIDER=openai</span> but <code>OPENAI_API_KEY</code> is not set; using extractive fallback."

    if mode == "ask" and q.strip():
        rag = rag_answer(q, dataroom=room, tag=tag)
        results_block = render_ask_block(q, rag)
    else:
        results = vector_search(q, k=MAX_RESULTS, dataroom=room, tag=tag) if q.strip() else []
        results_block = render_search_results(q, results)

    return HTML.format(
        msg=msg_html,
        q=escape_html(q),
        tag=escape_html(tag),
        dataroom_options=render_dataroom_options(room, db),
        llm_hint=llm_hint,
        total_files=total_files,
        total_chunks=total_chunks,
        results_block=results_block,
        files_block=render_files_block(db),
    )


@app.post("/upload")
def upload():
    dataroom = (request.form.get("dataroom") or "default").strip() or "default"
    manual_tags = (request.form.get("manual_tags") or "").strip()

    # gather uploaded files from both selectors
    uploads: List = []
    uploads.extend(request.files.getlist("files") or [])
    uploads.extend(request.files.getlist("folder") or [])

    # filter empties
    uploads = [f for f in uploads if f and f.filename]

    if not uploads:
        return redirect(url_for("index", msg="No files selected."))

    # store under uploads/<dataroom>/...
    room_dir = os.path.join(UPLOAD_DIR, secure_filename(dataroom) or "default")
    os.makedirs(room_dir, exist_ok=True)

    indexed_files = 0
    indexed_chunks = 0
    skipped = 0
    errors = 0

    for f in uploads:
        raw_name = f.filename  # may include relative path for folder uploads
        relpath = sanitize_relpath(raw_name)
        if not allowed_ext(relpath):
            skipped += 1
            continue

        # ensure unique path inside room_dir
        relpath_unique = ensure_unique_path(room_dir, relpath)
        abs_path = os.path.join(room_dir, relpath_unique)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        try:
            f.save(abs_path)
        except Exception:
            errors += 1
            continue

        try:
            _, n = index_one_file(dataroom, relpath_unique, abs_path, manual_tags=manual_tags)
            if n > 0:
                indexed_files += 1
                indexed_chunks += n
            else:
                skipped += 1
        except Exception:
            errors += 1
            continue

    msg = f"Upload complete. Indexed {indexed_files} file(s), {indexed_chunks} chunk(s)."
    if skipped:
        msg += f" Skipped {skipped} (unsupported/empty)."
    if errors:
        msg += f" Errors: {errors}."
    return redirect(url_for("index", msg=msg))


@app.post("/reindex")
def reindex():
    try:
        files_n, chunks_n = reindex_all_uploads()
        return redirect(url_for("index", msg=f"Reindexed {files_n} file(s), {chunks_n} chunk(s)."))
    except Exception as e:
        return redirect(url_for("index", msg=f"Reindex failed: {e}"))


@app.post("/remove")
def remove():
    file_id = (request.form.get("file_id") or "").strip()
    if not file_id:
        return redirect(url_for("index", msg="Missing file_id."))

    try:
        delete_file_from_index(file_id)
        return redirect(url_for("index", msg="Removed file from index and disk."))
    except Exception as e:
        return redirect(url_for("index", msg=f"Remove failed: {e}"))


@app.get("/download/<path:csv_name>")
def download(csv_name: str):
    # only allow download from INDEX_DIR
    csv_name = os.path.basename(csv_name)
    return send_from_directory(INDEX_DIR, csv_name, as_attachment=True)


# -------------------- JSON APIs --------------------
@app.get("/api/files")
def api_files():
    db = load_files_db()
    return jsonify(
        {
            "files": list_files(db),
            "total_files": len(db.get("files", {})),
            "total_chunks": int(sum(int(r.get("chunks", 0) or 0) for r in db.get("files", {}).values())),
        }
    )


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "")
    room = request.args.get("dataroom", "")
    tag = request.args.get("tag", "")
    k = request.args.get("k", "")
    try:
        k_int = int(k) if k else MAX_RESULTS
    except ValueError:
        k_int = MAX_RESULTS

    return jsonify(
        {
            "query": q,
            "dataroom": room,
            "tag": tag,
            "results": vector_search(q, k=k_int, dataroom=room, tag=tag),
        }
    )


@app.get("/api/ask")
def api_ask():
    q = request.args.get("q", "")
    room = request.args.get("dataroom", "")
    tag = request.args.get("tag", "")
    return jsonify(rag_answer(q, dataroom=room, tag=tag))


# -------------------- Main --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

