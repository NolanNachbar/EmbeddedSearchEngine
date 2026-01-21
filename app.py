# app.py
# pip install flask sentence-transformers torch numpy python-docx pypdf

from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, redirect, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------- Config ----------
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
INDEX_DIR = os.path.join(os.getcwd(), "index_data")

META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")        # one JSON record per paragraph
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")     # (N, D) float32 embeddings
ALLOWED_EXTS = {".pdf", ".docx"}

MAX_RESULTS = 20

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- Model (lazy-loaded) ----------
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


# ---------- Text extraction ----------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _clean_paragraphs(
    paragraphs: List[str],
    min_chars: int = 120,
    min_words: int = 6,
) -> List[str]:
    """
    Merge short lines into more paragraph-like chunks; drop very short leftovers.
    This reduces title-page / header line-splitting issues.
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


# ---------- Index state ----------
records: List[Dict] = []              # {"file": str, "para_id": int, "text": str}
embeddings: np.ndarray | None = None  # (N, D)


def load_index():
    global records, embeddings
    records = []
    embeddings = None

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if os.path.exists(EMB_PATH):
        try:
            embeddings = np.load(EMB_PATH)
        except Exception:
            embeddings = None

    if embeddings is not None and len(records) != embeddings.shape[0]:
        embeddings = None


def write_full_index(new_records: List[Dict], new_embs: np.ndarray):
    global records, embeddings
    records = new_records
    embeddings = new_embs

    # overwrite metadata
    with open(META_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # overwrite embeddings
    np.save(EMB_PATH, embeddings)


def append_to_index(new_records: List[Dict], new_embs: np.ndarray):
    global records, embeddings
    records.extend(new_records)
    embeddings = new_embs if embeddings is None else np.vstack([embeddings, new_embs])

    with open(META_PATH, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    np.save(EMB_PATH, embeddings)


load_index()


# ---------- Search ----------
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


def semantic_search(query: str, k: int = MAX_RESULTS) -> List[Dict]:
    if not query or not query.strip():
        return []
    if embeddings is None or embeddings.shape[0] == 0:
        return []

    qv = embed_texts([query.strip()])[0]
    scores = embeddings @ qv

    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out = []
    for i in idx:
        r = records[int(i)]
        out.append(
            {
                "file": r["file"],
                "para_id": r["para_id"],
                "score": float(scores[int(i)]),
                "text": r["text"],
            }
        )
    return out


def file_summary() -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["file"]] = counts.get(r["file"], 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))


def list_uploaded_files() -> List[str]:
    files = []
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXTS:
            files.append(name)
    files.sort(key=lambda s: s.lower())
    return files


def safe_join_upload(filename: str) -> str | None:
    # Only allow deleting files that exist in UPLOAD_DIR exactly
    filename = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(path):
        return path
    return None


def safe_join_index(filename: str) -> str | None:
    filename = os.path.basename(filename)
    path = os.path.join(INDEX_DIR, filename)
    if os.path.isfile(path):
        return path
    return None


# ---------- HTML (Dark Mode + Buttons) ----------
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="color-scheme" content="dark" />
    <title>Semantic Search</title>
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
        max-width: 980px;
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
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.25rem; }}
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
      }}
      .hint {{ color: var(--muted-2); margin-top: .5rem; }}
      .small {{ font-size: .92rem; color: var(--muted); }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}

      form {{ display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; }}
      .stack {{ display: grid; gap: .6rem; }}
      input[type="text"] {{
        padding: .6rem .7rem;
        width: min(620px, 95%);
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,.03);
        color: var(--text);
        outline: none;
      }}
      input[type="text"]::placeholder {{ color: rgba(159,176,192,.65); }}
      input[type="text"]:focus {{
        border-color: rgba(122,162,255,.75);
        box-shadow: 0 0 0 4px rgba(122,162,255,.12);
      }}
      input[type="file"] {{ color: var(--muted); }}

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
    </style>
  </head>
  <body>
    <h1>Semantic Search</h1>
    {msg}

    <div class="grid">
      <div class="card">
        <h2>Search</h2>
        <form method="get" action="/">
          <input name="q" type="text" placeholder="Search across uploaded documents..." value="{q}" />
          <button type="submit">Search</button>
        </form>
        <div class="hint">JSON endpoints: <code>/api/search?q=...</code>, <code>/api/files</code></div>
        <div class="small">Indexed paragraphs: <span class="mono">{total_paras}</span></div>
        {results}
      </div>

      <div class="card">
        <h2>Upload (PDF or DOCX)</h2>
        <form method="post" action="/upload" enctype="multipart/form-data">
          <input name="file" type="file" accept=".pdf,.docx" required />
          <button type="submit">Upload & Index</button>
        </form>

        <div class="row" style="margin-top:.9rem;">
          <form method="post" action="/reindex">
            <button type="submit">Reindex all uploads</button>
          </form>
          <span class="small">Rebuilds the index from files currently in <span class="mono">uploads/</span>.</span>
        </div>

        <h3>Files in index</h3>
        {files}
      </div>
    </div>
  </body>
</html>
"""


def render_files() -> str:
    items = file_summary()
    if not items:
        return "<p class='small'>No files indexed yet.</p>"

    lis = []
    for fname, count in items:
        csv_name = f"{fname}.csv"
        lis.append(
            "<li>"
            "<div class='fileline'>"
            f"<span class='mono'>{escape_html(fname)}</span>"
            f"<span class='pill'>{count} paragraphs</span>"
            f"<a href='/download/{escape_html(csv_name)}'>download CSV</a>"
            f"<form method='post' action='/remove/{escape_html(fname)}' onsubmit='return confirm(\"Remove {escape_html(fname)} from index?\")'>"
            f"<button class='danger' type='submit'>Remove</button>"
            "</form>"
            "</div>"
            "</li>"
        )
    return "<ul>" + "".join(lis) + "</ul>"


def render_results(q: str, results: List[Dict]) -> str:
    if not q.strip():
        return "<p class='small'>Enter a query to search.</p>"
    if embeddings is None or embeddings.shape[0] == 0:
        return "<p class='small'>No embeddings indexed yet. Upload a PDF/DOCX first.</p>"
    if not results:
        return "<p>No results.</p>"

    lis = []
    for r in results:
        esc_text = escape_html(r["text"])
        shown = highlight_escaped(esc_text, q)
        score = f"{r['score']:.4f}"
        lis.append(
            "<li>"
            f"<div class='row'><strong>{escape_html(r['file'])}</strong>"
            f"<span class='pill'>para {r['para_id']}</span>"
            f"<span class='pill'>score {score}</span></div>"
            f"<div class='small'>{shown}</div>"
            "</li>"
        )
    return f"<p>Top results ({len(results)}):</p><ul>{''.join(lis)}</ul>"


# ---------- Index rebuild / maintenance ----------
def clear_index_files():
    # remove master index files
    for p in (META_PATH, EMB_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # remove per-file CSVs in INDEX_DIR
    for name in os.listdir(INDEX_DIR):
        path = os.path.join(INDEX_DIR, name)
        if os.path.isfile(path) and name.lower().endswith((".pdf.csv", ".docx.csv")):
            try:
                os.remove(path)
            except Exception:
                pass


def reindex_all_uploads() -> Tuple[int, int]:
    """
    Rebuild everything from uploads/.
    Returns (files_indexed, paragraphs_indexed)
    """
    new_records: List[Dict] = []
    all_texts: List[str] = []

    files = list_uploaded_files()
    for fname in files:
        fpath = os.path.join(UPLOAD_DIR, fname)
        paragraphs = extract_paragraphs(fpath)
        if not paragraphs:
            continue

        # per-file CSV
        csv_name = f"{fname}.csv"
        csv_path = os.path.join(INDEX_DIR, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as out:
            w = csv.writer(out)
            w.writerow(["file", "para_id", "text"])
            for i, p in enumerate(paragraphs):
                w.writerow([fname, i, p])

        for i, p in enumerate(paragraphs):
            new_records.append({"file": fname, "para_id": i, "text": p})
            all_texts.append(p)

    if not new_records:
        write_full_index([], np.zeros((0, 1), dtype=np.float32))
        # keep embeddings None-safe by reloading
        load_index()
        return (0, 0)

    new_embs = embed_texts(all_texts)
    write_full_index(new_records, new_embs)
    return (len(files), len(new_records))


# ---------- Routes ----------
@app.get("/")
def index():
    q = request.args.get("q", "")
    msg = request.args.get("msg", "")
    results = semantic_search(q) if q.strip() else []

    msg_html = f"<div class='msg'>{escape_html(msg)}</div>" if msg else ""

    return HTML.format(
        q=escape_html(q),
        msg=msg_html,
        total_paras=len(records),
        results=render_results(q, results),
        files=render_files(),
    )


@app.post("/upload")
def upload():
    if "file" not in request.files:
        return redirect(url_for("index", msg="No file part in request."))

    f = request.files["file"]
    if not f or not f.filename:
        return redirect(url_for("index", msg="No file selected."))

    orig_name = secure_filename(f.filename)
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in ALLOWED_EXTS:
        return redirect(url_for("index", msg="Only .pdf and .docx are supported."))

    filename = orig_name
    base, _ = os.path.splitext(orig_name)
    n = 1
    while os.path.exists(os.path.join(UPLOAD_DIR, filename)):
        filename = f"{base}_{n}{ext}"
        n += 1

    file_path = os.path.join(UPLOAD_DIR, filename)
    f.save(file_path)

    try:
        paragraphs = extract_paragraphs(file_path)
    except Exception as e:
        return redirect(url_for("index", msg=f"Failed to parse file: {e}"))

    if not paragraphs:
        return redirect(url_for("index", msg="No text paragraphs found in that file."))

    new_records = [{"file": filename, "para_id": i, "text": p} for i, p in enumerate(paragraphs)]

    try:
        new_embs = embed_texts([r["text"] for r in new_records])
    except Exception as e:
        return redirect(url_for("index", msg=f"Embedding failed: {e}"))

    # per-file CSV
    csv_name = f"{filename}.csv"
    csv_path = os.path.join(INDEX_DIR, csv_name)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as out:
            w = csv.writer(out)
            w.writerow(["file", "para_id", "text"])
            for r in new_records:
                w.writerow([r["file"], r["para_id"], r["text"]])
    except Exception as e:
        return redirect(url_for("index", msg=f"Failed to write CSV: {e}"))

    try:
        append_to_index(new_records, new_embs)
    except Exception as e:
        return redirect(url_for("index", msg=f"Failed to save index: {e}"))

    return redirect(url_for("index", msg=f"Indexed {filename} ({len(new_records)} paragraphs)."))


@app.post("/reindex")
def reindex():
    try:
        clear_index_files()
        files_n, paras_n = reindex_all_uploads()
        load_index()
        return redirect(url_for("index", msg=f"Reindexed {files_n} file(s), {paras_n} paragraph(s)."))
    except Exception as e:
        return redirect(url_for("index", msg=f"Reindex failed: {e}"))


@app.post("/remove/<path:filename>")
def remove_file(filename: str):
    # delete from uploads
    upath = safe_join_upload(filename)
    if upath is None:
        return redirect(url_for("index", msg="File not found in uploads/."))

    try:
        os.remove(upath)
    except Exception as e:
        return redirect(url_for("index", msg=f"Failed to delete upload: {e}"))

    # delete per-file CSV
    csv_name = f"{os.path.basename(filename)}.csv"
    ipath = safe_join_index(csv_name)
    if ipath is not None:
        try:
            os.remove(ipath)
        except Exception:
            pass

    # rebuild index from remaining uploads (simple + safe)
    try:
        clear_index_files()
        files_n, paras_n = reindex_all_uploads()
        load_index()
        return redirect(url_for("index", msg=f"Removed {os.path.basename(filename)}. Reindexed {files_n} file(s), {paras_n} paragraph(s)."))
    except Exception as e:
        return redirect(url_for("index", msg=f"Removed file, but reindex failed: {e}"))


@app.get("/download/<path:csv_name>")
def download(csv_name: str):
    return send_from_directory(INDEX_DIR, csv_name, as_attachment=True)


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "")
    k = request.args.get("k", "")
    try:
        k_int = int(k) if k else MAX_RESULTS
    except ValueError:
        k_int = MAX_RESULTS
    return jsonify({"query": q, "results": semantic_search(q, k=k_int)})


@app.get("/api/files")
def api_files():
    return jsonify(
        {
            "files": [{"file": f, "paragraphs": n} for f, n in file_summary()],
            "total_paragraphs": len(records),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
