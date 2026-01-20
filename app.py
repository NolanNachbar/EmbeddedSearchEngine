from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Tiny in-memory "index" to search
DATA = [
    "Flask",
    "FastAPI",
    "Django",
    "Bottle",
    "Pyramid",
    "Sanic",
    "Tornado",
]

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Search</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem; }
      input { padding: .5rem; width: min(520px, 95%); }
      .hint { color: #666; margin-top: .5rem; }
      ul { padding-left: 1.25rem; }
      code { background: #f5f5f5; padding: .1rem .25rem; border-radius: .25rem; }
    </style>
  </head>
  <body>
    <h1>Search</h1>
    <form method="get" action="/">
      <input name="q" placeholder="Type to search..." value="{q}" />
      <button type="submit">Search</button>
    </form>
    <div class="hint">Try: <code>flask</code>, <code>api</code>, <code>dj</code></div>
    <p class="hint">JSON endpoint: <code>/api/search?q=flask</code></p>
    {results}
  </body>
</html>
"""


def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def search(query: str) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    return [item for item in DATA if q in item.lower()]


@app.get("/")
def index():
    q = request.args.get("q", "")
    matches = search(q)

    if q.strip() and not matches:
        results_html = "<p>No results.</p>"
    elif matches:
        items = "".join(f"<li>{escape_html(m)}</li>" for m in matches)
        results_html = f"<p>Results ({len(matches)}):</p><ul>{items}</ul>"
    else:
        results_html = "<p>Enter a query above.</p>"

    return HTML.format(q=escape_html(q), results=results_html)


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify({"query": q, "results": search(q)})


if __name__ == "__main__":
    # Codespaces: bind to 0.0.0.0 so the port can be forwarded
    # Prefer Codespaces/hosting-provided PORT if available.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
