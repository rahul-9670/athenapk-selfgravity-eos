#!/usr/bin/env python
"""Render ../EOS_VALIDATION.md to a single self-contained HTML file (figures embedded as base64),
../EOS_VALIDATION.html, readable without the folder."""
import base64, os, re
import markdown
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = open(os.path.join(ROOT, "EOS_VALIDATION.md")).read()


def embed(m):
    alt, path = m.group(1), m.group(2)
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return m.group(0)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"/><figcaption>{path}</figcaption></figure>'


src = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, src)
body = markdown.markdown(src, extensions=["tables", "fenced_code"])
css = """body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;
line-height:1.45;color:#1a1a19;background:#fff}table{border-collapse:collapse;margin:1em 0;font-size:.9em}
td,th{border:1px solid #d8d7d0;padding:3px 7px;vertical-align:top}th{background:#f3f2ec}code{background:#f3f2ec;
padding:0 3px}figure{margin:1.2em 0}img{max-width:100%}figcaption{font-size:.8em;color:#6b6a63}h2{border-bottom:
1px solid #d8d7d0;padding-bottom:.2em;margin-top:1.8em}"""
html = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>AthenaPK tabulated-EOS validation</title><style>{css}</style></head><body>{body}</body></html>"
out = os.path.join(ROOT, "EOS_VALIDATION.html")
open(out, "w").write(html)
print("wrote", out, os.path.getsize(out) // 1024, "kB")
