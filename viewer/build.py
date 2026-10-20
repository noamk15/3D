"""Build the static viewer: scene.js (data) + index.html (full page for GitHub Pages).

    python viewer/build.py [experiments/out/fountain_scene.json]
app.html is the page body; index.html wraps it in a standalone HTML document.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "experiments", "out", "fountain_scene.json")
with open(src) as f:
    scene = json.load(f)
with open(os.path.join(HERE, "scene.js"), "w") as f:
    f.write("window.SCENE=" + json.dumps(scene, separators=(",", ":")) + ";\n")
with open(os.path.join(HERE, "app.html")) as f:
    body = f.read()
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
with open(os.path.join(HERE, "index.html"), "w") as f:
    f.write(page)
print("wrote viewer/scene.js and viewer/index.html")
