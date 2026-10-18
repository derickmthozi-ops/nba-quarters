"""Build the browser versions of the predictor from the page template and model JSON.

Writes web/nba-quarter-predictor.html (Claude artifact) and ../index.html (the home screen app at the repo root).
"""
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
page = (d / "page.template.html").read_text().replace("__MODEL__", (d / "model.json").read_text())
(d / "nba-quarter-predictor.html").write_text(page)

head_extra = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#1b2230">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Quarters">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="icon-192.png" type="image/png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
"""
cut = page.index("<style>")
head = page[:cut].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
rest = page[cut:]
rest = rest.replace("<style>", "<style>\nhtml { padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }\nbody { margin: 0; }\n", 1)
rest = rest.replace("</style>\n", "</style>\n</head>\n<body>\n", 1)
rest = rest.replace("setup();\n</script>", 'setup();\nif ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});\n</script>\n</body>\n</html>')
(d.parent.parent / "index.html").write_text(head_extra + head + rest)
print("built")
