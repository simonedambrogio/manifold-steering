#!/usr/bin/env bash
# Rebuild the colorpair figures used in this presentation, then deploy them.
#
# Steps:
#   1. Regenerate each figure's HTML + PDF from its checkpoint/data (in colorpair/figures/).
#   2. Copy the HTML into this explorable's assets/, injecting the COLORPAIR_FIT margin shim.
#   3. Bump ASSET_VERSION in script.js so a normal browser refresh picks up the new figures.
#
# Usage (from anywhere):
#   presentation/colorpair/rebuild_figures.sh
#
# Add a figure: append a line to GENERATE below and an entry to MAP in the Python block.

set -euo pipefail

# Resolve paths relative to this script so it runs from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
FIGS="$REPO_ROOT/colorpair/figures"
ASSETS="$SCRIPT_DIR/assets"

cd "$REPO_ROOT"
[ -x "$PY" ] || { echo "error: interpreter not found at $PY" >&2; exit 1; }

echo "==> Regenerating figures (HTML + PDF) in colorpair/figures/"

"$PY" colorpair/figures/fig_pca3d.py \
    --checkpoint artifacts/agent/colorpair_optB.pt \
    --out colorpair/figures/colorpair_pca3d

PYTHONPATH="$REPO_ROOT" "$PY" colorpair/figures/fig_behavior_manifold.py \
    --data artifacts/manifolds/colorpair_behavior.pt \
    --out colorpair/figures/colorpair_behavior_manifold

echo "==> Deploying into assets/ (margin shim) and bumping cache version"
"$PY" - "$FIGS" "$ASSETS" "$SCRIPT_DIR/script.js" <<'PY'
import re, sys, pathlib

figs, assets, script = (pathlib.Path(p) for p in sys.argv[1:4])

# source figure HTML  ->  asset filename referenced by script.js
# (steer1-4.html are deployed by hand from fig_steering_compare.py / fig_steering_multi.py runs)
MAP = {
    "colorpair_pca3d.html":             "pca3d.html",
    "colorpair_behavior_manifold.html": "behavior_manifold.html",
}
SHIM = "<style>/*COLORPAIR_FIT*/html,body{margin:0;overflow:hidden}</style>\n</body>"

for src_name, dst_name in MAP.items():
    t = (figs / src_name).read_text()
    if "COLORPAIR_FIT" not in t:                      # native-size shim: kill default body margin
        t = t.replace("</body>", SHIM, 1)
    (assets / dst_name).write_text(t)
    print(f"  deployed {dst_name}")

# Bump ASSET_VERSION = 'N' -> 'N+1' so browsers refetch the iframes.
s = script.read_text()
m = re.search(r"var ASSET_VERSION = '(\d+)';", s)
if not m:
    raise SystemExit("error: could not find ASSET_VERSION in script.js")
new = int(m.group(1)) + 1
script.write_text(s[:m.start()] + f"var ASSET_VERSION = '{new}';" + s[m.end():])
print(f"  ASSET_VERSION -> {new}")
PY

echo "==> Done. Refresh the presentation to see the updated figures."
