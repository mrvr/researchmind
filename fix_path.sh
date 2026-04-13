#!/usr/bin/env bash
# fix_path.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run this ONCE after activating your venv to permanently add the project root
# to Python's search path. This means you never get "No module named 'utils'"
# regardless of which directory you launch uvicorn from.
#
# Usage:
#   source .venv/bin/activate   (or: source venv/bin/activate)
#   bash fix_path.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_DIR"

# Find the site-packages directory of the active venv
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages: $SITE_PACKAGES"

PTH_FILE="$SITE_PACKAGES/researchmind.pth"
echo "$PROJECT_DIR" > "$PTH_FILE"

echo ""
echo "✓ Created: $PTH_FILE"
echo "  Contents: $PROJECT_DIR"
echo ""
echo "Python will now always find 'utils', 'core', 'processors' from this venv."
echo "You can now run:  uvicorn api:app --host 0.0.0.0 --port 8000"