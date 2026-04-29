#!/usr/bin/env bash
# Install pre-commit hook that scans staged files for credential leaks.
# PRIVATE / Trade-secret per IMUTAVEL feedback_ip_protection_rule.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
HOOK_DIR="$REPO_ROOT/.git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

mkdir -p "$HOOK_DIR"

cat > "$HOOK_FILE" <<'EOF'
#!/usr/bin/env bash
# Pre-commit: scan staged files for credential leaks.
# Auto-installed by tools/security/install_hooks.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCANNER="$REPO_ROOT/tools/security/scan_secrets.py"

if [ ! -f "$SCANNER" ]; then
    echo "[pre-commit] WARN: scanner missing at $SCANNER -- allowing commit"
    exit 0
fi

# Try py launcher first (Windows), then python3 (Unix), then python.
PYBIN=""
for c in py python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        PYBIN="$c"
        break
    fi
done

if [ -z "$PYBIN" ]; then
    echo "[pre-commit] WARN: no python found -- allowing commit"
    exit 0
fi

# Use 'py -3' if py launcher.
if [ "$PYBIN" = "py" ]; then
    "$PYBIN" -3 "$SCANNER" --staged
else
    "$PYBIN" "$SCANNER" --staged
fi
EOF

chmod +x "$HOOK_FILE" 2>/dev/null || true
echo "[install_hooks] pre-commit installed at $HOOK_FILE"
