#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${1:-$ROOT_DIR/.venv-rocm/bin/python}"
VERSION="${MODEL_STUDIO_VERSION:-1.0.0}"
BIN_NAME="model-studio"
APP_NAME="Model Studio"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found/executable: $PYTHON_BIN"
  echo "Usage: $0 [/path/to/python]"
  exit 1
fi

if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "ERROR: PyInstaller is not installed in this environment."
  echo "Install with: $PYTHON_BIN -m pip install pyinstaller"
  exit 1
fi

APPIMAGETOOL_BIN="${APPIMAGETOOL_BIN:-}"
if [[ -z "$APPIMAGETOOL_BIN" ]]; then
  if command -v appimagetool >/dev/null 2>&1; then
    APPIMAGETOOL_BIN="$(command -v appimagetool)"
  else
    CANDIDATE="$(find "$ROOT_DIR/tools" -maxdepth 1 -type f -name 'appimagetool-*.AppImage' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$CANDIDATE" ]]; then
      APPIMAGETOOL_BIN="$CANDIDATE"
    fi
  fi
fi

if [[ -z "$APPIMAGETOOL_BIN" ]]; then
  echo "ERROR: appimagetool not found."
  echo "Install it or set APPIMAGETOOL_BIN=/path/to/appimagetool"
  exit 1
fi

BUILD_ROOT="$ROOT_DIR/build/appimage"
PYI_DIST="$BUILD_ROOT/pyinstaller-dist"
PYI_WORK="$BUILD_ROOT/pyinstaller-build"
APPDIR="$BUILD_ROOT/AppDir"
DIST_DIR="$ROOT_DIR/dist"

rm -rf "$BUILD_ROOT"
mkdir -p "$PYI_DIST" "$PYI_WORK" "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/scalable/apps" "$DIST_DIR"

ENTRYPOINT="$ROOT_DIR/scripts/python/model_viewer_gui.py"
if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "ERROR: GUI entrypoint not found: $ENTRYPOINT"
  exit 1
fi

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$BIN_NAME" \
  --distpath "$PYI_DIST" \
  --workpath "$PYI_WORK" \
  --specpath "$BUILD_ROOT" \
  "$ENTRYPOINT"

if [[ ! -d "$PYI_DIST/$BIN_NAME" ]]; then
  echo "ERROR: PyInstaller output not found: $PYI_DIST/$BIN_NAME"
  exit 1
fi

cp -a "$PYI_DIST/$BIN_NAME/." "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export APPDIR="$HERE"
exec "$HERE/usr/bin/model-studio" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/model-studio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=YOLO workflow GUI
Exec=$BIN_NAME
Icon=model-studio
Terminal=false
Categories=Development;Graphics;
DESKTOP
cp "$APPDIR/model-studio.desktop" "$APPDIR/usr/share/applications/model-studio.desktop"

cat > "$APPDIR/model-studio.svg" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect width="256" height="256" rx="36" fill="#323232"/>
  <rect x="28" y="28" width="200" height="200" rx="24" fill="#EEEEEE" opacity="0.08"/>
  <circle cx="84" cy="128" r="24" fill="#CF0A0A"/>
  <circle cx="128" cy="96" r="24" fill="#DC5F00"/>
  <circle cx="172" cy="144" r="24" fill="#CF0A0A"/>
</svg>
SVG
cp "$APPDIR/model-studio.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/model-studio.svg"

ARCH_VAL="${ARCH:-$(uname -m)}"
OUTPUT="$DIST_DIR/ModelStudio-${VERSION}-${ARCH_VAL}.AppImage"
rm -f "$OUTPUT"

if [[ "$APPIMAGETOOL_BIN" == *.AppImage ]]; then
  chmod +x "$APPIMAGETOOL_BIN"
  "$APPIMAGETOOL_BIN" --appimage-extract-and-run "$APPDIR" "$OUTPUT"
else
  "$APPIMAGETOOL_BIN" "$APPDIR" "$OUTPUT"
fi

chmod +x "$OUTPUT"
echo "Built AppImage: $OUTPUT"
