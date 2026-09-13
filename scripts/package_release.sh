#!/usr/bin/env bash
# Wiggle-3D Local Release Packaging Script
# Usage: ./scripts/package_release.sh [TARGET_TRIPLE]

set -euo pipefail

# Standardized ANSI Logging
NC='\033[0m'

log_info() {
    local green='\033[0;32m'
    printf "%b[INFO]%b %s\n" "$green" "$NC" "$1"
}

log_warn() {
    local yellow='\033[1;33m'
    printf "%b[WARN]%b %s\n" "$yellow" "$NC" "$1" >&2
}

log_error() {
    local red='\033[0;31m'
    printf "%b[ERROR]%b %s\n" "$red" "$NC" "$1" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$ROOT_DIR"

# 1. Resolve Version from Cargo.toml
VERSION=$(grep -m 1 '^version = ' Cargo.toml | sed -E 's/version = "([^"]+)"/\1/')
[ -n "$VERSION" ] || { log_error "Could not extract version from Cargo.toml"; exit 1; }

# 2. Resolve Target Triple
if [ -n "${1:-}" ]; then
    TARGET_TRIPLE="$1"
else
    # Auto-detect host target
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    case "$OS" in
        Linux)
            OS_TAG="unknown-linux-gnu"
            ;;
        Darwin)
            OS_TAG="apple-darwin"
            ;;
        *)
            log_error "Unsupported host OS: $OS"
            exit 1
            ;;
    esac
    case "$ARCH" in
        x86_64|amd64)
            ARCH_TAG="x86_64"
            ;;
        aarch64|arm64)
            ARCH_TAG="aarch64"
            ;;
        *)
            log_error "Unsupported host ARCH: $ARCH"
            exit 1
            ;;
    esac
    TARGET_TRIPLE="${ARCH_TAG}-${OS_TAG}"
fi

printf "============================================================\n"
printf " Packaging Wiggle-3D v%s for %s\n" "$VERSION" "$TARGET_TRIPLE"
printf "============================================================\n"

# 3. Build Release Binary
log_info "Building release binary..."
if [ "${TARGET_TRIPLE}" = "$(rustc -vV | grep 'host:' | cut -d' ' -f2)" ]; then
    cargo build --release --locked --bin reto-cli
    BIN_PATH="target/release/reto-cli"
else
    cargo build --release --locked --target "${TARGET_TRIPLE}" --bin reto-cli
    BIN_PATH="target/${TARGET_TRIPLE}/release/reto-cli"
fi

[ -f "$BIN_PATH" ] || { log_error "Binary not found at $BIN_PATH"; exit 1; }

# 4. Prepare Dist Staging Directory
DIST_DIR="dist"
STAGE_DIR="${DIST_DIR}/wiggle-3d-v${VERSION}-${TARGET_TRIPLE}"
ARCHIVE_NAME="wiggle-3d-v${VERSION}-${TARGET_TRIPLE}.tar.gz"

cleanup_stage() {
    rm -rf "$STAGE_DIR"
}
trap cleanup_stage EXIT INT TERM

mkdir -p "$DIST_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# 5. Copy Release Assets (lean packaging - no models or ONNX dylibs bundled)
log_info "Staging release files..."
cp "$BIN_PATH" "${STAGE_DIR}/reto-cli"
chmod +x "${STAGE_DIR}/reto-cli"

if [ -f "LICENSE" ]; then
    cp "LICENSE" "${STAGE_DIR}/"
fi
if [ -f "README.md" ]; then
    cp "README.md" "${STAGE_DIR}/"
fi

# 6. Create Tarball and Checksum
log_info "Creating archive ${DIST_DIR}/${ARCHIVE_NAME}..."
(
    cd "$DIST_DIR"
    tar -czf "${ARCHIVE_NAME}" -C "$(basename "$STAGE_DIR")" .
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
    fi
)

rm -rf "$STAGE_DIR"

ARCHIVE_PATH="${DIST_DIR}/${ARCHIVE_NAME}"
CHECKSUM_PATH="${DIST_DIR}/${ARCHIVE_NAME}.sha256"

printf "============================================================\n"
printf " Release artifact packaged successfully!\n"
printf " - Archive  : %s (%s)\n" "${ARCHIVE_PATH}" "$(du -h "${ARCHIVE_PATH}" | cut -f1)"
printf " - Checksum : %s\n\n" "${CHECKSUM_PATH}"
printf " To publish to GitHub Releases, run:\n"
printf "   gh release create v%s %s %s --title \"v%s\" --notes \"Release notes...\"\n" "$VERSION" "${ARCHIVE_PATH}" "${CHECKSUM_PATH}" "$VERSION"
printf "============================================================\n"
