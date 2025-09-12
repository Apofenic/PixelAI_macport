#!/usr/bin/env bash

# Build script for PixelAI Aseprite Extension
# Creates packaged .aseprite-extension files for distribution

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_ROOT/aseprite_ext_edit/PixelAI"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Version from package.json
VERSION=$(grep '"version"' "$SOURCE_DIR/package.json" | sed 's/.*"version": *"\([^"]*\)".*/\1/')

echo -e "${BLUE}🔨 Building PixelAI Extension v$VERSION${NC}"
echo

# Create build and dist directories
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Clean previous builds
echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
rm -rf "$BUILD_DIR"/*
rm -f "$DIST_DIR"/PixelAI-*.aseprite-extension

# Copy source files to build directory
echo -e "${YELLOW}📋 Copying source files...${NC}"
cp -r "$SOURCE_DIR" "$BUILD_DIR/"

# Remove development files from build
echo -e "${YELLOW}🗑️  Removing development files...${NC}"
find "$BUILD_DIR/PixelAI" -name ".DS_Store" -delete 2>/dev/null || true
find "$BUILD_DIR/PixelAI" -name "*.tmp" -delete 2>/dev/null || true
find "$BUILD_DIR/PixelAI" -name "*.bak" -delete 2>/dev/null || true
rm -rf "$BUILD_DIR/PixelAI/.qodo" 2>/dev/null || true

# Create the extension package
EXTENSION_NAME="PixelAI-v$VERSION-macos.aseprite-extension"
echo -e "${YELLOW}📦 Creating extension package: $EXTENSION_NAME${NC}"

cd "$BUILD_DIR"
zip -r "$DIST_DIR/$EXTENSION_NAME" PixelAI/ -x "*.DS_Store" "*/.*"

# Create a symlink to latest build
cd "$DIST_DIR"
ln -sf "$EXTENSION_NAME" "PixelAI-latest.aseprite-extension"

# Success message
echo
echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${GREEN}📦 Extension package: $DIST_DIR/$EXTENSION_NAME${NC}"
echo -e "${GREEN}🔗 Latest symlink: $DIST_DIR/PixelAI-latest.aseprite-extension${NC}"
echo

# Optional: Show package contents
if [[ "${1:-}" == "--verbose" || "${1:-}" == "-v" ]]; then
    echo -e "${BLUE}📋 Package contents:${NC}"
    unzip -l "$DIST_DIR/$EXTENSION_NAME"
fi

echo -e "${BLUE}💡 To install: Run './scripts/install.sh' or manually install in Aseprite${NC}"
