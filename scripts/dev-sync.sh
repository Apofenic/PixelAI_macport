#!/usr/bin/env bash

# Quick development sync script
# Copies source files to Aseprite and restarts the application

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

# Aseprite extensions directory
EXTENSIONS_DIR="$HOME/Library/Application Support/Aseprite/extensions"
EXTENSION_TARGET="$EXTENSIONS_DIR/PixelAI"

echo -e "${BLUE}⚡ Quick Development Sync${NC}"
echo

# Check if source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}❌ Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Check if extensions directory exists
if [[ ! -d "$EXTENSIONS_DIR" ]]; then
    echo -e "${RED}❌ Aseprite extensions directory not found: $EXTENSIONS_DIR${NC}"
    exit 1
fi

# Stop Aseprite if running
echo -e "${YELLOW}🛑 Checking for running Aseprite...${NC}"
ASEPRITE_PIDS=$(pgrep -f "[Aa]seprite" || true)

if [[ -n "$ASEPRITE_PIDS" ]]; then
    echo -e "${YELLOW}🛑 Stopping Aseprite...${NC}"
    echo "$ASEPRITE_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
    
    # Force kill if still running
    REMAINING_PIDS=$(pgrep -f "[Aa]seprite" || true)
    if [[ -n "$REMAINING_PIDS" ]]; then
        echo -e "${RED}💀 Force killing remaining processes...${NC}"
        echo "$REMAINING_PIDS" | xargs kill -9 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Aseprite stopped${NC}"
else
    echo -e "${BLUE}ℹ️  Aseprite not currently running${NC}"
fi

# Remove existing extension
if [[ -e "$EXTENSION_TARGET" ]]; then
    echo -e "${YELLOW}🗑️  Removing old extension...${NC}"
    rm -rf "$EXTENSION_TARGET"
fi

# Copy source files
echo -e "${YELLOW}📋 Copying source files...${NC}"
cp -r "$SOURCE_DIR" "$EXTENSION_TARGET"

# Remove development artifacts
echo -e "${YELLOW}🧹 Cleaning development artifacts...${NC}"
find "$EXTENSION_TARGET" -name ".DS_Store" -delete 2>/dev/null || true
find "$EXTENSION_TARGET" -name "*.tmp" -delete 2>/dev/null || true
find "$EXTENSION_TARGET" -name "*.bak" -delete 2>/dev/null || true
rm -rf "$EXTENSION_TARGET/.qodo" 2>/dev/null || true
rm -f "$EXTENSION_TARGET/.dev-mode" 2>/dev/null || true

echo -e "${GREEN}✅ Files synced to Aseprite${NC}"

# Start Aseprite
echo -e "${YELLOW}🚀 Starting Aseprite...${NC}"

# Try common Aseprite locations
ASEPRITE_PATHS=(
    "/Applications/Aseprite.app/Contents/MacOS/aseprite"
    "/Applications/Aseprite.app"
    "/usr/local/bin/aseprite"
    "$(which aseprite 2>/dev/null || true)"
)

ASEPRITE_FOUND=false
for path in "${ASEPRITE_PATHS[@]}"; do
    if [[ -n "$path" && -e "$path" ]]; then
        echo -e "${GREEN}✅ Found Aseprite at: $path${NC}"
        
        if [[ "$path" == *.app ]]; then
            open "$path" > /dev/null 2>&1 &
        else
            "$path" > /dev/null 2>&1 &
        fi
        
        ASEPRITE_FOUND=true
        break
    fi
done

if [[ "$ASEPRITE_FOUND" == false ]]; then
    echo -e "${RED}❌ Could not find Aseprite installation${NC}"
    echo -e "${BLUE}💡 Please start Aseprite manually${NC}"
    echo -e "${GREEN}✅ Extension files are ready in Aseprite${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🎉 Development sync complete!${NC}"
echo -e "${GREEN}🚀 Aseprite is starting with your updated extension${NC}"
echo
echo -e "${BLUE}💡 Next time, just run: ./scripts/dev-sync.sh${NC}"
