# PixelAI Development - Quick Reference

# PixelAI Development - Quick Reference

## 🚀 First Time Setup

```bash
./scripts/dev-sync.sh    # Install extension and start Aseprite
./start_server.sh        # Start Python server
```

## 🔄 Daily Development

```bash
# Edit files in: aseprite_ext_edit/PixelAI/
./scripts/dev-sync.sh    # Sync files + restart Aseprite (~3 seconds)
```

## 📦 Building

```bash
./scripts/build.sh       # Build extension package
```

## 📁 Key Directories

- **Edit here**: `aseprite_ext_edit/PixelAI/`
- **Built packages**: `dist/`
- **Scripts**: `scripts/`

## 🔍 Manual Cleanup (if needed)

```bash
# Remove extension from Aseprite
rm -rf "$HOME/Library/Application Support/Aseprite/extensions/PixelAI"

# Clean build artifacts  
rm -rf build/ dist/
```

See `DEVELOPMENT.md` for complete documentation.
