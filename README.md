# PixelAI_macport

MacOS port of the PixelAI extension for Aseprite - Generate AI art directly within Aseprite using local models.

Original PixelAI for Windows - <https://red335.itch.io/pixelai-local-ai-directly-in-aseprite>

## 🚀 Quick Start

### Prerequisites

- **Node.js** (LTS version) - Required for development scripts
- **Yarn** - Package manager for running development scripts
- **Python 3.x** - For the AI server backend

### First-Time Setup

```bash
# 1. Install dependencies
yarn install

# 2. Install extension for development
./scripts/dev-sync.sh
# OR use yarn:
yarn sync

# 3. Start the Python server
./start_server.sh
# OR use yarn:
yarn start
```

### Daily Development Workflow

```bash
# Edit Lua files in: aseprite_ext_edit/PixelAI/
# Then sync changes (~3 seconds):
./scripts/dev-sync.sh
# OR use yarn:
yarn sync
```

### Building for Distribution

```bash
./scripts/build.sh       # Creates .aseprite-extension package
# OR use yarn:
yarn build
```

## 📁 Project Structure

```text
PixelAI/
├── aseprite_ext_edit/PixelAI/     # Source files (edit here)
│   ├── main.lua                   # Extension entry point
│   ├── local-ui-main.lua          # UI components
│   ├── http-client.lua            # Server communication
│   ├── json.lua & base64.lua      # Utilities
│   └── package.json               # Extension metadata
├── scripts/                       # Development tools
│   ├── dev-sync.sh               # Fast development sync
│   └── build.sh                  # Package builder
├── dist/                         # Built extension packages
├── models/                       # AI models directory
├── loras/                        # LoRA files directory
├── cache/                        # Generated image cache
└── requirements.txt              # Python dependencies
```

## �️ Development Tools

### Available Scripts

- **`./scripts/dev-sync.sh`** - Copies files to Aseprite extensions directory (~3 seconds)
- **`./scripts/build.sh`** - Creates versioned `.aseprite-extension` packages
- **`yarn build`** - Alternative build command via yarn
- **`yarn sync`** - Alternative sync command via yarn

### Development Features

- **Fast iteration** - Quick sync and manual Aseprite restart
- **Clean build system** - Automated packaging with version management
- **Minimal toolset** - Only working, tested components included

## 🎯 Typical Development Session

1. **Initial setup** (one-time):

   ```bash
   ./scripts/dev-sync.sh  # Installs extension to Aseprite
   ```

2. **Start server**:

   ```bash
   ./start_server.sh      # Launches Python AI server
   ```

3. **Development cycle**:
   - Edit Lua files in `aseprite_ext_edit/PixelAI/`
   - Run `./scripts/dev-sync.sh` to sync changes
   - Manually restart Aseprite to see changes
   - Test functionality

4. **Build for distribution**:

   ```bash
   ./scripts/build.sh     # Creates distributable package
   ```

## 🔍 Troubleshooting

### Extension Not Loading

- Check Aseprite's Console (Help > Show Console) for errors
- Verify extension installed: `ls "$HOME/Library/Application Support/Aseprite/extensions/"`
- Re-run dev-sync: `./scripts/dev-sync.sh`

### Server Connection Issues

- Ensure Python server is running: `./start_server.sh`
- Check server logs for error messages
- Verify required Python packages: `pip install -r requirements.txt`

### File Sync Problems

- Check file permissions in source directory
- Manually verify files copied to extensions directory
- Try a fresh sync: `./scripts/dev-sync.sh`

## 🧹 Manual Cleanup

If you need to clean up the development environment:

```bash
# Remove extension from Aseprite
rm -rf "$HOME/Library/Application Support/Aseprite/extensions/PixelAI"

# Clean build artifacts
rm -rf build/ dist/
```

## 📦 Distribution

Built extensions are created in the `dist/` directory:

- `PixelAI-v{version}-macos.aseprite-extension` - Versioned package
- `PixelAI-latest.aseprite-extension` - Symlink to latest build

Share the versioned package for installation on other machines.
