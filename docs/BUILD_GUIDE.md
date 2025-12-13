# Daemon Desktop Executable Build Guide

## Overview

This guide covers building Daemon as a standalone desktop executable using PyInstaller.

**Build Status**: Implementation complete, ready for first full build.

## Prerequisites

1. **Python 3.11+** (tested with 3.11.8)
2. **PyInstaller 6.0+**: `pip install pyinstaller`
3. **All Daemon dependencies**: `pip install -r requirements.txt`
4. **spaCy model**: `python -m spacy download en_core_web_sm`

## Quick Build

```bash
# Clean previous builds
rm -rf build/ dist/

# Run build (takes 10-20 minutes)
pyinstaller daemon.spec --clean --noconfirm

# Output will be in dist/Daemon/
```

## Build Output

```
dist/Daemon/
├── Daemon              # Main executable (Daemon.exe on Windows)
├── _internal/          # Python runtime and packages
└── assets/             # Icons and splash screen
```

## Files Created for Build

| File | Purpose |
|------|---------|
| `daemon.spec` | PyInstaller configuration (one-dir mode) |
| `utils/bootstrap.py` | Frozen executable bootstrap (paths, migration) |
| `utils/startup.py` | Progress tracking during startup |
| `hooks/hook-*.py` | Custom PyInstaller hooks for complex packages |
| `hooks/runtime_hook.py` | Pre-import environment setup |
| `assets/daemon_icon.ico` | Windows icon |
| `assets/daemon_icon.png` | Linux/general icon |
| `assets/splash.png` | Splash screen image |

## Architecture Decisions

### One-Dir Mode (Not One-File)
- **Reason**: One-file mode extracts to temp on every launch (30s+ delay)
- **Benefit**: Instant startup, easier debugging, smaller AV false positive rate

### External Data Not Bundled
- Wikipedia FAISS index (781MB) and dump (102GB) are NOT bundled
- Features gracefully disabled when external data is missing
- Users can optionally provide external data via `DAEMON_EXTERNAL_DATA` env var

### User Data Directory
When running as frozen executable, data is stored in:
- **Windows**: `%APPDATA%/Daemon/`
- **macOS**: `~/Library/Application Support/Daemon/`
- **Linux**: `~/.daemon/`

This includes: corpus, ChromaDB, user profile, .env, logs

### Migration Support
The bootstrap module automatically migrates data from `./data/` to the user data directory on first frozen run.

## Testing the Executable

After build completes:

```bash
# Linux/macOS
./dist/Daemon/Daemon

# Windows
dist\Daemon\Daemon.exe
```

### Test Checklist

- [ ] App launches (splash appears within 1-2s)
- [ ] GUI appears (within 15-20s)
- [ ] First-run wizard works (if no existing data)
- [ ] Chat functionality works
- [ ] Graceful shutdown (close window, data saved)

## Troubleshooting

### Build Fails with Missing Module
Add to `hiddenimports` in `daemon.spec`:
```python
hiddenimports += ['missing_module']
```

### Build Fails with Missing Data File
Add to `datas` in `daemon.spec`:
```python
datas.append(('path/to/file', 'destination'))
```

### Executable Crashes on Start
1. Build with console enabled for debugging:
   - Edit `daemon.spec`: `console=True`
   - Rebuild and run from terminal to see errors

2. Check for missing imports:
   ```bash
   grep -i "modulenotfound" build.log
   ```

### Large Bundle Size
Current estimate: ~650MB (with torch CPU)

To reduce:
- Exclude unused torch features in spec file
- Use CPU-only torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)

## Creating Installers

### Windows (Inno Setup)
```bash
# After PyInstaller build
iscc installer/daemon_setup.iss
```

### macOS (DMG)
```bash
# After PyInstaller build
./scripts/build_macos_dmg.sh
```

### Linux (AppImage)
```bash
# After PyInstaller build
./scripts/build_linux_appimage.sh
```

## CI/CD

See `.github/workflows/release.yml` for automated builds on all platforms.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Installer size | <400MB | Compressed |
| Cold start to splash | <2s | Native splash |
| Cold start to GUI | <20s | Model loading |
| Memory usage | <2GB | During operation |

## Key Files Reference

- `main.py`: Entry point with freeze_support() and bootstrap integration
- `utils/bootstrap.py`: Path resolution, data migration, environment setup
- `utils/startup.py`: Staged imports with progress feedback
- `daemon.spec`: PyInstaller configuration
- `build/dependency_manifest.py`: Dependency classification

## Development vs Frozen Mode

| Aspect | Development | Frozen |
|--------|-------------|--------|
| Data location | `./data/` | User data dir |
| Config | `config/config.yaml` | Bundled in `_internal/` |
| .env | Project root | User data dir |
| Wiki data | Optional local | External only |
| IS_FROZEN | False | True |
