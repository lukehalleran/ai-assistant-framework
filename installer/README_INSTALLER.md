# Daemon Windows Installer

This directory contains the Inno Setup configuration for creating a professional Windows installer for Daemon.

## Prerequisites

1. **Inno Setup 6.x** - Download from https://jrsoftware.org/issetup.html
2. **PyInstaller output** - The `dist/Daemon/` folder must exist with `Daemon.exe`

## Building the Installer

### Option 1: Using the Build Script (Recommended)

```batch
cd installer
build_installer.bat
```

The script will:
- Check for Inno Setup installation
- Verify PyInstaller output exists
- Build the installer
- Output to `installer/output/DaemonSetup-1.0.0.exe`

### Option 2: Using Inno Setup GUI

1. Open Inno Setup Compiler
2. File > Open > `installer/daemon_installer.iss`
3. Build > Compile (or press Ctrl+F9)

### Option 3: Command Line

```batch
cd installer
"C:\Program Files (x86)\Inno Setup 6\iscc.exe" daemon_installer.iss
```

## Complete Build Process

From the project root:

```batch
:: Step 1: Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install pyinstaller

:: Step 2: Build the PyInstaller executable
pyinstaller daemon.spec --clean --noconfirm

:: Step 3: Build the installer
cd installer
build_installer.bat
```

## Output

The installer will be created at:
```
installer/output/DaemonSetup-1.0.0.exe
```

Size: ~293MB (LZMA2 compressed from ~1.2GB PyInstaller output)

## What the Installer Does

### During Installation
1. Copies Daemon application to `C:\Program Files\Daemon\` (or user-chosen location)
2. Creates desktop shortcut (optional, default: yes)
3. Creates Start Menu shortcuts (optional, default: yes)
4. Registers uninstaller in Windows Add/Remove Programs
5. Optionally launches Daemon after install

### First Run
1. Application launches with splash screen
2. Console window appears (shows startup progress)
3. First-run wizard opens in browser
4. Wizard collects:
   - **Mode**: Personal (streamlined) or Developer (all features)
   - **OpenRouter API key** (required, validated with live API call)
   - **Tavily key** (optional, web search)
   - **Wolfram Alpha App ID** (optional, computational queries)
   - **E2B API key** (optional, dev mode only, code sandbox)
   - **Communication style** (warm/balanced/direct)
   - **Name and pronouns** (optional)
   - **Obsidian vault path** (strongly recommended for daily summaries)
   - **Wikipedia index path** (optional, ~2GB separate download)
   - **Background info** (optional, facts extracted via LLM)
5. Configuration saved to `%APPDATA%\Daemon\.env`
6. User closes browser tab → process shuts down gracefully
7. Relaunch to start chatting

### User Data Locations
| Item | Location |
|------|----------|
| Application | `C:\Program Files\Daemon\` |
| User Config | `%APPDATA%\Daemon\.env` |
| User Profile | `%APPDATA%\Daemon\user_profile.json` |
| Conversation Data | `%APPDATA%\Daemon\corpus_v4.json` |
| ChromaDB | `%APPDATA%\Daemon\chroma_db_v4\` |
| Startup Log | `%APPDATA%\Daemon\daemon_startup.log` |
| Wiki Index | User-specified via `DAEMON_EXTERNAL_DATA` |

### User vs Developer Mode

The wizard asks users to choose a mode:

- **Personal mode**: Chat, memory, web search, computation, daily summaries, personality customization. Synthesis, proposals, debug trace, logs, and architecture docs are disabled.
- **Developer mode**: All features enabled including synthesis pipeline, code proposals, debug trace, logs, memory maintenance UI, and architecture docs in context.

## Uninstallation

The uninstaller (via Add/Remove Programs or Start Menu):
1. Removes all application files from Program Files
2. Prompts user: "Do you want to delete your Daemon user data?"
   - **No**: Preserves `%APPDATA%\Daemon\` (conversations, API key, settings)
   - **Yes**: Deletes all user data

## Distribution

Upload to GitHub Releases:
- `DaemonSetup-1.0.0.exe` — Main installer
- `daemon-wiki-index-v1.zip` — Optional Wikipedia FAISS index (separate, ~2GB)

## Customization

### Changing Version Number
Edit `daemon_installer.iss`:
```pascal
#define MyAppVersion "1.0.0"
```

### Changing Publisher Info
Edit the following in `daemon_installer.iss`:
```pascal
#define MyAppPublisher "Daemon Project"
#define MyAppURL "https://github.com/lukeh/daemon"
```

### Modifying the GUID
**Warning**: Do NOT change the AppId GUID between versions. Windows uses this to identify the application for upgrades.

## Troubleshooting

### "Inno Setup Compiler not found"
- Install Inno Setup 6 from https://jrsoftware.org/issetup.html
- Or add to PATH: `C:\Program Files (x86)\Inno Setup 6\`

### "PyInstaller output not found"
- Run `pyinstaller daemon.spec --clean --noconfirm` from project root first

### OneDrive lock errors during build
- Pause OneDrive sync before building (right-click tray icon → Pause syncing)
- Kill any running Daemon.exe: `taskkill /F /IM Daemon.exe`

### Desktop shortcut not created
- User may have unchecked the option during install
- Can be created manually: right-click Daemon.exe > Send to > Desktop

## Files in This Directory

| File | Purpose |
|------|---------|
| `daemon_installer.iss` | Main Inno Setup script |
| `build_installer.bat` | Automated build script |
| `LICENSE.txt` | License displayed during install |
| `README_INSTALLER.md` | This documentation |
| `output/` | Generated installer output directory |

## Testing Checklist

Before distribution, test on a clean Windows system:

- [ ] Installer launches correctly
- [ ] License agreement displays
- [ ] Custom install path works
- [ ] Desktop shortcut created
- [ ] Start Menu shortcuts created
- [ ] Application launches after install
- [ ] First-run wizard appears with mode selection
- [ ] All wizard steps work (keys, obsidian, wiki index)
- [ ] Chat functionality works after wizard
- [ ] File upload works
- [ ] Closing browser tab shuts down process
- [ ] Relaunch skips wizard
- [ ] Uninstall removes app files
- [ ] Uninstall preserves user data (when "No" selected)
- [ ] Uninstall removes user data (when "Yes" selected)
- [ ] Reinstall/upgrade works correctly
