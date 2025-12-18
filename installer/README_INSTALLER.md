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
:: Step 1: Build the PyInstaller executable
pyinstaller daemon.spec --clean --noconfirm

:: Step 2: Build the installer
cd installer
build_installer.bat
```

## Output

The installer will be created at:
```
installer/output/DaemonSetup-1.0.0.exe
```

Expected size: ~650MB (compressed from ~1.5GB PyInstaller output)

## What the Installer Does

### During Installation
1. Copies Daemon application to `C:\Program Files\Daemon\` (or user-chosen location)
2. Creates desktop shortcut (optional, default: yes)
3. Creates Start Menu shortcuts (optional, default: yes)
4. Registers uninstaller in Windows Add/Remove Programs
5. Optionally launches Daemon after install

### First Run
1. Application launches and displays splash screen
2. First-run wizard appears (if no existing configuration)
3. User enters OpenAI API key
4. Configuration saved to `%APPDATA%\Daemon\.env`
5. Browser opens to Gradio interface at http://localhost:7860

### User Data Locations
| Item | Location |
|------|----------|
| Application | `C:\Program Files\Daemon\` |
| User Config | `%APPDATA%\Daemon\.env` |
| User Profile | `%APPDATA%\Daemon\user_profile.json` |
| Conversation Data | `%APPDATA%\Daemon\data\` |
| ChromaDB | `%APPDATA%\Daemon\chroma\` |

## Uninstallation

The uninstaller (via Add/Remove Programs or Start Menu):
1. Removes all application files from Program Files
2. Prompts user: "Do you want to delete your Daemon user data?"
   - **No**: Preserves `%APPDATA%\Daemon\` (conversations, API key, settings)
   - **Yes**: Deletes all user data

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

### Adding Files
To include additional files, edit the `[Files]` section:
```pascal
Source: "..\path\to\file"; DestDir: "{app}\subfolder"; Flags: ignoreversion
```

## Troubleshooting

### "Inno Setup Compiler not found"
- Install Inno Setup 6 from https://jrsoftware.org/issetup.html
- Or add to PATH: `C:\Program Files (x86)\Inno Setup 6\`

### "PyInstaller output not found"
- Run `pyinstaller daemon.spec --clean --noconfirm` from project root first

### Installer too large
- The ~650MB size is expected due to PyTorch, transformers, and ML dependencies
- LZMA2 compression is already enabled (best compression ratio)

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
- [ ] First-run wizard appears
- [ ] API key entry works
- [ ] Chat functionality works
- [ ] Uninstall removes app files
- [ ] Uninstall preserves user data (when "No" selected)
- [ ] Uninstall removes user data (when "Yes" selected)
- [ ] Reinstall works correctly
