; Daemon Windows Installer Script
; Inno Setup 6.x
;
; Build with: iscc daemon_installer.iss
; Or run: build_installer.bat

#define MyAppName "Daemon"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Daemon Project"
#define MyAppURL "https://github.com/lukeh/daemon"
#define MyAppExeName "Daemon.exe"

[Setup]
; Unique identifier for the application (do not change between versions)
AppId={{9E159A0D-B1A8-4D39-A149-0004154425E1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE.txt
OutputDir=output
OutputBaseFilename=DaemonSetup-{#MyAppVersion}
SetupIconFile=..\assets\daemon_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Allow install without admin rights (installs to user's Program Files)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; 64-bit only
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startmenuicon"; Description: "Create a &Start Menu shortcut"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Main application executable
Source: "..\dist\Daemon\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; All supporting files from PyInstaller output (recursive)
; This includes _internal/, assets/, and all DLLs
Source: "..\dist\Daemon\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Note: User data is NOT included here - it's created at runtime in %APPDATA%\Daemon\

[Icons]
; Desktop shortcut (created if user selects the task)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

; Start Menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startmenuicon
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

[Run]
; Option to launch application after install completes
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove application directory on uninstall
Type: filesandordirs; Name: "{app}"

[Code]
// Ask user about deleting user data on uninstall
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  UserDataPath: String;
  MsgResult: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    UserDataPath := ExpandConstant('{userappdata}\Daemon');
    if DirExists(UserDataPath) then
    begin
      MsgResult := MsgBox('Do you want to delete your Daemon user data?' + #13#10 +
                          'This includes your conversation history, API keys, and settings.' + #13#10 + #13#10 +
                          'Location: ' + UserDataPath, mbConfirmation, MB_YESNO);
      if MsgResult = IDYES then
      begin
        DelTree(UserDataPath, True, True, True);
      end;
    end;
  end;
end;
