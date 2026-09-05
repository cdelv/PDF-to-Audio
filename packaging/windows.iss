#define AppVersion "0.2.0"
[Setup]
AppId=io.github.pdftoaudio.Desktop
AppName=PDF to Audio
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\PDF to Audio
DefaultGroupName=PDF to Audio
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
WizardStyle=modern
OutputDir=..\dist\installers
OutputBaseFilename=PDF-to-Audio-Windows-x64-Setup
SetupIconFile=..\build\icons\icon.ico
UninstallDisplayIcon={app}\pdf-to-audio.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
[Files]
Source: "..\dist\PDF-to-Audio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{userprograms}\PDF to Audio"; Filename: "{app}\pdf-to-audio.exe"; AppUserModelID: "io.github.pdftoaudio.Desktop"
[Run]
Filename: "{app}\pdf-to-audio.exe"; Description: "Finish setup and download the default models"; Flags: nowait postinstall skipifsilent
