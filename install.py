#!/usr/bin/python3
"""Register this checkout as a user application; no root privileges needed."""
import os
from pathlib import Path
import shutil
import subprocess

from core import APP_ID, ROOT, RUNTIME, load_settings, save_settings


def desktop_quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%') + '"'


def main():
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    applications = data / "applications"
    icons = data / "icons/hicolor/scalable/apps"
    applications.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    icon_name = APP_ID
    icon_path = icons / (icon_name + ".svg")
    shutil.copy2(ROOT / "assets/icon.svg", icon_path)
    content = "\n".join([
        "[Desktop Entry]", "Type=Application", "Version=1.0", "Name=PDF to Audio",
        "Comment=Turn documents into speech with a local Qwen voice",
        f"Exec={desktop_quote(RUNTIME)} {desktop_quote(ROOT / 'app.py')} %F",
        # COSMIC can miss named icons in the user's hicolor directory. A direct
        # path bypasses that lookup; Qt windows load the same bundled SVG.
        "Icon=" + str(icon_path.resolve()), "Terminal=false", "StartupNotify=true",
        "Categories=AudioVideo;Audio;", "Keywords=PDF;TTS;Speech;Audiobook;Qwen;",
        "MimeType=application/pdf;text/plain;text/markdown;text/x-rst;text/csv;",
        "StartupWMClass=" + icon_name, "",
    ])
    for path in (applications / (icon_name + ".desktop"), ROOT / "PDF to Audio.desktop"):
        path.write_text(content)
        path.chmod(0o755)
        if shutil.which("gio"):
            subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"], capture_output=True)
    config = load_settings()
    save_settings(config)
    for command in (["update-desktop-database", str(applications)],
                    ["gtk-update-icon-cache", "-f", "-t", str(data / "icons/hicolor")]):
        if shutil.which(command[0]):
            subprocess.run(command, check=False, capture_output=True)
    print("Installed PDF to Audio in your application menu.")
    print("You can also double-click:", ROOT / "PDF to Audio.desktop")


if __name__ == "__main__":
    main()
