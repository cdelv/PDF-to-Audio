#!/bin/sh
set -eu
flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
# Serve the model-free repository locally, emulating an HTTPS release host.
python3 -m http.server 8765 --bind 127.0.0.1 --directory /bundle >/tmp/repository.log 2>&1 &
repository_pid=$!
trap 'kill "$repository_pid"' EXIT
python3 -c 'import time, urllib.request
for attempt in range(100):
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/repo/summary", timeout=1).close()
        break
    except OSError:
        time.sleep(0.1)
else:
    raise SystemExit("Test repository did not start")'
flatpak remote-add --user --no-gpg-verify pdf-to-audio-test http://127.0.0.1:8765/repo
flatpak install --user --noninteractive -y pdf-to-audio-test io.github.pdftoaudio.Desktop
flatpak info --user io.github.pdftoaudio.Desktop
flatpak run --user --unshare=network --command=/app/share/pdf-to-audio/.venv/bin/python \
  io.github.pdftoaudio.Desktop -I /app/share/pdf-to-audio/verify_install.py --defaults-only --cpu-inference
xvfb-run -a flatpak run --user --unshare=network --socket=x11 --env=QT_QPA_PLATFORM=xcb --command=/app/share/pdf-to-audio/.venv/bin/python \
  io.github.pdftoaudio.Desktop -c 'import sys; sys.path.insert(0,"/app/share/pdf-to-audio"); from app import App, DesktopApplication, QTimer; app=DesktopApplication([]); window=App(); window.show(); QTimer.singleShot(3000, app.quit); raise SystemExit(app.exec())'
printf '%s\n' 'Clean-container installation, offline CPU inference, and GUI launch passed.'
