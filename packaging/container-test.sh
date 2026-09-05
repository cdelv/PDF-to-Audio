#!/bin/sh
set -eu
flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install --user --noninteractive -y /bundle/PDF-to-Audio.flatpak
flatpak info --user io.github.pdftoaudio.Desktop
flatpak run --user --unshare=network --command=/app/share/pdf-to-audio/.venv/bin/python \
  io.github.pdftoaudio.Desktop -I /app/share/pdf-to-audio/verify_install.py --cpu-inference
xvfb-run -a flatpak run --user --unshare=network --socket=x11 --env=QT_QPA_PLATFORM=xcb --command=/app/share/pdf-to-audio/.venv/bin/python \
  io.github.pdftoaudio.Desktop -c 'import sys; sys.path.insert(0,"/app/share/pdf-to-audio"); from app import App, DesktopApplication, QTimer; app=DesktopApplication([]); window=App(); window.show(); QTimer.singleShot(3000, app.quit); raise SystemExit(app.exec())'
printf '%s\n' 'Clean-container installation, offline CPU inference, and GUI launch passed.'
