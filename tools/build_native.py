"""Create native CPU preview installers; invoke using the private build runtime."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    if sys.platform in ('darwin', 'win32'):
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtCore import Qt
        from PIL import Image
        folder = ROOT / 'build/icons'
        folder.mkdir(parents=True, exist_ok=True)
        image = QImage(1024, 1024, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        QSvgRenderer(str(ROOT / 'assets/icon.svg')).render(painter)
        painter.end()
        image.save(str(folder / 'icon.png'))
        with Image.open(folder / 'icon.png') as bitmap:
            bitmap.save(folder / ('icon.icns' if sys.platform == 'darwin' else 'icon.ico'))
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', 'packaging/desktop.spec')
    if args.freeze_only:
        return
    installers = ROOT / 'dist/installers'
    installers.mkdir(exist_ok=True)
    if sys.platform == 'win32':
        compiler = shutil.which('ISCC') or r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
        run(compiler, 'packaging/windows.iss')
    elif sys.platform == 'darwin':
        stage = ROOT / 'build/dmg'
        stage.mkdir(exist_ok=True)
        run('ditto', ROOT / 'dist/PDF to Audio.app', stage / 'PDF to Audio.app')
        if not (stage / 'Applications').exists():
            (stage / 'Applications').symlink_to('/Applications')
        run('hdiutil', 'create', '-volname', 'PDF to Audio', '-srcfolder', stage,
            '-ov', '-format', 'UDZO', installers / f'PDF-to-Audio-macOS-{platform.machine()}.dmg')
    else:
        package = ROOT / 'build/deb'
        target = package / 'opt/pdf-to-audio'
        shutil.copytree(ROOT / 'dist/PDF-to-Audio', target, dirs_exist_ok=True)
        for directory in ('DEBIAN', 'usr/share/applications', 'usr/share/icons/hicolor/scalable/apps'):
            (package / directory).mkdir(parents=True, exist_ok=True)
        (package / 'DEBIAN/control').write_text('Package: pdf-to-audio\nVersion: 0.2.0\nArchitecture: amd64\nMaintainer: cdelv\nSection: sound\nPriority: optional\nDepends: libegl1, libopengl0, libdbus-1-3, libxkbcommon-x11-0, libxcb-cursor0\nDescription: Local document narration with voice cloning\n')
        (package / 'usr/share/applications/io.github.pdftoaudio.Desktop.desktop').write_text(
            '[Desktop Entry]\nType=Application\nName=PDF to Audio\nExec=/opt/pdf-to-audio/pdf-to-audio %F\nIcon=io.github.pdftoaudio.Desktop\nTerminal=false\nCategories=AudioVideo;Audio;\nStartupWMClass=io.github.pdftoaudio.Desktop\n')
        shutil.copy2(ROOT / 'assets/icon.svg', package / 'usr/share/icons/hicolor/scalable/apps/io.github.pdftoaudio.Desktop.svg')
        run('dpkg-deb', '--build', '--root-owner-group', package, installers / 'PDF-to-Audio-Linux-amd64.deb')
        rpm = ROOT / 'build/rpm'
        rpm.mkdir(exist_ok=True)
        spec = rpm / 'package.spec'
        spec.write_text(f'''Name: pdf-to-audio
Version: 0.2.0
Release: 1
Summary: Local document narration with voice cloning
License: LicenseRef-proprietary
BuildArch: x86_64
AutoReqProv: no
Requires: glibc >= 2.39, libglvnd-egl, libglvnd-opengl, libX11, libxcb, xcb-util-cursor, libxkbcommon-x11, dbus-libs
%description
Local document narration with downloadable Qwen models.
%install
mkdir -p "%{{buildroot}}/opt" "%{{buildroot}}/usr/share"
cp -a "{package / 'opt/pdf-to-audio'}" "%{{buildroot}}/opt/"
cp -a "{package / 'usr/share/applications'}" "%{{buildroot}}/usr/share/"
cp -a "{package / 'usr/share/icons'}" "%{{buildroot}}/usr/share/"
%files
/opt/pdf-to-audio
/usr/share/applications/io.github.pdftoaudio.Desktop.desktop
/usr/share/icons/hicolor/scalable/apps/io.github.pdftoaudio.Desktop.svg
''')
        run('rpmbuild', '-bb', '--define', f'_topdir {rpm}', '--define', '__os_install_post %{nil}', str(spec))
        for artifact in (rpm / 'RPMS/x86_64').glob('*.rpm'):
            shutil.copy2(artifact, installers / 'PDF-to-Audio-Linux-x86_64.rpm')


if __name__ == '__main__':
    main()
