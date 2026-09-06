"""Install/unpack release artifacts and exercise the packaged executables."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    if sys.platform == 'win32':
        target = ROOT / 'test-output/installed-windows'
        run(ROOT / 'dist/installers/PDF-to-Audio-Windows-x64-Setup.exe', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', f'/DIR={target}')
    elif sys.platform == 'darwin':
        mount = ROOT / 'test-output/dmg-mount'
        mount.parent.mkdir(parents=True, exist_ok=True)
        image = next((ROOT / 'dist/installers').glob('*.dmg'))
        run('hdiutil', 'attach', '-nobrowse', '-mountpoint', mount, image)
        try:
            app = ROOT / 'test-output/installed-macos/PDF to Audio.app'
            run('ditto', mount / 'PDF to Audio.app', app)
        finally:
            run('hdiutil', 'detach', mount)
        target = app / 'Contents/MacOS'
    else:
        run('sudo', 'apt-get', 'install', '-y', str(ROOT / 'dist/installers/PDF-to-Audio-Linux-amd64.deb'))
        run('rpm', '-qpl', ROOT / 'dist/installers/PDF-to-Audio-Linux-x86_64.rpm')
        target = Path('/opt/pdf-to-audio')
    helper = target / ('pdf-to-audio-worker.exe' if sys.platform == 'win32' else 'pdf-to-audio-worker')
    gui = target / ('pdf-to-audio.exe' if sys.platform == 'win32' else 'pdf-to-audio')
    platform = 'windows' if sys.platform == 'win32' else 'cocoa' if sys.platform == 'darwin' else 'xcb'
    env = dict(os.environ, HF_HUB_OFFLINE='1', QT_QPA_PLATFORM=platform)
    run(helper, '--check', env=env)
    prefix = ['xvfb-run', '-a'] if sys.platform.startswith('linux') else []
    run(*prefix, gui, '--gui-smoke', env=env, timeout=60)
    run(helper, '--setup-models', env=env, timeout=3600)
    run(helper, '--self-test', env=env, timeout=1800)
    if sys.platform != 'darwin':
        # Hosted runners have no NVIDIA GPU: verify the actual CUDA runtime installs
        # and imports without a driver. Hardware inference is checked locally.
        run(helper, '--check-cuda', env=env, timeout=3600)
    else:
        run(helper, '--check-metal', env=env, timeout=1800)


if __name__ == '__main__':
    main()
