"""Build an offline-ready Flatpak; development tooling, never an end-user step."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APP_ID = 'io.github.pdftoaudio.Desktop'
RUNTIME = '25.08'


def run(*args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=ROOT / 'test-output/flatpak-build')
    parser.add_argument('--skip-runtime-install', action='store_true')
    parser.add_argument('--sdk', default=f'org.freedesktop.Sdk/x86_64/{RUNTIME}')
    parser.add_argument('--reuse-model-cache', type=Path, help='Optional read-only build cache; models are copied into the bundle')
    args = parser.parse_args()
    work = args.workdir.resolve()
    build = work / 'app'
    if not shutil.which('flatpak'):
        raise SystemExit('Build machine needs Flatpak. End users install the resulting .flatpak file.')
    if not args.skip_runtime_install:
        run('flatpak', 'remote-add', '--user', '--if-not-exists', 'flathub', 'https://flathub.org/repo/flathub.flatpakrepo')
        run('flatpak', 'install', '--user', '--noninteractive', '-y', 'flathub',
            f'org.freedesktop.Platform//{RUNTIME}', args.sdk)
    work.mkdir(parents=True, exist_ok=True)
    if not (build / 'metadata').exists():
        run('flatpak', 'build-init', build, APP_ID, args.sdk, f'org.freedesktop.Platform/x86_64/{RUNTIME}')
    app = build / 'files/share/pdf-to-audio'
    app.mkdir(parents=True, exist_ok=True)
    for name in ('app.py', 'core.py', 'worker.py', 'hardware.py', 'languages.py', 'pdf_input.py', 'requirements.txt', 'requirements-cuda.txt'):
        shutil.copy2(ROOT / name, app / name)
    shutil.copytree(ROOT / 'assets', app / 'assets', dirs_exist_ok=True)
    for name in ('models.json', 'download_models.py', 'verify_install.py'):
        shutil.copy2(ROOT / 'packaging' / name, app / name)
    for relative in ('bin', 'share/applications', 'share/metainfo', 'share/icons/hicolor/scalable/apps'):
        (build / 'files' / relative).mkdir(parents=True, exist_ok=True)
    launcher = build / 'files/bin/pdf-to-audio'
    shutil.copy2(ROOT / 'packaging/pdf-to-audio', launcher)
    launcher.chmod(0o755)
    for suffix, directory in (('desktop', 'applications'), ('metainfo.xml', 'metainfo')):
        shutil.copy2(ROOT / f'packaging/{APP_ID}.{suffix}', build / f'files/share/{directory}/{APP_ID}.{suffix}')
    shutil.copy2(ROOT / 'assets/icon.svg', build / f'files/share/icons/hicolor/scalable/apps/{APP_ID}.svg')
    prefix = '/app/share/pdf-to-audio'
    python = prefix + '/.venv/bin/python'
    if not (app / '.venv/bin/python').exists():
        run('flatpak', 'build', build, 'python3', '-m', 'venv', prefix + '/.venv')
    run('flatpak', 'build', '--share=network', build, python, '-m', 'pip', 'install', '--no-cache-dir',
        '-r', prefix + '/requirements-cuda.txt')
    run('flatpak', 'build', build, python, '-m', 'pip', 'check')
    if args.reuse_model_cache:
        cache = args.reuse_model_cache.resolve()
        run('flatpak', 'build', f'--filesystem={cache}:ro', build, python,
            prefix + '/download_models.py', prefix, str(cache))
    else:
        run('flatpak', 'build', '--share=network', build, python, prefix + '/download_models.py', prefix)
    run('flatpak', 'build', build, python, '-I', prefix + '/verify_install.py')
    run('flatpak', 'build-finish', '--command=pdf-to-audio', '--socket=wayland', '--socket=fallback-x11',
        '--share=ipc', '--device=dri', '--filesystem=xdg-documents:ro', '--filesystem=xdg-music',
        '--env=HF_HUB_OFFLINE=1', build)
    run('flatpak', 'build-export', work / 'repo', build, 'stable')
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    run('flatpak', 'build-bundle', '--runtime-repo=https://flathub.org/repo/flathub.flatpakrepo',
        work / 'repo', dist / 'PDF-to-Audio.flatpak', APP_ID, 'stable')
    print('Ready to install:', dist / 'PDF-to-Audio.flatpak')


if __name__ == '__main__':
    main()
