"""Install a tiny probe using the real extra-data hook, without model weights.

Requires the development Flatpak SDK/runtime. Registers only a temporary,
uniquely named test app, then uninstalls it; never touches the real app.
"""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT
from model_store import DEFAULT_MODELS, MODELS
import packaging_sources

APP_ID = 'io.github.pdftoaudio.ModelDownloadTest'


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def main():
    if subprocess.run(['flatpak', 'info', '--user', APP_ID], capture_output=True).returncode == 0:
        raise SystemExit('Probe app already exists; refusing to change it.')
    models = {name: dict(revision=MODELS[name]['revision'],
                        files=[f for f in MODELS[name]['files'] if f['path'] == 'config.json'])
              for name in DEFAULT_MODELS}
    installed = False
    with tempfile.TemporaryDirectory(prefix='pdf-audio-flatpak-') as temp:
        root = Path(temp)
        build = root / 'app'
        run('flatpak', 'build-init', build, APP_ID, 'org.kde.Sdk/x86_64/6.11',
            'org.freedesktop.Platform/x86_64/25.08')
        prefix = build / 'files/share/pdf-to-audio'
        prefix.mkdir(parents=True)
        for name in ('core.py', 'model_store.py'):
            shutil.copy2(ROOT / name, prefix / name)
        shutil.copy2(ROOT / 'packaging/apply_extra.py', prefix / 'apply_extra.py')
        (prefix / 'model-files.json').write_text(json.dumps(models))
        (prefix / '.venv/bin').mkdir(parents=True)
        (prefix / '.venv/bin/python').symlink_to('/usr/bin/python3')
        (build / 'files/bin').mkdir()
        hook = build / 'files/bin/apply_extra'
        shutil.copy2(ROOT / 'packaging/apply_extra', hook)
        hook.chmod(0o755)
        with patch.object(packaging_sources, 'MODELS', models):
            run('flatpak', 'build-finish', '--command=python3', '--metadata=Extra Data=NoRuntime=false',
                *packaging_sources.extra_data_args(), build)
        run('flatpak', 'build-export', root / 'repo', build, 'stable')
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/repo'
        packaging_sources.write_flatpakref(root / 'probe.flatpakref', url, APP_ID)
        run('flatpak', 'remote-add', '--user', '--no-gpg-verify', APP_ID, url)
        try:
            run('flatpak', 'install', '--user', '--noninteractive', '-y', root / 'probe.flatpakref')
            installed = True
            run('flatpak', 'run', '--user', '--unshare=network', '--command=python3', APP_ID, '-c',
                'import sys; sys.path.insert(0,"/app/share/pdf-to-audio"); '
                'from model_store import *; '
                'assert all(local_model(n).startswith("/app/extra/models/") for n in DEFAULT_MODELS); '
                'print("Install-time downloads and offline model discovery passed.")')
        finally:
            if installed:
                run('flatpak', 'uninstall', '--user', '--noninteractive', '-y', APP_ID)
            run('flatpak', 'remote-delete', '--user', APP_ID)
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    main()
