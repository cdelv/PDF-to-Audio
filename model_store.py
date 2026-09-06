"""Pinned model downloads, outside the app bundle; no inference imports needed."""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from core import DATA, ROOT

DEFAULT_MODELS = ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base')
manifest_path = ROOT / 'model-files.json'
if not manifest_path.exists():
    manifest_path = ROOT / 'packaging/model-files.json'
MODELS = json.loads(manifest_path.read_text())
MODEL_HOME = DATA / 'models'


def complete(path, model, legacy=False):
    try:
        return all((path / f['path']).is_file() and (path / f['path']).stat().st_size == f['size']
                   for f in model['files'] if not (legacy and f['path'].startswith('LICENSE')))
    except OSError:
        return False


def find_model(name):
    if Path(name).expanduser().is_dir():
        return str(Path(name).expanduser().resolve())
    if name not in MODELS:
        return None
    model = MODELS[name]
    relative = Path(name.replace('/', '--')) / model['revision']
    hf_home = Path(os.environ.get('HF_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface'))
    cache = Path(os.environ.get('HF_HUB_CACHE', hf_home / 'hub'))
    path = MODEL_HOME / relative
    if complete(path, model):
        return str(path)
    path = cache / ('models--' + name.replace('/', '--')) / 'snapshots' / model['revision']
    if complete(path, model, legacy=True):
        return str(path)
    return None


def local_model(name):
    path = find_model(name)
    if path:
        return path
    raise ValueError(f'{name} is not installed. Connect to the internet and retry model setup in the app.')


def missing_models(names):
    return [name for name in dict.fromkeys(names) if not find_model(name)]


def download_file(source, target, progress):
    """Resume interrupted transfers and publish only checksum-verified files."""
    import ssl
    import certifi
    tls = ssl.create_default_context(cafile=certifi.where())
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    error = None
    for _ in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset > source['size']:
            with part.open('wb'):
                pass
            offset = 0
        try:
            if offset < source['size']:
                request = Request(source['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
                with urlopen(request, timeout=30, context=tls) as response:
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid download resume response')
                    count = offset if resumed else 0
                    with part.open('ab' if resumed else 'wb') as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                            count += len(block)
                            progress(count)
                            if count > source['size']:
                                raise ValueError('Download exceeds expected size')
            if part.stat().st_size != source['size']:
                raise ValueError('Incomplete download')
            digest = hashlib.sha256()
            with part.open('rb') as content:
                while block := content.read(4 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != source['sha256']:
                with part.open('wb'):
                    pass
                raise ValueError('Download checksum mismatch')
            part.replace(target)
            progress(source['size'])
            return
        except OSError as exc:
            error = exc
        except ValueError as exc:
            error = exc
    raise RuntimeError(f"Could not download {source['path']}: {error}. Check your connection and free disk space, then Retry. Partial downloads are kept.")


def ensure_models(names, emit):
    from filelock import FileLock
    for name in dict.fromkeys(names):
        if find_model(name):
            continue
        if name not in MODELS:
            raise ValueError(f'Unknown model: {name}. Choose a listed Qwen model or an existing local model folder.')
        model = MODELS[name]
        target = MODEL_HOME / name.replace('/', '--') / model['revision']
        target.mkdir(parents=True, exist_ok=True)
        emit('download', message=f'Preparing {name}…', fraction=0)
        with FileLock(str(target / '.download.lock'), timeout=1):
            if find_model(name):
                continue
            total = sum(f['size'] for f in model['files'])
            done = 0
            last_update = 0
            for source in model['files']:
                path = target / source['path']

                def progress(count):
                    nonlocal last_update
                    now = time.monotonic()
                    if now - last_update < 0.15 and count != source['size']:
                        return
                    last_update = now
                    emit('download', fraction=min(1, (done + count) / total),
                         message=f"Downloading {name} · {(done + count) / 2**30:.2f} / {total / 2**30:.2f} GiB · {source['path']}")

                if not path.is_file() or path.stat().st_size != source['size']:
                    download_file(source, path, progress)
                done += source['size']
            if not complete(target, model):
                raise RuntimeError('Model installation is incomplete. Retry setup.')
            emit('download', message=f'{name} is ready.', fraction=1)
