"""Download uv's platform-specific resolution with byte progress and resume."""
import hashlib
import ssl
import time
import tomllib
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import certifi
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from model_store import download_file


def artifacts(lock):
    # uv filters the lock for this platform; prefer its best native wheel over
    # pure-Python alternatives (notably SoundFile, which needs bundled libsndfile).
    ranks = {tag: rank for rank, tag in enumerate(sys_tags())}
    result = []
    for package in lock['packages']:
        candidates = []
        for wheel in package.get('wheels', []):
            name = unquote(urlsplit(wheel['url']).path.rsplit('/', 1)[-1])
            tags = parse_wheel_filename(name)[3]
            rank = min((ranks[tag] for tag in tags if tag in ranks), default=float('inf'))
            if rank != float('inf'):
                candidates.append((rank, wheel))
        source = min(candidates, key=lambda item: item[0])[1] if candidates else package['sdist']
        result.append(dict(url=source['url'], size=source.get('size'),
                           sha256=source['hashes']['sha256'],
                           path=unquote(urlsplit(source['url']).path.rsplit('/', 1)[-1])))
    return result


def download_dependencies(lock_path, emit):
    with lock_path.open('rb') as content:
        sources = artifacts(tomllib.load(content))
    tls = ssl.create_default_context(cafile=certifi.where())
    for source in sources:
        if source['size'] is None:  # PyTorch's index omits wheel sizes.
            emit('download', message=f"Checking download size: {source['path']}…", fraction=None)
            with urlopen(Request(source['url'], method='HEAD', headers={'User-Agent': 'PDF-to-Audio/0.2'}),
                         timeout=30, context=tls) as response:
                source['size'] = int(response.headers['Content-Length'])
    total = sum(source['size'] for source in sources)
    done = 0
    last_update = 0
    paths = []
    for source in sources:
        path = lock_path.parent / source['sha256'] / source['path']

        def progress(count):
            nonlocal last_update
            now = time.monotonic()
            if now - last_update < 0.15 and count != source['size']:
                return
            last_update = now
            fraction = min(1, (done + count) / total)
            emit('download', fraction=fraction,
                 message="Downloading app dependencies · "
                         f"{(done + count) / 2**20:.1f} / {total / 2**20:.1f} MiB · {source['path']}")

        valid = False
        if path.is_file() and path.stat().st_size == source['size']:
            with path.open('rb') as content:
                valid = hashlib.file_digest(content, 'sha256').hexdigest() == source['sha256']
        if not valid:
            download_file(source, path, progress)
        progress(source['size'])
        done += source['size']
        paths.append(path)
    # Local URLs also work with spaces and Windows drive letters. uv still
    # builds source-only packages, but cannot redownload the large wheels.
    requirements = lock_path.parent / 'local-requirements.txt'
    requirements.write_text(''.join(path.resolve().as_uri() + '\n' for path in paths), encoding='utf-8')
    return requirements
