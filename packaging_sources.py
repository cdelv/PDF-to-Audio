"""Flatpak metadata uses URLs and hashes, never embeds model payloads."""
from model_store import DEFAULT_MODELS, MODELS
from core import APP_ID


def write_flatpakref(path, url, app_id=APP_ID):
    if not url.startswith(('https://', 'http://127.0.0.1:', 'http://localhost:')) or '\n' in url or '\r' in url:
        raise ValueError('Use an HTTPS repository URL (localhost HTTP is allowed for tests).')
    path.write_text('\n'.join(('[Flatpak Ref]', 'Title=PDF to Audio', 'Name=' + app_id,
                              'Branch=stable', 'Url=' + url.rstrip('/'), 'IsRuntime=false',
                              'RuntimeRepo=https://flathub.org/repo/flathub.flatpakrepo', '')))


def extra_data_args():
    result = []
    for name in DEFAULT_MODELS:
        for index, source in enumerate(MODELS[name]['files']):
            filename = name.replace('/', '--') + '-' + str(index)
            result.append(f"--extra-data={filename}:{source['sha256']}:{source['size']}:{source['size']}:{source['url']}")
    return result
