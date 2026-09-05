"""Arrange Flatpak's install-time, checksum-verified downloads. No network here."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_store import DEFAULT_MODELS, MODELS


def sources():
    for name in DEFAULT_MODELS:
        model = MODELS[name]
        for index, source in enumerate(model['files']):
            filename = name.replace('/', '--') + '-' + str(index)
            relative = Path('models') / name.replace('/', '--') / model['revision'] / source['path']
            yield filename, relative, source


def install(extra=Path('/app/extra')):
    for filename, relative, source in sources():
        path = extra / filename
        target = extra / relative
        if not path.is_file() or path.stat().st_size != source['size']:
            raise ValueError('Missing or incomplete installation download: ' + filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)


if __name__ == '__main__':
    install()
