"""Build-time download only. Released applications run offline."""
import json
from pathlib import Path
import sys
import subprocess

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
revisions = json.loads((root / 'models.json').read_text())
for name, revision in revisions.items():
    print(f'Downloading {name} at {revision}', flush=True)
    if len(sys.argv) > 2:
        source = snapshot_download(name, revision=revision, cache_dir=sys.argv[2], local_files_only=True)
        target = root / 'models' / name.replace('/', '--')
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-aL', '--reflink=auto', source + '/.', str(target)], check=True)
        continue
    snapshot_download(name, revision=revision,
                      local_dir=root / 'models' / name.replace('/', '--'),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*'],
                      max_workers=4)
