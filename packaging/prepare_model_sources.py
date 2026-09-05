"""Maintainer tool: refresh pinned file metadata, never download model weights."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi, hf_hub_url


def main():
    root = Path(__file__).resolve().parent
    result = {}
    for name, revision in json.loads((root / 'models.json').read_text()).items():
        files = []
        for item in HfApi().model_info(name, revision=revision, files_metadata=True).siblings:
            if not (item.rfilename.endswith(('.json', '.safetensors', '.txt', '.model'))
                    or item.rfilename.startswith('LICENSE')):
                continue
            url = hf_hub_url(name, item.rfilename, revision=revision)
            if item.lfs:
                digest = item.lfs.sha256
            else:
                if item.size > 10_000_000:
                    raise ValueError('Refusing to fetch a large non-LFS file: ' + item.rfilename)
                with urlopen(url, timeout=60) as response:
                    digest = hashlib.sha256(response.read()).hexdigest()
            files.append(dict(path=item.rfilename, size=item.size, sha256=digest, url=url))
        result[name] = dict(revision=revision, files=files)
        print(name, sum(f['size'] for f in files), 'bytes', flush=True)
    (root / 'model-files.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
