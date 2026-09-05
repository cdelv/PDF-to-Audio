"""One-time offline migration of the app's borrowed packages into its own venv.

Run with .venv/bin/python before removing existing_qwen_runtime.pth. Copies only
the declared dependency closure, using independent copy-on-write files on Btrfs.
No hard links, model downloads, or changes to the source environment.
"""
from collections import deque
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys
import sysconfig

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(sysconfig.get_path('purelib')).resolve()


def main():
    if Path(sys.prefix).resolve() != (ROOT/'.venv').resolve():
        raise RuntimeError('Run this migration with the app private .venv/bin/python.')
    queue = deque(Requirement(line) for line in (ROOT/'requirements.txt').read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith(('#', '--')))
    seen, distributions = set(), {}
    while queue:
        request = queue.popleft()
        name = canonicalize_name(request.name)
        distribution = metadata.distribution(request.name)
        if not request.specifier.contains(distribution.version, prereleases=True):
            raise RuntimeError(f'Installed {name} {distribution.version} does not satisfy {request}.')
        distributions[name] = distribution
        for extra in {'', *request.extras}:
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            for dependency in distribution.requires or []:
                requirement = Requirement(dependency)
                if requirement.marker is None or requirement.marker.evaluate({'extra': extra}):
                    queue.append(requirement)
    sources = set()
    for distribution in distributions.values():
        base = Path(distribution.locate_file('')).resolve()
        if base == TARGET:
            continue
        for entry in distribution.files or []:
            # Console entry points are not used by the app; copy package data,
            # extensions, metadata, and needed .pth files inside site-packages.
            top = entry.parts[0]
            if top not in ('.', '..'):
                source = base/top
                if source.exists():
                    sources.add(source)
    if sources:
        subprocess.run(['cp', '--archive', '--reflink=auto', '--no-clobber', '--',
                        *map(str, sorted(sources)), str(TARGET)], check=True)
    print(f'Copied the dependency closure of {len(distributions)} packages into {TARGET}.')
    print('Remove existing_qwen_runtime.pth, then verify imports and pip check.')


if __name__ == '__main__':
    main()
