"""Exercise a fresh private runtime install on the current operating system."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = ROOT / '.venv'
    if runtime.exists():
        raise SystemExit('Installation test requires a fresh checkout without .venv. Existing environments are never modified.')
    subprocess.run([sys.executable, '-I', '-m', 'venv', str(runtime)], check=True)
    python = runtime / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    requirements = 'requirements.txt' if sys.platform == 'darwin' else 'requirements-cpu.txt'
    for args in (['-m', 'pip', 'install', '--upgrade', 'pip'],
                 ['-m', 'pip', 'install', '--no-input', '-r', str(ROOT / requirements)],
                 ['-m', 'pip', 'check']):
        subprocess.run([str(python), '-I', *args], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
