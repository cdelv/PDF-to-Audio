"""Cross-platform installed-runtime checks; full model inference is opt-in."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action='store_true', help='Download only defaults and test short offline CPU inference')
    args = parser.parse_args()
    python = ROOT / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    output = ROOT / 'test-output/ci'
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, QT_QPA_PLATFORM=os.environ.get('QT_QPA_PLATFORM', 'offscreen'),
                       PYTHONUTF8='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
                       HF_HOME=str(output / 'huggingface'), HF_HUB_CACHE=str(output / 'huggingface/hub'),
                       TOKENIZERS_PARALLELISM='false')
    print(f'Testing {platform.platform()} / {platform.machine()}', flush=True)

    def run(name, *command, timeout=600):
        print('Checking:', name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            run_environment = dict(environment)
            if name == 'native-gui':
                run_environment['QT_QPA_PLATFORM'] = 'windows' if sys.platform == 'win32' else 'cocoa' if sys.platform == 'darwin' else 'xcb'
            prefix = ['xvfb-run', '-a'] if name == 'native-gui' and sys.platform == 'linux' and shutil.which('xvfb-run') else []
            process = subprocess.run([*prefix, str(python), *command], cwd=ROOT, env=run_environment,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        print((output / (name + '.log')).read_text(encoding='utf-8', errors='replace'), flush=True)
        process.check_returncode()

    run('isolation', '-I', '-c',
        'import pathlib,site,sys; '
        f'assert pathlib.Path(sys.prefix).resolve() == pathlib.Path({str(ROOT / ".venv")!r}).resolve(); '
        'assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE; '
        'assert site.getusersitepackages() not in sys.path; print(sys.version); print(sys.prefix)')
    run('dependencies', '-I', '-m', 'pip', 'check')
    run('imports', '-I', 'native_worker.py', '--check')
    run('unit', '-m', 'unittest', 'discover', '-s', 'tests', '-v')
    for name in ('appearance_smoke', 'gui_smoke', 'download_gui_smoke', 'pdf_extract_smoke'):
        run(name, 'tests/' + name + '.py')
    run('native-gui', 'app.py', '--gui-smoke')
    if sys.platform == 'linux':
        # Redirect registration to a test directory, never alter the runner's desktop.
        environment.update(XDG_DATA_HOME=str(output / 'data'), XDG_CONFIG_HOME=str(output / 'config'))
        run('desktop-registration', '-I', '-c',
            'import sys; sys.path.insert(0,"."); from install import main; main(); '
            'from core import APP_ID, RUNTIME; import os; from pathlib import Path; '
            'entry=Path(os.environ["XDG_DATA_HOME"])/"applications"/(APP_ID+".desktop"); '
            'assert str(RUNTIME) in entry.read_text(); assert "StartupWMClass="+APP_ID in entry.read_text()')
    if args.models:
        run('default-model-downloads', '-I', 'native_worker.py', '--setup-models', timeout=3600)
        run('offline-inference', '-I', 'native_worker.py', '--self-test', timeout=1800)
    print('Installed runtime checks passed.', flush=True)


if __name__ == '__main__':
    main()
