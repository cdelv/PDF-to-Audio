"""Download isolated CPU/CUDA runtimes for native installers, never system Python."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys

from core import DATA, ROOT
from hardware import gpu_memory


def backend(device='auto'):
    if sys.platform == 'darwin':
        return 'cpu'
    return 'cu128' if device.startswith('cuda') or gpu_memory() is not None else 'cpu'


def runtime_paths(kind):
    digest = hashlib.sha256((ROOT / 'requirements-engine.txt').read_bytes()).hexdigest()[:12]
    folder = DATA / 'runtime' / f'{kind}-{digest}'
    return folder, folder / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def ready(device='auto'):
    if not getattr(sys, 'frozen', False):
        return True
    folder, python = runtime_paths(backend(device))
    return python.is_file() and (folder / '.ready').is_file()


def child_environment():
    # Do not let the frozen GUI's shared libraries contaminate external Python.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'UV_', 'PIP_')) and k != 'VIRTUAL_ENV'}
    env.pop('LD_LIBRARY_PATH', None)
    if os.environ.get('LD_LIBRARY_PATH_ORIG'):
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH_ORIG']
    env.update(UV_PYTHON_INSTALL_DIR=str(DATA / 'python'), UV_NO_CONFIG='1',
               UV_NO_CACHE='1', UV_PYTHON_DOWNLOADS='automatic', PYTHONUTF8='1',
               PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1')
    return env


def run_child(command, **kwargs):
    """Forward cancellation to the entire setup/inference subprocess group."""
    with subprocess.Popen([str(arg) for arg in command], env=child_environment(),
                          start_new_session=sys.platform != 'win32', **kwargs) as child:
        def cancel(signum, frame):
            if sys.platform == 'win32':
                child.terminate()
            else:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)  # Finish before the GUI's five-second kill timer.
            except subprocess.TimeoutExpired:
                if sys.platform == 'win32':
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
            raise SystemExit(128 + signum)
        previous = {s: signal.signal(s, cancel) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            child.wait()
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def ensure_runtime(kind, emit):
    from filelock import FileLock
    folder, python = runtime_paths(kind)
    marker = folder / '.ready'
    folder.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder) + '.lock', timeout=1):
        if python.is_file() and marker.is_file():
            return python
        label = 'CPU + Apple Metal' if sys.platform == 'darwin' else 'CUDA + CPU' if kind == 'cu128' else 'CPU'
        uv = ROOT / ('uv.exe' if sys.platform == 'win32' else 'uv')
        if not python.is_file():
            emit('download', message='Downloading and installing Python for the app…', fraction=None)
            run_child([uv, 'venv', '--managed-python', '--python', '3.12', folder], stdout=sys.stderr)
        from runtime_downloads import download_dependencies
        downloads = folder / 'downloads'
        downloads.mkdir(exist_ok=True)
        lock = downloads / 'pylock.toml'
        if not lock.is_file():
            emit('download', message=f'Finding {label} downloads… Calculating the download size.', fraction=None)
            pending = downloads / 'pylock.pending.toml'
            run_child([uv, 'pip', 'compile', '--python', python, '--torch-backend', kind,
                       ROOT / 'requirements-engine.txt', '--format', 'pylock.toml',
                       '-o', pending, '--quiet'], stdout=sys.stderr)
            pending.replace(lock)
        requirements = download_dependencies(lock, emit)
        emit('download', message=f'Downloads complete. Installing {label} dependencies… Small build tools may also download.', fraction=None)
        run_child([uv, 'pip', 'install', '--python', python, '--no-deps',
                   '-r', requirements], stdout=sys.stderr)
        emit('download', message='Downloads installed. Checking the app dependencies…', fraction=None)
        run_child([uv, 'pip', 'check', '--python', python], stdout=sys.stderr)
        verify = "assert torch.version.cuda == '12.8'" if kind == 'cu128' else 'assert torch.version.cuda is None'
        if sys.platform == 'darwin':
            verify += '; assert torch.backends.mps.is_built()'
        run_child([python, '-I', '-c', 'import torch, torchaudio, qwen_tts; ' + verify],
                  stdout=sys.stderr)
        marker.touch()
        shutil.rmtree(downloads)  # Only verified runtime installers, not models or user documents.
        emit('download', message='App dependencies installed. Preparing the selected models…', fraction=None)
    return python


def windows_child_guard():
    """Closing/killing the helper also kills its downloads and inference on Windows."""
    import ctypes
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', w.DWORD), ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t),
                    ('process_limit', w.DWORD), ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
    class Limits(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.SetDllDirectoryW.argtypes = [w.LPCWSTR]
    job = kernel.CreateJobObjectW(None, None)
    limits = Limits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.SetDllDirectoryW(None)
    # Keep the handle open until process exit; Windows then closes it automatically.
    return job


def main():
    if sys.platform == 'win32':
        windows_child_guard()
    request = None if len(sys.argv) > 1 else sys.stdin.readline()
    def emit(event, **data):
        print(json.dumps(dict(event=event, **data)), flush=True)
    try:
        payload = json.loads(request) if request else {}
        if payload.get('ping'):
            emit('models_ready', message='Worker connection verified.')
            return 0
        device = payload.get('config', {}).get('device', 'auto')
        kind = 'cu128' if '--check-cuda' in sys.argv else backend(device)
        python = ensure_runtime(kind, emit)
        command = [python, '-I', '-u', ROOT / 'engine/native_worker.py', *sys.argv[1:]]
        if request:
            # A temporary input file avoids buffered stdin losing the request at handoff.
            import tempfile
            with tempfile.TemporaryFile() as channel:
                channel.write(request.encode('utf-8'))
                channel.seek(0)
                run_child(command, stdin=channel)
        else:
            run_child(command)
        return 0
    except Exception as error:
        emit('fatal', message=f'Setup or conversion failed: {error}. Check your internet connection and free disk space, then Retry.')
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
