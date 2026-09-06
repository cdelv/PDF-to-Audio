import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import runtime_setup as setup
from hardware import virtual_metal


class RuntimeTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'Windows packaged children use a kill-on-close job object.')
    def test_cancellation_kills_stubborn_child_before_gui_timeout(self):
        child_code = 'import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)'
        code = f'import sys; from runtime_setup import run_child; run_child([sys.executable, "-c", {child_code!r}])'
        with subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True) as process:
            child_pid = int(process.stdout.readline())
            try:
                process.terminate()
                self.assertEqual(process.wait(timeout=5), 143)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass

    def test_virtual_metal_detection(self):
        from types import SimpleNamespace
        for name, expected in (('Apple Paravirtual device', True), ('Apple M3 Pro', False)):
            virtual_metal.cache_clear()
            with patch('hardware.sys.platform', 'darwin'), \
                 patch('hardware.subprocess.run', return_value=SimpleNamespace(stdout=name)):
                self.assertEqual(virtual_metal(), expected)
        virtual_metal.cache_clear()

    def test_backend_selection_and_driver_added_later(self):
        for platform in ('linux', 'win32'):
            with patch.object(setup.sys, 'platform', platform), patch.object(setup, 'gpu_memory', return_value=None):
                self.assertEqual(setup.backend(), 'cpu')
                self.assertEqual(setup.backend('cuda:0'), 'cu128')
            with patch.object(setup.sys, 'platform', platform), patch.object(setup, 'gpu_memory', return_value=4):
                self.assertEqual(setup.backend(), 'cu128')
                self.assertEqual(setup.backend('cpu'), 'cu128')  # CUDA runtime supports CPU too.
        with patch.object(setup.sys, 'platform', 'darwin'), patch.object(setup, 'gpu_memory') as probe:
            self.assertEqual(setup.backend('mps'), 'cpu')  # macOS wheel includes Metal.
            probe.assert_not_called()

    def test_runtime_is_private_reused_and_not_marked_ready_on_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'DATA', Path(temp)), \
             patch.object(setup, 'run_child') as run, \
             patch('runtime_downloads.download_dependencies', return_value=Path(temp) / 'local.txt'):
            folder, python = setup.runtime_paths('cpu')
            python.parent.mkdir(parents=True)
            python.touch()
            run.side_effect = RuntimeError('Interrupted download')
            with self.assertRaises(RuntimeError):
                setup.ensure_runtime('cpu', lambda *_a, **_k: None)
            self.assertFalse((folder / '.ready').exists())
            run.side_effect = None
            (folder / 'downloads/pylock.toml').touch()
            self.assertEqual(setup.ensure_runtime('cpu', lambda *_a, **_k: None), python)
            self.assertTrue((folder / '.ready').is_file())
            run.reset_mock()
            setup.ensure_runtime('cpu', lambda *_a, **_k: None)
            run.assert_not_called()
            self.assertNotEqual(setup.runtime_paths('cu128')[0], folder)

    def test_external_runtime_does_not_inherit_python_or_packager_paths(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/untrusted', 'PYTHONHOME': '/untrusted',
                                    'UV_INDEX_URL': 'https://example.invalid', 'VIRTUAL_ENV': '/elsewhere',
                                    'LD_LIBRARY_PATH': '/frozen', 'LD_LIBRARY_PATH_ORIG': '/driver'}):
            env = setup.child_environment()
            for key in ('PYTHONPATH', 'PYTHONHOME', 'UV_INDEX_URL', 'VIRTUAL_ENV'):
                self.assertNotIn(key, env)
            self.assertEqual(env['LD_LIBRARY_PATH'], '/driver')
            self.assertEqual(env['UV_NO_CONFIG'], '1')


if __name__ == '__main__':
    unittest.main()
