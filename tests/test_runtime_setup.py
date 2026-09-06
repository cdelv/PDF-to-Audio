import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime_setup as setup


class RuntimeTests(unittest.TestCase):
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
             patch.object(setup, 'run_child') as run:
            folder, python = setup.runtime_paths('cpu')
            python.parent.mkdir(parents=True)
            python.touch()
            run.side_effect = RuntimeError('Interrupted download')
            with self.assertRaises(RuntimeError):
                setup.ensure_runtime('cpu', lambda *_a, **_k: None)
            self.assertFalse((folder / '.ready').exists())
            run.side_effect = None
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
