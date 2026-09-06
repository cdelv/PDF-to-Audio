import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runtime_downloads as downloads


class RuntimeDownloadTests(unittest.TestCase):
    def test_real_bytes_mid_file_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / 'fixture.whl'
            fixture.write_bytes(b'a' * (3 * 1024 * 1024))
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            lock = root / 'downloads/pylock.toml'
            lock.parent.mkdir()
            lock.write_text(f'''[[packages]]
name = "fixture"
wheels = [{{url = "{fixture.as_uri()}", size = {fixture.stat().st_size}, hashes = {{sha256 = "{digest}"}}}}]
''')
            source = dict(url=fixture.as_uri(), size=fixture.stat().st_size,
                          sha256=digest, path='fixture.whl')
            events = []
            def emit(event, **data):
                events.append(data)
            with patch.object(downloads, 'artifacts', return_value=[source]), \
                 patch.object(downloads.time, 'monotonic', side_effect=range(100)):
                requirements = downloads.download_dependencies(lock, emit)
                self.assertTrue(any(0 < e['fraction'] < 1 for e in events))
                self.assertEqual(events[-1]['fraction'], 1)
                self.assertIn('3.0 / 3.0 MiB', events[-1]['message'])
                self.assertIn(digest, requirements.read_text())
                with patch.object(downloads, 'download_file') as transfer:
                    downloads.download_dependencies(lock, emit)
                    transfer.assert_not_called()
                # A same-size damaged cache must be downloaded again.
                (lock.parent / digest / 'fixture.whl').write_bytes(b'b' * fixture.stat().st_size)
                with patch.object(downloads, 'download_file', wraps=downloads.download_file) as transfer:
                    downloads.download_dependencies(lock, emit)
                    transfer.assert_called_once()

    def test_native_wheel_preferred_and_source_only_supported(self):
        from packaging.tags import Tag
        native = dict(url='https://example.org/soundfile-1-py3-none-manylinux_2_28_x86_64.whl',
                      hashes={'sha256': 'native'})
        pure = dict(url='https://example.org/soundfile-1-py3-none-any.whl', hashes={'sha256': 'pure'})
        source = dict(url='https://example.org/sox-1.tar.gz', size=50, hashes={'sha256': 'source'})
        with patch.object(downloads, 'sys_tags', return_value=[Tag('py3', 'none', 'manylinux_2_28_x86_64'),
                                                            Tag('py3', 'none', 'any')]):
            selected = downloads.artifacts({'packages': [dict(wheels=[pure, native]), dict(sdist=source)]})
        self.assertEqual([s['sha256'] for s in selected], ['native', 'source'])


if __name__ == '__main__':
    unittest.main()
