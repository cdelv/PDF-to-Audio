import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import core
import model_store as store


class ModelTests(unittest.TestCase):
    def test_default_models_and_pinned_sources(self):
        config = core.defaults()
        self.assertEqual((config['llm'], config['tts']), store.DEFAULT_MODELS)
        self.assertTrue(all('0.6B' in name for name in store.DEFAULT_MODELS))
        self.assertEqual(len(store.MODELS), 4)
        for model in store.MODELS.values():
            self.assertEqual(len(model['revision']), 40)
            self.assertTrue(model['files'])
            for source in model['files']:
                path = Path(source['path'])
                self.assertFalse(path.is_absolute() or '..' in path.parts)
                self.assertIn(model['revision'], source['url'])
                self.assertEqual(len(source['sha256']), 64)
                self.assertGreater(source['size'], 0)
                self.assertTrue(source['url'].startswith('https://huggingface.co/Qwen/'))

    def test_partial_snapshot_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = dict(files=[dict(path='config.json', size=2), dict(path='model.safetensors', size=4)])
            (root / 'config.json').write_text('{}')
            self.assertFalse(store.complete(root, model))
            (root / 'model.safetensors').write_bytes(b'123')
            self.assertFalse(store.complete(root, model))
            (root / 'model.safetensors').write_bytes(b'1234')
            self.assertTrue(store.complete(root, model))

    def test_install_defaults_only_and_reuse_offline(self):
        names = store.DEFAULT_MODELS
        optional = 'Qwen/Qwen3-1.7B'
        payload = b'test model'
        source = dict(path='config.json', size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), url='https://example.invalid/file')
        models = {n: dict(revision='pinned', files=[source]) for n in (*names, optional)}
        with tempfile.TemporaryDirectory() as temp, patch.object(store, 'MODELS', models), \
             patch.object(store, 'MODEL_HOME', Path(temp) / 'models'), \
             patch.dict('os.environ', {'HF_HUB_CACHE': temp + '/cache'}):
            downloaded = []

            def download(item, target, progress):
                downloaded.append(str(target))
                target.write_bytes(payload)
                progress(len(payload))

            with patch.object(store, 'download_file', side_effect=download):
                store.ensure_models(names, lambda *_args, **_kwargs: None)
                self.assertEqual(len(downloaded), 2)
                self.assertEqual(store.missing_models([*names, optional]), [optional])
                store.ensure_models(names, lambda *_args, **_kwargs: None)
                self.assertEqual(len(downloaded), 2)
                store.ensure_models([optional], lambda *_args, **_kwargs: None)
                self.assertEqual(len(downloaded), 3)
                self.assertTrue(store.find_model(optional))
            with self.assertRaisesRegex(ValueError, 'Unknown model'):
                store.ensure_models(['unknown/model'], lambda *_args, **_kwargs: None)

    def test_resume_ignored_range_bad_hash_and_network_failure(self):
        payload = b'weights' * 1024
        ranges = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                ranges.append(self.headers.get('Range'))
                offset = int(self.headers.get('Range', 'bytes=0-')[6:-1]) if self.path == '/resume' else 0
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length', str(len(payload) - offset))
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
                self.end_headers()
                self.wfile.write(payload[offset:])

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), path='model.safetensors')
                for route in ('resume', 'ignore-range'):
                    target = root / route
                    target.with_name(target.name + '.part').write_bytes(payload[:13])
                    source['url'] = f'http://127.0.0.1:{server.server_port}/{route}'
                    store.download_file(source, target, lambda _: None)
                    self.assertEqual(target.read_bytes(), payload)
                    self.assertEqual(ranges[-1], 'bytes=13-')
                bad = dict(source, sha256='0' * 64)
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    store.download_file(bad, root / 'bad', lambda _: None)
                self.assertFalse((root / 'bad').exists())
                partial = root / 'offline.part'
                partial.write_bytes(payload[:20])
                with patch.object(store, 'urlopen', side_effect=OSError('offline')):
                    with self.assertRaisesRegex(RuntimeError, 'Retry'):
                        store.download_file(source, root / 'offline', lambda _: None)
                self.assertEqual(partial.read_bytes(), payload[:20])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__ == '__main__':
    unittest.main()
