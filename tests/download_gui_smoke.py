"""Exercise real download-worker protocol with tiny fixtures, never model weights."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory() as temp:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ['HF_HUB_CACHE'] = temp + '/hub'
        import core
        core.DATA, core.CONFIG = Path(temp) / 'data', Path(temp) / 'settings.json'
        import model_store as store
        from app import App, DesktopApplication
        payload = Path(temp) / 'fixture.json'
        payload.write_bytes(b'{"fixture":true}')
        source = dict(path='config.json', size=payload.stat().st_size,
                      sha256=hashlib.sha256(payload.read_bytes()).hexdigest(), url=payload.as_uri())
        models = {name: dict(revision='fixture', files=[dict(source)]) for name in store.MODELS}
        store.MODELS = models
        application = DesktopApplication([])
        window = App()
        window.show()

        def command():
            code = (f'import sys; sys.path.insert(0, {str(core.ROOT)!r}); import model_store as s; '
                    f's.MODELS={models!r}; s.MODEL_HOME=s.Path({str(store.MODEL_HOME)!r}); '
                    'from worker import main; sys.exit(main())')
            return [sys.executable, '-u', '-c', code]

        def pump(seconds=0.1):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.01)

        def finish():
            deadline = time.monotonic() + 10
            while window.process and time.monotonic() < deadline:
                pump()
            assert window.process is None, window.status.text()

        with patch('app.worker_command', side_effect=command):
            window.setup_models()
            assert window.process and not window.settings_button.isEnabled()
            assert window.setup_panel.isVisible()
            assert window.progress.maximum() == 0, 'Setup must animate before a measurable download starts'
            assert window.cancel_button.text() == 'Cancel setup'
            assert not window.timer_label.isVisible(), 'Setup must not look like audio conversion'
            finish()
            assert window.terminal_event == 'models_ready', window.status.text()
            assert not window.setup_panel.isVisible()
            assert window.progress.maximum() == 1000
            assert not store.missing_models(store.DEFAULT_MODELS)
            assert store.find_model('Qwen/Qwen3-1.7B') is None
            window.settings()
            dialog = window.settings_dialog
            llm = dialog.fields['llm']
            llm.setCurrentIndex(llm.findData('Qwen/Qwen3-1.7B'))
            dialog.save()
            pump()
            finish()
            assert store.find_model('Qwen/Qwen3-1.7B')
            optional = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
            models[optional]['files'][0]['url'] = (Path(temp) / 'missing').as_uri()
            window.setup_models([optional])
            finish()
            assert window.terminal_event == 'fatal'
            assert window.retry_button.isVisible() and window.retry_button.isEnabled()
            assert 'Retry' in window.status.text()
            models[optional]['files'][0]['url'] = payload.as_uri()
            code = 'import sys,time; print("Downloading torch (782 MiB)",file=sys.stderr,flush=True); time.sleep(30)'
            with patch('app.worker_command', return_value=[sys.executable, '-c', code]):
                window.retry_button.click()
                pump()
                window.event(dict(event='download', message='Downloading and installing dependencies…', fraction=None))
                assert window.progress.maximum() == 0
                assert 'Downloading torch' in window.setup_details.toPlainText()
                assert 'Downloading torch' in (core.DATA / 'conversion.log').read_text()
                assert window.cancel_button.isEnabled()
                snapshot = core.ROOT / 'test-output/appearance/setup.png'
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                assert window.grab().save(str(snapshot))
                window.event(dict(event='download', message='Downloading a model…', fraction=0.25))
                assert window.status.text().startswith('25%')
                assert window.progress.maximum() == 1000 and window.progress.value() == 250
                window.cancel_button.click()
                finish()
                assert window.retry_button.isVisible()
                assert window.progress.maximum() == 1000
                assert window.setup_heading.text() == 'Setup cancelled'
            window.retry_button.click()
            finish()
            assert window.terminal_event == 'models_ready'
            assert not window.retry_button.isVisible()
            assert store.find_model(optional)
            window.setup_models()
            assert window.process is None, 'Installed models should not start a download worker'
            with patch('app.runtime_ready', return_value=False), \
                 patch('app.worker_command', return_value=[sys.executable, '-c', 'import time; time.sleep(30)']):
                window.setup_models()
                pump()
                window.close()
                finish()
                assert not window.isVisible(), 'Close must cancel setup and close the window'
        window.close()
        application.quit()
        print('GUI default setup, optional selection, download failure, cancellation, retry, and reuse passed.')


if __name__ == '__main__':
    main()
