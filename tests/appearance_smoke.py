"""Check the shared Qt interface, both palettes, and independent settings."""
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory() as directory:
        os.environ['XDG_DATA_HOME'] = directory + '/data'
        os.environ['XDG_CONFIG_HOME'] = directory + '/config'
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import core
        core.DATA = Path(directory) / 'data'
        core.CONFIG = Path(directory) / 'config/settings.json'
        from app import App, DesktopApplication, QPalette, QColor, Qt, ROOT, combo_value
        from core import load_settings
        application = DesktopApplication([])
        window = App()
        window.show()

        def pump(seconds=0.2):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.01)

        with patch('app.gpu_memory', return_value=4.0):
            window.settings()
            pump()
        dialog = window.settings_dialog
        dialog.fields['llm'].setCurrentIndex(dialog.fields['llm'].findData('Qwen/Qwen3-1.7B'))
        assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ['Voice', 'Languages', 'Cleanup prompt', 'Models']
        output = ROOT / 'test-output/appearance'
        output.mkdir(parents=True, exist_ok=True)
        for name, dark in [('light', False), ('dark', True)]:
            palette = application.style().standardPalette()
            for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button, QPalette.ColorRole.Base):
                palette.setColor(role, QColor('#282828' if dark else '#fafafa'))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
                palette.setColor(role, QColor('#eeeeee' if dark else '#222222'))
            application.setPalette(palette)
            pump()
            dialog.tabs.setCurrentIndex(3)
            pump()
            llm, tts = dialog.fields['llm'], dialog.fields['tts']
            assert '~5 GiB VRAM' in llm.currentText()
            assert 'exceeds GPU' in llm.currentText()
            assert '~3 GiB VRAM' in tts.currentText()
            red = llm.itemData(llm.findData('Qwen/Qwen3-1.7B'), Qt.ItemDataRole.ForegroundRole)
            assert red is not None and red.color().red() > red.color().green()
            assert llm.lineEdit().palette().color(QPalette.ColorRole.Text) == red.color()
            assert combo_value(llm) == 'Qwen/Qwen3-1.7B'
            assert window.grab().save(str(output / f'qt-{name}-window.png'))
            assert dialog.grab().save(str(output / f'qt-{name}-models-4gb.png'))
            llm.setCurrentIndex(llm.findData('Qwen/Qwen3-0.6B'))
            assert llm.lineEdit().palette().color(QPalette.ColorRole.Text) == palette.color(QPalette.ColorRole.Text)
            llm.setCurrentIndex(llm.findData('Qwen/Qwen3-1.7B'))
        dialog.language_fields['document_language'].setCurrentIndex(0)
        voice = dialog.language_fields['voice_language']
        voice.setCurrentIndex(voice.findData('English'))
        dialog.save()
        pump()
        config = load_settings()
        assert config['document_language'] == 'Auto'
        assert config['voice_language'] == 'English'
        assert config['llm'] == 'Qwen/Qwen3-1.7B'
        assert config['tts'] == 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
        assert config == window.config
        assert 'python' not in config
        window.close()
        application.quit()
        print('Qt light/dark palettes, VRAM colors, model IDs, and language persistence passed.')


if __name__ == '__main__':
    main()
