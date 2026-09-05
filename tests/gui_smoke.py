"""Shared GUI smoke test. Default uses a protocol stub; --audio speaks a short TXT."""
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
        from app import App, DesktopApplication, QUrl, Qt
        from PySide6.QtCore import QMimeData, QPoint, QPointF
        from PySide6.QtGui import QDragEnterEvent, QDropEvent
        from core import ROOT
        application = DesktopApplication([])
        window = App()
        window.show()
        window.config['output'] = str(ROOT / 'test-output/qt-audio')
        path = Path(directory) / 'Note.txt'
        path.write_text('Hello. This is a local audio test.')

        def pump(seconds=0.1):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                application.processEvents()
                time.sleep(0.01)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        enter = QDragEnterEvent(QPoint(30, 30), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        application.sendEvent(window, enter)
        assert enter.isAccepted()
        drop = QDropEvent(QPointF(30, 30), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        application.sendEvent(window, drop)
        assert drop.isAccepted() and len(window.rows) == 1
        window.add_files([str(path)])
        assert len(window.rows) == 1
        second = Path(directory) / 'Another.md'
        second.write_text('Another note.')
        window.add_files([str(second)])
        assert len(window.rows) == 2
        window.remove(window.rows[-1])
        window.settings()
        pump()
        window.settings_dialog.reject()
        protocol = '''import json, pathlib, sys
request = json.loads(sys.stdin.readline())
root = pathlib.Path(request['config']['output']); root.mkdir(parents=True, exist_ok=True)
target = root / 'protocol-test.wav'; target.write_bytes(b'GUI protocol test only')
print(json.dumps(dict(event='done', index=0, fraction=1, message='Ready', folder=str(root), audio=str(target))), flush=True)
print(json.dumps(dict(event='finished', completed=1, failed=0)), flush=True)
'''
        if '--audio' in sys.argv:
            window.start()
        else:
            with patch('app.missing_models', return_value=[]), \
                 patch('app.worker_command', return_value=[sys.executable, '-u', '-c', protocol]):
                window.start()
        assert window.process is not None and not window.start_button.isEnabled()
        deadline = time.monotonic() + 180
        while window.process is not None and time.monotonic() < deadline:
            pump()
        assert window.process is None, 'Worker timed out'
        assert window.rows[0]['audio'], window.status.text()
        assert window.rows[0]['play'].isEnabled() and window.rows[0]['files'].isEnabled()
        window.clear()
        window.add_files([str(path)])
        with patch('app.missing_models', return_value=[]), \
             patch('app.worker_command', return_value=[sys.executable, '-u', '-c', 'import time; time.sleep(30)']):
            window.start()
        pump(0.2)
        window.cancel()
        deadline = time.monotonic() + 8
        while window.process and time.monotonic() < deadline:
            pump()
        assert window.process is None, 'Cancellation left a worker running'
        assert window.start_button.isEnabled()
        checkpoint = Path(directory) / 'saved-conversion'
        checkpoint.mkdir()
        window.rows[0]['folder'] = str(checkpoint)
        window.elapsed = 65
        window.update_clock()
        assert window.timer_label.text().endswith('00:01:05')
        window.save_queue()
        restored = App()
        restored.restore_queue()
        assert restored.elapsed == 65
        assert restored.rows[0]['folder'] == str(checkpoint)
        assert restored.start_button.text() == 'Resume audio'
        restored.restart(restored.rows[0])
        assert restored.rows[0]['folder'] is None and checkpoint.exists()
        restored.close()
        window.close()
        application.quit()
        print('Qt drag/drop, settings, worker events, and cancellation passed.')


if __name__ == '__main__':
    main()
