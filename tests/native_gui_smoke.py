"""Launch the real Windows/Cocoa/X11 Qt backend, without starting conversion."""
import os
from pathlib import Path
import sys

os.environ['QT_QPA_PLATFORM'] = 'windows' if sys.platform == 'win32' else 'cocoa' if sys.platform == 'darwin' else 'xcb'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from app import App, DesktopApplication, QTimer
    from core import ROOT
    application = DesktopApplication([])
    window = App()
    window.show()
    result = []

    def check():
        try:
            assert application.platformName() == os.environ['QT_QPA_PLATFORM']
            assert window.isVisible()
            output = ROOT / 'test-output/appearance'
            output.mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(output / 'native-window.png'))
            result.append(True)
        finally:
            window.close()
            application.quit()

    QTimer.singleShot(700, check)
    application.exec()
    assert result, 'Native GUI did not render successfully'
    print('Native Qt GUI launch passed:', application.platformName())


if __name__ == '__main__':
    main()
