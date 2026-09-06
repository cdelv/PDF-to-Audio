# Build on the target OS. Models are downloaded by setup, never collected.
from pathlib import Path
import sys
import uv

root = Path(SPECPATH).parent
datas = [(str(root / 'assets'), 'assets'), (str(root / 'packaging/model-files.json'), '.')]
datas += [(str(root / name), 'engine') for name in
          ('native_worker.py', 'worker.py', 'core.py', 'hardware.py', 'model_store.py', 'checkpoints.py', 'languages.py', 'pdf_input.py')]
datas += [(str(root / 'requirements-engine.txt'), '.')]
a = Analysis([str(root / 'app.py'), str(root / 'native_worker.py')], pathex=[str(root)],
             binaries=[(uv.find_uv_bin(), '.')], datas=datas, hiddenimports=['filelock'], hookspath=[],
             excludes=['worker', 'torch', 'torchaudio', 'qwen_tts', 'transformers', 'soundfile',
                       'pdf_input', 'langdetect', 'gradio', 'matplotlib', 'IPython', 'pytest', 'tensorboard', 'torchvision'],
             noarchive=False)
pyz = PYZ(a.pure)
icons = root / 'build/icons'
icon = str(icons / ('icon.icns' if sys.platform == 'darwin' else 'icon.ico')) if sys.platform in ('win32', 'darwin') else None
gui = EXE(pyz, [s for s in a.scripts if s[0] != 'native_worker'], [], exclude_binaries=True,
          name='pdf-to-audio', console=False, icon=icon, upx=False)
worker = EXE(pyz, [s for s in a.scripts if s[0] != 'app'], [], exclude_binaries=True,
             name='pdf-to-audio-worker', console=True, upx=False)
bundle = COLLECT(gui, worker, a.binaries, a.datas, name='PDF-to-Audio', upx=False)
if sys.platform == 'darwin':
    app = BUNDLE(bundle, name='PDF to Audio.app', icon=icon,
                 bundle_identifier='io.github.pdftoaudio.Desktop',
                 info_plist={'CFBundleDisplayName': 'PDF to Audio', 'NSHighResolutionCapable': True})
