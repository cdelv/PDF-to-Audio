# Build on the target OS. Models are downloaded by setup, never collected.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root / 'assets'), 'assets'), (str(root / 'packaging/model-files.json'), '.')]
datas += collect_data_files('qwen_tts')
datas += copy_metadata('qwen-tts', recursive=True)
hidden = collect_submodules('qwen_tts', filter=lambda n: '.cli' not in n)
hidden += ['transformers.models.qwen3.modeling_qwen3', 'transformers.models.qwen3.configuration_qwen3',
           'transformers.models.qwen2.tokenization_qwen2_fast', 'transformers.models.whisper.feature_extraction_whisper',
           'transformers.models.wav2vec2.feature_extraction_wav2vec2']
a = Analysis([str(root / 'app.py'), str(root / 'native_worker.py')], pathex=[str(root)],
             binaries=[], datas=datas, hiddenimports=hidden, hookspath=[],
             excludes=['gradio', 'matplotlib', 'IPython', 'pytest', 'tensorboard', 'torchvision'],
             module_collection_mode={'qwen_tts': 'pyz+py'}, noarchive=False)
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
