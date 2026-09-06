"""One Qt desktop interface for Linux, Windows, and macOS."""
import json
from pathlib import Path
import shutil
import sys
import threading
import time

# Frozen applications launch this same executable in worker mode. Do this
# before importing Qt so inference has no window or GUI runtime overhead.
if __name__ == '__main__' and '--worker' in sys.argv:
    from worker import main as worker_main
    sys.exit(worker_main())

from PySide6.QtCore import QEvent, QObject, QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPalette
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from core import APP_ID, DATA, ROOT, RUNTIME, SUPPORTED, LANGUAGES, load_settings, save_settings, worker_command
from hardware import gpu_memory, model_label
from model_store import DEFAULT_MODELS, missing_models
from checkpoints import atomic_json
from runtime_setup import ready as runtime_ready


def label(text, name=None):
    result = QLabel(text)
    result.setWordWrap(True)
    if name:
        result.setObjectName(name)
    return result


def button(text, callback, name=None):
    result = QPushButton(text)
    result.clicked.connect(callback)
    if name:
        result.setObjectName(name)
    return result


def combo_value(combo):
    index = combo.findText(combo.currentText())
    return combo.itemData(index) if index >= 0 else combo.currentText().strip()


class GpuProbe(QObject):
    ready = Signal(object)

    def run(self):
        memory = gpu_memory()
        try:
            self.ready.emit(memory)
        except RuntimeError:
            pass  # Settings may have closed while the driver was being queried.


class SettingsDialog(QDialog):
    saved = Signal(dict)

    def __init__(self, window):
        super().__init__(window)
        self.owner = window
        self.config = dict(window.config)
        self.capacity = None
        self.setWindowTitle('Voice & settings')
        self.setWindowIcon(window.windowIcon())
        self.resize(690, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.addWidget(label('Make it sound like you.', 'heading'))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        def tab(title):
            widget = QWidget()
            body = QVBoxLayout(widget)
            body.setSpacing(12)
            body.setContentsMargins(16, 16, 16, 16)
            self.tabs.addTab(widget, title)
            return body

        def editor(path):
            view = QPlainTextEdit()
            if Path(path).is_file():
                view.setPlainText(Path(path).read_text(encoding='utf-8'))
            return view

        voice = tab('Voice')
        voice.addWidget(label('Voice sample · 2–30 seconds of clear speech', 'muted'))
        self.voice_path = QLineEdit(self.config['voice'])
        self.voice_path.setReadOnly(True)
        voice.addWidget(self.voice_path)
        actions = QHBoxLayout()
        actions.addWidget(button('Choose audio…', self.choose_voice))
        actions.addWidget(button('Listen to sample', lambda: window.open_path(self.voice_path.text())))
        voice.addLayout(actions)
        voice.addWidget(label('Exact transcript of the sample'))
        self.transcript = editor(self.config['transcript'])
        voice.addWidget(self.transcript, 1)
        voice.addWidget(button('Import transcript…', self.import_transcript))

        languages = tab('Languages')
        self.language_fields = {}
        for key, title, auto, hint in (
            ('document_language', 'Document / narration text language', 'Automatic — detect each document',
             'Always preserve the original language. A manual choice corrects ambiguous detection; it never translates.'),
            ('voice_language', 'Voice sample / reference transcript language', 'Automatic — detect from sample transcript',
             'Independent of the document language. A manual choice checks the reference transcript; the recording determines the voice and accent.'),
        ):
            languages.addWidget(label(title))
            field = QComboBox()
            field.addItem(auto, 'Auto')
            for name in LANGUAGES:
                field.addItem(name, name)
            field.setCurrentIndex(max(0, field.findData(self.config[key])))
            languages.addWidget(field)
            languages.addWidget(label(hint, 'muted'))
            self.language_fields[key] = field
        languages.addWidget(label('Supported: ' + ', '.join(LANGUAGES) + '.', 'muted'))
        languages.addStretch()

        prompt = tab('Cleanup prompt')
        prompt.addWidget(label('Instructions for PDF cleanup. Text files bypass this step.', 'muted'))
        self.prompt = editor(self.config['prompt'])
        prompt.addWidget(self.prompt, 1)

        models = tab('Models')
        models.addWidget(label('Both 0.6B models are installed by default. Saving a missing 1.7B selection downloads it automatically; models run locally afterwards.', 'muted'))
        self.gpu_status = label('Detecting GPU memory…', 'muted')
        models.addWidget(self.gpu_status)
        models.addWidget(label('VRAM figures are estimates for NVIDIA GPUs. Red means the model exceeds your GPU capacity. Choose both 0.6B models for a 4 GB card; select CPU if you run out of GPU memory.', 'muted'))
        self.fields = {}
        for key, title, choices in (
            ('tts', 'Speech model', ['Qwen/Qwen3-TTS-12Hz-0.6B-Base', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base']),
            ('llm', 'PDF cleanup model', ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B']),
            ('device', 'Processor', ['auto', 'mps', 'cpu'] if sys.platform == 'darwin' else ['auto', 'cuda:0', 'cpu']),
        ):
            models.addWidget(label(title))
            field = QComboBox()
            field.setEditable(True)
            if self.config[key] not in choices:
                choices.append(self.config[key])
            for name in choices:
                field.addItem(model_label(name, None)[0] if key != 'device' else
                              {'auto': 'Automatic', 'mps': 'Apple Metal (MPS)', 'cuda:0': 'NVIDIA CUDA', 'cpu': 'CPU'}.get(name, name), name)
            field.setCurrentIndex(field.findData(self.config[key]))
            field.setMinimumContentsLength(24)
            field.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            field.currentTextChanged.connect(self.color_models)
            models.addWidget(field)
            self.fields[key] = field
        if sys.platform == 'darwin':
            models.addWidget(label('Automatic uses Apple Metal when available. Unsupported operations can use CPU. Metal inference still needs validation on physical Macs; CPU is always selectable.', 'muted'))
        models.addStretch()
        self.feedback = label('Voice, transcript, and prompt are saved as ordinary files.', 'muted')
        layout.addWidget(self.feedback)
        actions = QHBoxLayout()
        actions.addWidget(button('Cancel', self.reject))
        actions.addStretch()
        actions.addWidget(button('Save settings', self.save, 'primary'))
        layout.addLayout(actions)
        self.probe = GpuProbe(self)
        self.probe.ready.connect(self.show_gpu)
        threading.Thread(target=self.probe.run, daemon=True).start()
        QApplication.instance().paletteChanged.connect(self.color_models)

    def choose_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose voice sample', '', 'Audio (*.wav *.flac)')
        if path:
            self.voice_path.setText(path)

    def import_transcript(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import transcript', '', 'Text (*.txt *.md)')
        if path:
            try:
                self.transcript.setPlainText(Path(path).read_text(encoding='utf-8-sig'))
            except Exception as error:
                self.feedback.setText(str(error))

    def show_gpu(self, capacity):
        self.capacity = capacity
        self.gpu_status.setText(f'Detected GPU: {capacity:.1f} GiB VRAM' if capacity is not None
            else 'Apple Metal uses shared system memory; availability is checked when a model loads.' if sys.platform == 'darwin'
            else 'NVIDIA GPU unavailable. Automatic can use CPU.')
        for key in ('tts', 'llm'):
            combo = self.fields[key]
            selected = combo_value(combo)
            combo.blockSignals(True)
            for index in range(combo.count()):
                combo.setItemText(index, model_label(combo.itemData(index), capacity)[0])
            index = combo.findData(selected)
            if index >= 0:
                combo.setCurrentIndex(index)
                combo.setEditText(combo.itemText(index))
            combo.blockSignals(False)
        self.color_models()

    def color_models(self, *_):
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        red = QColor('#ff8580' if dark else '#c01c28')
        for key in ('tts', 'llm'):
            combo = self.fields.get(key)
            if combo is None:
                continue
            for index in range(combo.count()):
                exceeds = model_label(combo.itemData(index), self.capacity)[1]
                combo.setItemData(index, QBrush(red) if exceeds else None, Qt.ItemDataRole.ForegroundRole)
            exceeds = model_label(combo_value(combo), self.capacity)[1]
            palette = QApplication.palette()
            if exceeds:
                palette.setColor(QPalette.ColorRole.Text, red)
            combo.lineEdit().setPalette(palette)

    def save(self):
        try:
            config = dict(self.config)
            for key, field in self.language_fields.items():
                config[key] = field.currentData()
            for key, field in self.fields.items():
                config[key] = combo_value(field)
                if not config[key]:
                    raise ValueError(f'Choose a {key} value.')
            source = Path(self.voice_path.text())
            if not source.is_file():
                raise ValueError('Choose a voice sample first.')
            texts = {'transcript': self.transcript.toPlainText().strip(), 'prompt': self.prompt.toPlainText().strip()}
            if not all(texts.values()):
                raise ValueError('The transcript and prompt must not be empty.')
            assets = DATA / 'assets'
            assets.mkdir(parents=True, exist_ok=True)
            target = assets / ('voice' + source.suffix.lower())
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            config['voice'] = str(target)
            for key, text in texts.items():
                target = assets / (key + '.txt')
                target.write_text(text + '\n', encoding='utf-8')
                config[key] = str(target)
            save_settings(config)
            self.saved.emit(config)
            self.accept()
        except Exception as error:
            self.feedback.setText(str(error))


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_settings()
        self.process = None
        self.rows, self.active = [], []
        self.terminal_event = None
        self.closing = self.cancelling = False
        self.stdout_buffer = b''
        self.job = 'conversion'
        self.download_names = []
        self.after_download = None
        self.elapsed = 0.0
        self.started_at = None
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self.update_clock)
        self.setWindowTitle('PDF to Audio')
        self.setWindowIcon(QIcon(str(ROOT / 'assets/icon.svg')))
        self.setAcceptDrops(True)
        self.resize(800, 780)
        central = QWidget()
        self.setCentralWidget(central)
        body = QVBoxLayout(central)
        body.setContentsMargins(28, 24, 28, 24)
        body.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(label('Turn pages into listening.', 'title'), 1)
        self.settings_button = button('Settings', self.settings)
        header.addWidget(self.settings_button)
        body.addLayout(header)
        body.addWidget(label('Your documents, read in your voice. Made locally on this computer.', 'muted'))
        drop = QFrame()
        drop.setObjectName('dropZone')
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(18, 20, 18, 20)
        title = label('Drop your documents here', 'heading')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(title)
        subtitle = label('PDF · TXT · Markdown · RST · CSV · LOG   /   Add several at once', 'muted')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(subtitle)
        self.add_button = button('Choose files…', self.choose_documents, 'primary')
        drop_layout.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignCenter)
        body.addWidget(drop)
        queue_header = QHBoxLayout()
        self.count = label('DOCUMENTS · 0', 'muted')
        queue_header.addWidget(self.count, 1)
        self.clear_button = button('Clear', self.clear)
        self.resume_button = button('Resume folder…', self.resume_folder)
        queue_header.addWidget(self.resume_button)
        queue_header.addWidget(self.clear_button)
        body.addLayout(queue_header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(130)
        container = QWidget()
        self.queue = QVBoxLayout(container)
        self.queue.setContentsMargins(0, 0, 0, 0)
        self.queue.addStretch()
        scroll.setWidget(container)
        body.addWidget(scroll, 1)
        output = QHBoxLayout()
        output_text = QVBoxLayout()
        output_text.addWidget(label('SAVE AUDIO TO', 'muted'))
        self.output_label = label(self.config['output'])
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output_text.addWidget(self.output_label)
        output.addLayout(output_text, 1)
        self.output_button = button('Change…', self.choose_output)
        output.addWidget(self.output_button)
        output.addWidget(button('Open folder', lambda: self.open_path(self.config['output'])))
        body.addLayout(output)
        self.setup_panel = QFrame()
        setup = QVBoxLayout(self.setup_panel)
        setup.setContentsMargins(0, 8, 0, 8)
        self.setup_heading = label('Setup — downloading and installing', 'heading')
        setup.addWidget(self.setup_heading)
        setup.addWidget(label('Preparing app dependencies and models, not creating audio. Downloads can total several GB. '
                              'Conversion controls unlock when setup finishes. You can cancel setup or close the window.', 'muted'))
        self.setup_panel.hide()
        body.addWidget(self.setup_panel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        body.addWidget(self.progress)
        self.status = label('Ready. PDFs are cleaned before narration; text files go straight to speech.', 'muted')
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(self.status)
        self.timer_label = label('Elapsed · 00:00:00', 'muted')
        body.addWidget(self.timer_label)
        actions = QHBoxLayout()
        actions.addWidget(button('View log', lambda: self.open_path(DATA / 'conversion.log')))
        self.retry_button = button('Retry setup', self.retry_download)
        self.retry_button.hide()
        actions.addWidget(self.retry_button)
        actions.addStretch()
        self.cancel_button = button('Cancel', self.cancel)
        self.start_button = button('Create audio', self.start, 'primary')
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.start_button)
        body.addLayout(actions)
        self.apply_theme()
        QApplication.instance().paletteChanged.connect(self.apply_theme)
        self.refresh()

    def apply_theme(self, *_):
        palette = QApplication.palette()
        accent = palette.color(QPalette.ColorRole.Highlight).name()
        selected = palette.color(QPalette.ColorRole.HighlightedText).name()
        text = palette.color(QPalette.ColorRole.WindowText)
        muted = QColor(text)
        muted.setAlpha(170)
        self.setStyleSheet(f'''
            QLabel#title {{ font-size: 27px; font-weight: 700; }}
            QLabel#heading {{ font-size: 18px; font-weight: 600; }}
            QLabel#muted {{ color: {muted.name(QColor.NameFormat.HexArgb)}; }}
            QFrame#dropZone {{ border: 2px dashed {accent}; border-radius: 14px; }}
            QFrame#card {{ background: palette(base); border: 1px solid palette(mid); border-radius: 10px; }}
            QPushButton {{ padding: 8px 12px; border-radius: 6px; }}
            QPushButton#primary:enabled {{ background: {accent}; color: {selected}; border: none; }}
            QComboBox, QLineEdit {{ padding: 7px; }}
            QPlainTextEdit {{ padding: 6px; }}
            QProgressBar {{ border: none; background: palette(midlight); border-radius: 3px; }}
            QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
        ''')

    def notify(self, message):
        self.status.setText(message)

    def update_clock(self):
        seconds = int(self.elapsed + (time.monotonic() - self.started_at if self.started_at is not None else 0))
        self.timer_label.setText(f'Elapsed · {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}')

    def stop_clock(self):
        if self.started_at is not None:
            self.elapsed += time.monotonic() - self.started_at
            self.started_at = None
        self.clock.stop()
        self.update_clock()
        self.save_queue()

    def save_queue(self):
        elapsed = self.elapsed + (time.monotonic() - self.started_at if self.started_at is not None else 0)
        try:
            atomic_json(DATA / 'queue.json', dict(elapsed=elapsed, rows=[
                {key: row[key] for key in ('path', 'folder', 'audio', 'fraction')} for row in self.rows]))
        except OSError as error:
            self.notify(f'Could not save the queue: {error}. Resume folder can still reopen saved conversions.')

    def restore_queue(self):
        path = DATA / 'queue.json'
        if not path.exists():
            return
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.elapsed = max(0, float(saved.get('elapsed', 0)))
            for item in saved['rows']:
                self.add_files([item['path']])
                row = next((r for r in self.rows if r['path'] == item['path']), None)
                if row is not None:
                    row['folder'] = item.get('folder')
                    row['audio'] = item.get('audio') if item.get('audio') and Path(item['audio']).is_file() else None
                    row['fraction'] = 1 if row['audio'] else 0
                    row['play'].setEnabled(bool(row['audio']))
                    row['files'].setEnabled(bool(row['folder']))
                    row['status'].setText('Ready to listen' if row['audio'] else 'Ready to resume' if row['folder'] else 'Queued')
            self.update_clock()
            self.refresh()
            self.save_queue()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.notify('Could not restore the queue: ' + str(error))

    def resume_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose a saved conversion folder', self.config['output'])
        if not folder:
            return
        try:
            job = json.loads((Path(folder) / 'job.json').read_text(encoding='utf-8'))
            self.add_files([job['source']])
            row = next(r for r in self.rows if r['path'] == job['source'])
            row.update(folder=folder, audio=None, fraction=0)
            row['status'].setText('Ready to resume')
            row['files'].setEnabled(True)
            row['play'].setEnabled(False)
            self.refresh()
            self.save_queue()
        except (OSError, ValueError, KeyError, StopIteration) as error:
            self.notify('Cannot open that checkpoint: ' + str(error))

    def restart(self, row):
        # The old output directory is kept; this only detaches it from the queue.
        row.update(folder=None, audio=None, fraction=0)
        row['status'].setText('Queued for a new conversion · previous files kept')
        row['play'].setEnabled(False)
        row['files'].setEnabled(False)
        self.refresh()
        self.save_queue()

    def open_path(self, path):
        if not path or not Path(path).expanduser().exists():
            self.notify('This file or folder has not been created yet.')
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).expanduser().resolve()))):
            self.notify('No application is available to open this file.')

    def settings(self):
        if self.process:
            return
        self.settings_dialog = SettingsDialog(self)
        self.settings_dialog.saved.connect(self.settings_saved)
        self.settings_dialog.open()

    def settings_saved(self, config):
        self.config = config
        self.notify('Settings saved.')
        # Wait until the settings dialog closes before beginning a transfer.
        QTimer.singleShot(0, lambda: self.setup_models([config['llm'], config['tts']]))

    def retry_download(self):
        self.setup_models(self.download_names, self.after_download)

    def setup_models(self, names=DEFAULT_MODELS, after=None):
        if self.process:
            return
        needed = missing_models(names)
        if not needed and runtime_ready(self.config['device']):
            self.retry_button.hide()
            if after:
                after()
            return
        self.download_names, self.after_download = needed, after
        self.active = []
        self.launch(dict(download_models=needed, config=self.config), 'download')

    def choose_documents(self):
        patterns = ' '.join('*' + ext for ext in sorted(SUPPORTED))
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose documents', '', f'Documents ({patterns});;All files (*)')
        self.add_files(paths)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Save audio in…', self.config['output'])
        if path:
            self.config['output'] = path
            save_settings(self.config)
            self.output_label.setText(path)

    def dragEnterEvent(self, event):
        if not self.process and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not self.process:
            self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()

    def add_files(self, paths):
        if self.process:
            self.notify('Wait for this batch to finish before adding documents.')
            return
        ignored = []
        for filename in paths:
            path = Path(filename).resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                ignored.append(path.name)
                continue
            if any(row['path'] == str(path) for row in self.rows):
                continue
            card = QFrame()
            card.setObjectName('card')
            content = QHBoxLayout(card)
            content.setContentsMargins(14, 12, 14, 12)
            info = QVBoxLayout()
            name = label(path.name)
            name.setToolTip(str(path))
            info.addWidget(name)
            status = label('Queued · ' + ('PDF cleanup → narration' if path.suffix.lower() == '.pdf' else 'Direct narration'), 'muted')
            info.addWidget(status)
            content.addLayout(info, 1)
            row = dict(path=str(path), status=status, fraction=0, audio=None, folder=None, widget=card)
            row['play'] = button('Play', lambda checked=False, r=row: self.open_path(r['audio']))
            row['files'] = button('Files', lambda checked=False, r=row: self.open_path(r['folder']))
            row['remove'] = button('×', lambda checked=False, r=row: self.remove(r))
            row['restart'] = button('Start over', lambda checked=False, r=row: self.restart(r))
            row['remove'].setToolTip('Remove document')
            row['play'].setEnabled(False)
            row['files'].setEnabled(False)
            for key in ('play', 'files', 'restart', 'remove'):
                content.addWidget(row[key])
            self.rows.append(row)
            self.queue.insertWidget(self.queue.count() - 1, card)
        self.refresh()
        self.save_queue()
        if ignored:
            self.notify('Unsupported or missing files: ' + ', '.join(ignored))

    def remove(self, row):
        if not self.process:
            self.queue.removeWidget(row['widget'])
            row['widget'].deleteLater()
            self.rows.remove(row)
            self.refresh()
            self.save_queue()

    def clear(self):
        for row in list(self.rows):
            self.remove(row)
        self.progress.setValue(0)

    def refresh(self):
        busy = self.process is not None
        self.count.setText(f'DOCUMENTS · {len(self.rows)}')
        for widget in (self.settings_button, self.output_button, self.add_button, self.clear_button, self.resume_button):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy and any(not row['audio'] for row in self.rows))
        self.start_button.setText('Resume audio' if any(row['folder'] and not row['audio'] for row in self.rows) else 'Create audio')
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setText('Cancel setup' if busy and self.job == 'download' else 'Cancel')
        self.retry_button.setEnabled(not busy)
        for row in self.rows:
            row['remove'].setEnabled(not busy)
            row['restart'].setEnabled(not busy and bool(row['folder']))

    def start(self):
        if self.process:
            return
        self.active = [row for row in self.rows if not row['audio']]
        if not self.active:
            return
        try:
            for name in ('voice', 'transcript', 'prompt'):
                if not Path(self.config[name]).is_file():
                    raise ValueError(f'Missing {name}. Set it in Settings.')
            if not RUNTIME.is_file():
                raise ValueError("The app's private environment is missing. Reinstall PDF to Audio.")
            names = [self.config['tts']]
            if any(Path(row['path']).suffix.lower() == '.pdf' for row in self.active):
                names.append(self.config['llm'])
            if missing_models(names) or not runtime_ready(self.config['device']):
                self.setup_models(names, self.start)
                return
            for row in self.active:
                row['fraction'] = 0
                row['status'].setText('Queued')
            resume = {row['path']: row['folder'] for row in self.active if row['folder']}
            if not resume:
                self.elapsed = 0.0
            self.launch(dict(config=self.config, files=[row['path'] for row in self.active], resume=resume), 'conversion')
        except Exception as error:
            self.notify(str(error))

    def launch(self, request, job):
        self.job = job
        if job == 'conversion':
            self.started_at = time.monotonic()
            self.clock.start()
            self.update_clock()
        self.terminal_event = None
        self.cancelling = False
        self.stdout_buffer = b''
        self.retry_button.hide()
        self.setup_panel.setVisible(job == 'download')
        self.setup_heading.setText('Setup — downloading and installing')
        self.timer_label.setVisible(job != 'download')
        self.progress.setRange(0, 0 if job == 'download' else 1000)
        self.progress.setValue(0)
        self.process = process = QProcess(self)
        process.readyReadStandardError.connect(self.read_diagnostics)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('HF_HUB_OFFLINE', '1' if job == 'conversion' else '0')
        environment.insert('TOKENIZERS_PARALLELISM', 'false')
        process.setProcessEnvironment(environment)

        def started():
            process.write((json.dumps(request) + '\n').encode('utf-8'))
            process.closeWriteChannel()

        process.started.connect(started)
        process.readyReadStandardOutput.connect(self.read_worker)
        process.finished.connect(self.exited)
        process.errorOccurred.connect(self.process_error)
        command = worker_command()
        process.start(command[0], command[1:])
        self.notify('Downloading and installing required app components… Please keep your internet connection active.' if job == 'download'
                    else 'Starting local conversion… First model load can take a little while.')
        self.refresh()

    def read_diagnostics(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError())
        if not data:
            return
        try:
            with (DATA / 'conversion.log').open('ab') as log:
                log.write(data)
        except OSError:
            pass  # Keep the window responsive even when a log cannot be written.

    def read_worker(self):
        if self.process is None:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split(b'\n', 1)
            try:
                self.event(json.loads(line.decode('utf-8')))
            except (ValueError, UnicodeError):
                continue

    def event(self, event):
        # QMainWindow also calls event(QEvent); JSON worker events share this
        # entry point only to keep the pipeline UI adapter small.
        if not isinstance(event, dict):
            return super().event(event)
        kind = event['event']
        if kind in ('download', 'models_ready'):
            fraction = event.get('fraction', 1 if kind == 'models_ready' else None)
            self.progress.setRange(0, 0 if fraction is None else 1000)
            if fraction is not None:
                self.progress.setValue(round(1000 * fraction))
            self.notify(event['message'] if fraction is None or kind == 'models_ready'
                        else f"{fraction:.0%} — {event['message']}")
            if kind == 'models_ready':
                self.terminal_event = kind
        elif 'index' in event and 0 <= event['index'] < len(self.active):
            row = self.active[event['index']]
            row['status'].setText(event['message'])
            row['fraction'] = event.get('fraction', 1 if kind == 'error' else row['fraction'])
            if event.get('folder'):
                row['folder'] = event['folder']
                row['files'].setEnabled(True)
            if kind == 'done':
                row['audio'] = event['audio']
                row['play'].setEnabled(True)
            self.progress.setValue(round(1000 * sum(row['fraction'] for row in self.active) / len(self.active)))
            self.notify(event['message'])
            self.save_queue()
        elif kind == 'finished':
            self.terminal_event = kind
            self.notify(f"Finished · {event['completed']} audio files created · {event['failed']} failed. Use Play to listen or Files to review the text and audio.")
        elif kind in ('cancelled', 'fatal'):
            self.terminal_event = kind
            self.notify(event['message'])
            for row in self.active:
                if row['fraction'] < 1:
                    row['status'].setText('Cancelled' if kind == 'cancelled' else 'Failed · ' + event['message'])
        return True

    def process_error(self, error):
        if self.process and error == QProcess.ProcessError.FailedToStart:
            self.notify('Could not start the private worker: ' + self.process.errorString())
            self.process.deleteLater()
            self.process = None
            self.stop_clock()
            self.retry_button.setVisible(self.job == 'download')
            self.setup_heading.setText('Setup could not start')
            self.progress.setRange(0, 1000)
            self.refresh()

    def exited(self, code, _status):
        self.read_worker()
        self.read_diagnostics()
        process, self.process = self.process, None
        if process:
            process.deleteLater()
        self.stop_clock()
        self.progress.setRange(0, 1000)
        if self.cancelling:
            self.notify('Setup cancelled. Installed components and completed model downloads are kept. Retry setup to continue.' if self.job == 'download'
                        else 'Cancelled. Completed passages are kept in the output folder.')
            for row in self.active:
                if row['fraction'] < 1:
                    row['status'].setText('Cancelled')
        elif self.terminal_event is None:
            self.notify(f'{"Setup" if self.job == "download" else "Conversion"} stopped (exit {code}). See View log for details.')
        ready = self.job == 'download' and self.terminal_event == 'models_ready' and code == 0 and not self.cancelling
        self.retry_button.setVisible(self.job == 'download' and not ready)
        if ready:
            self.setup_panel.hide()
        elif self.job == 'download':
            self.setup_heading.setText('Setup cancelled' if self.cancelling else 'Setup did not finish')
        self.refresh()
        if self.closing:
            self.close()
        elif ready and self.after_download:
            callback, self.after_download = self.after_download, None
            QTimer.singleShot(0, callback)

    def cancel(self):
        if self.process:
            self.cancelling = True
            process = self.process
            process.terminate()
            self.notify('Stopping download… Partial files will be kept.' if self.job == 'download'
                        else 'Stopping… Completed passages will be kept.')
            self.cancel_button.setEnabled(False)

            def force_stop():
                if self.process is process and process.state() != QProcess.ProcessState.NotRunning:
                    process.kill()

            QTimer.singleShot(5000, force_stop)

    def closeEvent(self, event):
        if self.process:
            self.closing = True
            self.cancel()
            event.ignore()
        else:
            self.save_queue()
            event.accept()


class DesktopApplication(QApplication):
    def __init__(self, args):
        super().__init__(args)
        if sys.platform.startswith('linux'):
            from system_theme import SystemTheme
            self.system_theme = SystemTheme(self)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen and hasattr(self, 'window'):
            self.window.add_files([event.file()])
            self.window.showNormal()
            self.window.activateWindow()
            return True
        return super().event(event)


def main():
    global DATA
    smoke = '--gui-smoke' in sys.argv
    if smoke:
        import tempfile
        import core
        smoke_directory = tempfile.TemporaryDirectory(prefix='pdf-to-audio-gui-test-')
        DATA = core.DATA = Path(smoke_directory.name) / 'data'
        core.CONFIG = Path(smoke_directory.name) / 'settings.json'
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    application = DesktopApplication(sys.argv)
    application.setApplicationName(APP_ID)
    application.setApplicationDisplayName('PDF to Audio')
    application.setDesktopFileName(APP_ID)
    application.setWindowIcon(QIcon(str(ROOT / 'assets/icon.svg')))
    application.window = window = App()
    if not smoke:
        window.restore_queue()
    window.add_files([arg for arg in sys.argv[1:] if not arg.startswith('-')])
    window.show()
    if smoke:
        window.launch(dict(ping=True), 'download')
        probe = QTimer(application)

        def check_helper():
            if window.process is None:
                success = window.terminal_event == 'models_ready' and window.isVisible() and not window.grab().isNull()
                application.exit(0 if success else 1)

        probe.timeout.connect(check_helper)
        probe.start(100)
        QTimer.singleShot(20000, lambda: application.exit(1))
    else:
        QTimer.singleShot(0, window.setup_models)
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
