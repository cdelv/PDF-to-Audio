#!/usr/bin/python3
"""Native GTK window. Inference lives in a separate, cancellable process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from core import APP_ID, DATA, ROOT, RUNTIME, SUPPORTED, LANGUAGES, load_settings, save_settings
from hardware import gpu_memory, model_label

CSS = b"""
window { background: @window_bg_color; color: @window_fg_color; }
headerbar { background: @headerbar_bg_color; color: @headerbar_fg_color; }
.title { font-size: 29px; font-weight: 800; letter-spacing: -1px; }
.subtitle, .muted { opacity: 0.7; }
.drop { background: alpha(@accent_bg_color, 0.08); border: 2px dashed alpha(@accent_bg_color, 0.5); border-radius: 16px; padding: 16px; }
.drop-title { font-size: 18px; font-weight: 700; }
.card { background: @card_bg_color; color: @card_fg_color; border: 1px solid alpha(@window_fg_color, 0.12); border-radius: 12px; padding: 14px; }
list { background: transparent; }
list > row { padding: 5px 0; }
button { border-radius: 8px; padding: 8px 13px; }
button.flat { background: transparent; border: none; box-shadow: none; }
.error { color: @error_color; }
.success { color: @success_color; }
textview, textview text { background: @view_bg_color; color: @view_fg_color; }
"""


def label(text, css=None):
    widget = Gtk.Label(label=text, xalign=0, wrap=True)
    if css:
        widget.add_css_class(css)
    return widget


def button(text, callback, css=None):
    widget = Gtk.Button(label=text)
    widget.connect("clicked", callback)
    if css:
        widget.add_css_class(css)
    return widget


def box(vertical=True, spacing=10):
    return Gtk.Box(orientation=Gtk.Orientation.VERTICAL if vertical else Gtk.Orientation.HORIZONTAL, spacing=spacing)


def margins(widget, n):
    for edge in ("top", "bottom", "start", "end"):
        getattr(widget, "set_margin_" + edge)(n)


def combo_value(combo):
    active = combo.get_active_id()
    if active:
        return active
    text = combo.get_child().get_text().strip()
    return next((row[1] for row in combo.get_model() if row[0] == text), text)


class App(Adw.Application):
    def __init__(self):
        # Match the desktop entry for Wayland dialogs and X11 window grouping.
        GLib.set_prgname(APP_ID)
        GLib.set_application_name("PDF to Audio")
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)
        self.window = None
        self.process = None
        self.rows = []
        self.active = []
        self.closing = False
        self.terminal_event = None
        self.choosers = []

    def do_startup(self):
        Adw.Application.do_startup(self)
        Gtk.Window.set_default_icon_name(APP_ID)

    def do_activate(self):
        if self.window is None:
            self.build()
        self.window.present()

    def do_open(self, files, _count, _hint):
        self.do_activate()
        self.add_files([file.get_path() for file in files if file.get_path()])

    def build(self):
        self.config = load_settings()
        # Follow the desktop color scheme live, including already-open settings.
        self.get_style_manager().set_color_scheme(Adw.ColorScheme.DEFAULT)
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider,
                                                 Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.window = Gtk.ApplicationWindow(application=self, title="PDF to Audio", default_width=780, default_height=770)
        self.window.set_icon_name(APP_ID)
        self.window.connect("close-request", self.close_request)
        header = Gtk.HeaderBar()
        header.set_title_widget(Gtk.Label(label="PDF to Audio"))
        self.settings_button = button("Settings", lambda _: self.settings())
        header.pack_end(self.settings_button)
        self.window.set_titlebar(header)
        body = box(spacing=16)
        margins(body, 28)
        self.window.set_child(body)
        body.append(label("Turn pages into listening.", "title"))
        body.append(label("Your documents, read in your voice. Made locally on this computer.", "subtitle"))
        self.drop = box(spacing=9)
        self.drop.add_css_class("drop")
        icon = Gtk.Image.new_from_icon_name("document-send-symbolic")
        icon.set_pixel_size(32)
        self.drop.append(icon)
        title = label("Drop your documents here", "drop-title")
        title.set_halign(Gtk.Align.CENTER)
        self.drop.append(title)
        subtitle = label("PDF · TXT · Markdown · RST · CSV · LOG   /   Add several at once", "muted")
        subtitle.set_halign(Gtk.Align.CENTER)
        self.drop.append(subtitle)
        self.add_button = button("Choose files…", self.choose_documents, "suggested-action")
        self.add_button.set_halign(Gtk.Align.CENTER)
        self.drop.append(self.add_button)
        target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        target.connect("drop", self.on_drop)
        self.window.add_controller(target)
        body.append(self.drop)
        queue_header = box(False)
        self.count = label("DOCUMENTS · 0", "muted")
        self.count.set_hexpand(True)
        queue_header.append(self.count)
        self.clear_button = button("Clear", self.clear, "flat")
        queue_header.append(self.clear_button)
        body.append(queue_header)
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.list.set_placeholder(label("Add a document to get started.", "muted"))
        scroll = Gtk.ScrolledWindow(vexpand=True, min_content_height=130)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(self.list)
        body.append(scroll)
        output = box(False)
        output_text = box(spacing=3)
        output_text.set_hexpand(True)
        output_text.append(label("SAVE AUDIO TO", "muted"))
        self.output_label = label(self.config["output"])
        self.output_label.set_ellipsize(3)
        output_text.append(self.output_label)
        output.append(output_text)
        self.output_button = button("Change…", self.choose_output)
        output.append(self.output_button)
        output.append(button("Open folder", lambda _: self.open_path(self.config["output"])))
        body.append(output)
        self.progress = Gtk.ProgressBar(show_text=False)
        body.append(self.progress)
        self.status = label("Ready. PDFs are cleaned before narration; text files go straight to speech.", "muted")
        self.status.set_selectable(True)
        body.append(self.status)
        actions = box(False)
        actions.append(button("View log", lambda _: self.open_path(DATA / "conversion.log"), "flat"))
        spacer = Gtk.Box(hexpand=True)
        actions.append(spacer)
        self.cancel_button = button("Cancel", self.cancel)
        self.cancel_button.set_sensitive(False)
        actions.append(self.cancel_button)
        self.start_button = button("Create audio", self.start, "suggested-action")
        self.start_button.set_sensitive(False)
        actions.append(self.start_button)
        body.append(actions)

    def notify(self, message):
        self.status.set_text(message)

    def open_path(self, path):
        try:
            path = Path(path).expanduser()
            if not path.exists():
                self.notify("This file or folder has not been created yet.")
                return
            Gio.AppInfo.launch_default_for_uri(path.resolve().as_uri(), None)
        except GLib.Error as error:
            self.notify(str(error))

    def chooser(self, title, callback, *, folder=False, multiple=False, patterns=None, parent=None):
        chooser = Gtk.FileChooserNative(title=title, transient_for=parent or self.window,
            action=Gtk.FileChooserAction.SELECT_FOLDER if folder else Gtk.FileChooserAction.OPEN,
            accept_label="Select", cancel_label="Cancel")
        chooser.set_select_multiple(multiple)
        if patterns:
            filt = Gtk.FileFilter(name="Supported files")
            for pattern in patterns:
                filt.add_pattern(pattern)
            chooser.add_filter(filt)
        def response(dialog, result):
            try:
                if result == Gtk.ResponseType.ACCEPT:
                    files = dialog.get_files()
                    paths = [files.get_item(i).get_path() for i in range(files.get_n_items())]
                    callback([path for path in paths if path])
            except Exception as error:
                self.notify(str(error))
            finally:
                self.choosers.remove(dialog)
                dialog.destroy()
        chooser.connect("response", response)
        self.choosers.append(chooser)
        chooser.show()

    def choose_documents(self, _):
        self.chooser("Choose documents", self.add_files, multiple=True,
                     patterns=["*" + ext for ext in sorted(SUPPORTED)] + ["*.PDF", "*.TXT", "*.MD"])

    def on_drop(self, _target, value, _x, _y):
        if self.process:
            self.notify("Wait for this batch to finish before adding more documents.")
            return False
        self.add_files([f.get_path() for f in value.get_files() if f.get_path()])
        return True

    def add_files(self, paths):
        if self.process:
            self.notify("Wait for this batch to finish before adding more documents.")
            return
        ignored = []
        for filename in paths:
            path = Path(filename).resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                ignored.append(path.name)
                continue
            if any(row["path"] == str(path) for row in self.rows):
                continue
            content = box(False, 12)
            content.add_css_class("card")
            content.append(Gtk.Image.new_from_icon_name("application-pdf-symbolic" if path.suffix.lower() == ".pdf" else "text-x-generic-symbolic"))
            info = box(spacing=4)
            info.set_hexpand(True)
            name = label(path.name)
            name.set_wrap(False)
            name.set_ellipsize(3)
            name.set_tooltip_text(str(path))
            info.append(name)
            status = label("Queued · " + ("PDF cleanup → narration" if path.suffix.lower() == ".pdf" else "Direct narration"), "muted")
            info.append(status)
            content.append(info)
            row = dict(path=str(path), status=status, fraction=0, audio=None, folder=None)
            play = button("Play", lambda _, r=row: self.open_path(r["audio"]))
            play.set_sensitive(False)
            content.append(play)
            folder = button("Files", lambda _, r=row: self.open_path(r["folder"]))
            folder.set_sensitive(False)
            content.append(folder)
            remove = button("×", lambda _, r=row: self.remove(r), "flat")
            remove.set_tooltip_text("Remove document")
            content.append(remove)
            wrapper = Gtk.ListBoxRow(child=content)
            row.update(widget=wrapper, play=play, files=folder, remove=remove)
            self.rows.append(row)
            self.list.append(wrapper)
        self.refresh()
        if ignored:
            self.notify("Unsupported or missing files: " + ", ".join(ignored))

    def remove(self, row):
        if not self.process:
            self.list.remove(row["widget"])
            self.rows.remove(row)
            self.refresh()

    def clear(self, _):
        for row in list(self.rows):
            self.remove(row)
        self.progress.set_fraction(0)

    def refresh(self):
        busy = self.process is not None
        self.count.set_text(f"DOCUMENTS · {len(self.rows)}")
        for widget in (self.settings_button, self.output_button, self.add_button, self.clear_button):
            widget.set_sensitive(not busy)
        self.start_button.set_sensitive(not busy and any(not r["audio"] for r in self.rows))
        self.cancel_button.set_sensitive(busy)
        for row in self.rows:
            row["remove"].set_sensitive(not busy)

    def choose_output(self, _):
        def selected(paths):
            if paths:
                self.config["output"] = paths[0]
                save_settings(self.config)
                self.output_label.set_text(paths[0])
        self.chooser("Save audio in…", selected, folder=True)

    def start(self, _):
        if self.process:
            return
        self.active = [row for row in self.rows if not row["audio"]]
        self.terminal_event = None
        if not self.active:
            return
        try:
            for name in ("voice", "transcript", "prompt"):
                if not Path(self.config[name]).is_file():
                    raise ValueError(f"Missing {name}. Set it in Settings.")
            if not RUNTIME.is_file():
                raise ValueError("The app's private environment is missing. Reinstall PDF to Audio.")
            log = open(DATA / "conversion.log", "a", buffering=1)
            try:
                self.process = subprocess.Popen([str(RUNTIME), "-I", "-u", str(ROOT / "worker.py")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            finally:
                log.close()
            for row in self.active:
                row["fraction"] = 0
                row["status"].set_text("Queued")
                row["status"].remove_css_class("error")
            process = self.process
            threading.Thread(target=self.read_worker, args=(process, dict(self.config), [r["path"] for r in self.active]), daemon=True).start()
            self.notify("Starting local conversion… First model load can take a little while.")
            self.refresh()
        except Exception as error:
            self.notify(str(error))

    def read_worker(self, process, config, files):
        try:
            process.stdin.write(json.dumps(dict(config=config, files=files)) + "\n")
            process.stdin.close()
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                GLib.idle_add(self.event, event)
        except Exception as error:
            GLib.idle_add(self.event, dict(event="fatal", message=str(error)))
        finally:
            process.stdout.close()
            code = process.wait()
            GLib.idle_add(self.exited, process, code)

    def event(self, event):
        kind = event["event"]
        if "index" in event:
            row = self.active[event["index"]]
            row["status"].set_text(event["message"])
            row["fraction"] = event.get("fraction", 1 if kind == "error" else row["fraction"])
            if event.get("folder"):
                row["folder"] = event["folder"]
                row["files"].set_sensitive(True)
            if kind == "done":
                row["audio"] = event["audio"]
                row["play"].set_sensitive(True)
                row["status"].add_css_class("success")
            if kind == "error":
                row["status"].add_css_class("error")
            self.progress.set_fraction(sum(r["fraction"] for r in self.active) / len(self.active))
            self.notify(event["message"])
        elif kind == "finished":
            self.terminal_event = kind
            self.notify(f"Finished · {event['completed']} audio files created · {event['failed']} failed. Use Play to listen or Files to see the text and audio.")
        elif kind in ("cancelled", "fatal"):
            self.terminal_event = kind
            self.notify(event["message"])
            for row in self.active:
                if row["fraction"] < 1:
                    row["status"].set_text("Cancelled" if kind == "cancelled" else "Failed · " + event["message"])
        return False

    def exited(self, process, code):
        if self.process is process:
            self.process = None
            if code != 0 and self.terminal_event is None:
                self.notify(f"Conversion stopped (exit {code}). See the document status or View log for details.")
                for row in self.active:
                    if row["fraction"] < 1:
                        row["status"].set_text("Stopped · partial files are kept")
            self.refresh()
            if self.closing:
                self.window.destroy()
        return False

    def cancel(self, _):
        if self.process:
            process = self.process
            if process.poll() is None:
                process.terminate()
                self.notify("Stopping… Completed passages will be kept.")
                self.cancel_button.set_sensitive(False)
                def force_stop():
                    if process.poll() is None:
                        process.kill()
                    return False
                GLib.timeout_add_seconds(5, force_stop)

    def close_request(self, _):
        # Registered settings windows must not keep the app alive after closing
        # its main window (some compositors do not close transients for us).
        for window in self.get_windows():
            if window.get_transient_for() is self.window:
                window.destroy()
        if self.process:
            self.closing = True
            self.cancel(None)
            return True
        return False

    def settings(self):
        dialog = Gtk.ApplicationWindow(application=self, title="Voice & settings", transient_for=self.window,
                            modal=True, destroy_with_parent=True,
                            default_width=670, default_height=690)
        body = box(spacing=14)
        margins(body, 22)
        dialog.set_child(body)
        body.append(label("Make it sound like you.", "drop-title"))
        notebook = Gtk.Notebook(vexpand=True)
        body.append(notebook)
        voice_tab = box(spacing=12)
        margins(voice_tab, 16)
        voice_tab.append(label("Voice sample · 2–30 seconds of clear speech", "muted"))
        voice_path = Gtk.Entry(text=self.config["voice"], editable=False)
        voice_tab.append(voice_path)
        voice_actions = box(False)
        voice_actions.append(button("Choose audio…", lambda _: self.chooser("Choose voice sample",
            lambda paths: voice_path.set_text(paths[0]) if paths else None, parent=dialog, patterns=["*.wav", "*.flac", "*.WAV", "*.FLAC"])))
        voice_actions.append(button("Listen to sample", lambda _: self.open_path(voice_path.get_text())))
        voice_tab.append(voice_actions)
        voice_tab.append(label("Exact transcript of the sample", "muted"))

        def editor(path, parent):
            view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
            view.set_left_margin(10)
            view.set_right_margin(10)
            view.set_top_margin(10)
            if Path(path).exists():
                view.get_buffer().set_text(Path(path).read_text(encoding="utf-8"))
            scroll = Gtk.ScrolledWindow(vexpand=True, min_content_height=180)
            scroll.set_child(view)
            parent.append(scroll)
            return view.get_buffer()

        transcript = editor(self.config["transcript"], voice_tab)
        voice_tab.append(button("Import transcript…", lambda _: self.chooser("Choose transcript",
            lambda paths: transcript.set_text(Path(paths[0]).read_text(encoding="utf-8-sig")) if paths else None,
            parent=dialog, patterns=["*.txt"])))
        notebook.append_page(voice_tab, Gtk.Label(label="Voice"))
        language_tab = box(spacing=14)
        margins(language_tab, 16)
        language_tab.append(label("Original language, every time.", "drop-title"))
        language_tab.append(label("An English voice sample can narrate a Spanish PDF. Each document keeps its own language; nothing is translated.", "muted"))
        language_fields = {}
        for key, title, auto, hint in (
            ("document_language", "Document / narration text language", "Automatic — detect each document",
             "Leave Automatic selected for PDFs in different languages in the same batch. Choose a language only to correct an ambiguous detection."),
            ("voice_language", "Voice sample / reference transcript language", "Automatic — detect from sample transcript",
             "Identifies the recording's language independently. A manual choice checks the reference transcript; the recording itself determines the voice and accent."),
        ):
            language_tab.append(label(title))
            field = Gtk.ComboBoxText()
            field.append("Auto", auto)
            for choice in LANGUAGES:
                field.append(choice, choice)
            field.set_active_id(self.config[key])
            language_tab.append(field)
            language_tab.append(label(hint, "muted"))
            language_fields[key] = field
        language_tab.append(label("Supported: " + ", ".join(LANGUAGES) + ".", "muted"))
        notebook.append_page(language_tab, Gtk.Label(label="Languages"))
        prompt_tab = box()
        margins(prompt_tab, 16)
        prompt_tab.append(label("Instructions for PDF cleanup. Text files bypass this step.", "muted"))
        prompt = editor(self.config["prompt"], prompt_tab)
        notebook.append_page(prompt_tab, Gtk.Label(label="Cleanup prompt"))
        models_tab = box(spacing=12)
        margins(models_tab, 16)
        models_tab.append(label("Models run locally. The smaller voice model is the faster default.", "muted"))
        gpu_status = label("Detecting GPU memory…", "muted")
        models_tab.append(gpu_status)
        models_tab.append(label("VRAM figures are estimates, not guarantees. Red means the model exceeds your card's capacity. Automatic uses CPU when the selected model will not fit; on 4 GB cards choose both 0.6B models for GPU use.", "muted"))
        fields = {}
        capacity = [None]

        def color_model(_layout, renderer, model, tree_iter, _data):
            exceeds = model_label(model.get_value(tree_iter, 1), capacity[0])[1]
            found, color = models_tab.get_style_context().lookup_color("error_color")
            renderer.set_property("foreground", color.to_string() if found else "#e01b24")
            renderer.set_property("foreground-set", exceeds)

        for key, title, choices in (
            ("tts", "Speech model", ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]),
            ("llm", "PDF cleanup model", ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"]),
            ("device", "Processor", ["auto", "cuda:0", "cpu"]),
        ):
            models_tab.append(label(title))
            field = Gtk.ComboBoxText.new_with_entry()
            if self.config[key] not in choices:
                choices.append(self.config[key])
            for choice in choices:
                field.append(choice, model_label(choice, None)[0] if key != "device" else choice)
            field.set_active_id(self.config[key])
            field.get_child().set_width_chars(28)
            if key != "device":
                field.set_cell_data_func(field.get_cells()[0], color_model, None)

                def update_color(combo):
                    name = combo_value(combo)
                    if model_label(name, capacity[0])[1]:
                        combo.get_child().add_css_class("error")
                    else:
                        combo.get_child().remove_css_class("error")

                field.connect("changed", update_color)
            models_tab.append(field)
            fields[key] = field

        def show_gpu(memory):
            capacity[0] = memory
            gpu_status.set_text(f"Detected GPU: {memory:.1f} GiB VRAM" if memory is not None
                                else "NVIDIA GPU memory unavailable. Estimates are shown; Automatic can use CPU.")
            for key in ("llm", "tts"):
                combo = fields[key]
                selected = combo.get_active_id()
                for row in combo.get_model():
                    row[0] = model_label(row[1], memory)[0]
                if selected:
                    combo.set_active_id(selected)
                    combo.get_child().set_text(model_label(selected, memory)[0])
                update_color(combo)
            return False

        threading.Thread(target=lambda: GLib.idle_add(show_gpu, gpu_memory()), daemon=True).start()
        notebook.append_page(models_tab, Gtk.Label(label="Models"))
        feedback = label("Voice, transcript, and prompt are saved as ordinary files.", "muted")
        body.append(feedback)
        actions = box(False)
        actions.append(button("Cancel", lambda _: dialog.destroy()))
        actions.append(Gtk.Box(hexpand=True))

        def save(_):
            try:
                config = dict(self.config)
                for key, field in language_fields.items():
                    config[key] = field.get_active_id()
                    if config[key] not in ["Auto", *LANGUAGES]:
                        raise ValueError("Choose a valid language or Automatic.")
                for key, field in fields.items():
                    config[key] = combo_value(field)
                    if not config[key]:
                        raise ValueError(f"Choose a {key} value.")
                audio = Path(voice_path.get_text())
                if not audio.is_file():
                    raise ValueError("Choose a voice sample first.")
                texts = {"transcript": transcript.get_text(transcript.get_start_iter(), transcript.get_end_iter(), True).strip(),
                         "prompt": prompt.get_text(prompt.get_start_iter(), prompt.get_end_iter(), True).strip()}
                if not all(texts.values()):
                    raise ValueError("The transcript and prompt must not be empty.")
                assets = DATA / "assets"
                target = assets / ("voice" + audio.suffix.lower())
                if audio.resolve() != target.resolve():
                    shutil.copy2(audio, target)
                config["voice"] = str(target)
                for key, text in texts.items():
                    path = assets / (key + ".txt")
                    path.write_text(text + "\n", encoding="utf-8")
                    config[key] = str(path)
                save_settings(config)
                self.config = config
                dialog.destroy()
                self.notify("Settings saved.")
            except Exception as error:
                feedback.set_text(str(error))
        actions.append(button("Save settings", save, "suggested-action"))
        body.append(actions)
        dialog.present()


if __name__ == "__main__":
    sys.exit(App().run(sys.argv))
