"""Check live light/dark switching and language settings without changing the desktop."""
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
        from app import App, Adw, GLib, Gtk, ROOT
        from core import load_settings
        app = App()
        app.set_application_id('io.github.pdftoaudio.AppearanceTest')
        app.register(None)
        app.activate()
        gpu_probe = patch('app.gpu_memory', return_value=4.0)
        gpu_probe.start()
        app.settings()
        context = GLib.MainContext.default()

        def pump(seconds=0.3):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                while context.pending():
                    context.iteration(False)
                time.sleep(0.01)

        def walk(widget):
            yield widget
            child = widget.get_first_child()
            while child:
                yield from walk(child)
                child = child.get_next_sibling()

        pump()
        gpu_probe.stop()
        windows = Gtk.Window.get_toplevels()
        dialog = next(windows.get_item(i) for i in range(windows.get_n_items())
                      if windows.get_item(i).get_title() == 'Voice & settings')
        notebook = next(w for w in walk(dialog) if isinstance(w, Gtk.Notebook))
        assert not any(isinstance(w, Gtk.Entry) and '.venv/bin/python' in w.get_text()
                       for w in walk(dialog))
        assert not any(isinstance(w, Gtk.Label) and 'Python environment' in w.get_text()
                       for w in walk(dialog))
        output = ROOT/'test-output/appearance'
        output.mkdir(parents=True, exist_ok=True)
        manager = app.get_style_manager()
        assert manager.get_color_scheme() == Adw.ColorScheme.DEFAULT

        def capture(window, path):
            paintable = Gtk.WidgetPaintable.new(window)
            snapshot = Gtk.Snapshot.new()
            paintable.snapshot(snapshot, window.get_width(), window.get_height())
            texture = window.get_renderer().render_texture(snapshot.to_node(), None)
            texture.save_to_png(str(path))

        colors = []
        for name, scheme, dark in [('light', Adw.ColorScheme.FORCE_LIGHT, False),
                                    ('dark', Adw.ColorScheme.FORCE_DARK, True)]:
            manager.set_color_scheme(scheme)
            notebook.set_current_page(0)
            pump()
            assert manager.get_dark() == dark
            found, color = app.window.get_style_context().lookup_color('window_bg_color')
            assert found
            colors.append(color.red + color.green + color.blue)
            capture(app.window, output/(name+'-window.png'))
            capture(dialog, output/(name+'-voice.png'))
            notebook.set_current_page(1)
            pump()
            capture(dialog, output/(name+'-languages.png'))
            notebook.set_current_page(3)
            pump()
            model_choices = [w for w in walk(notebook.get_nth_page(3)) if isinstance(w, Gtk.ComboBoxText)]
            tts, llm, processor = model_choices
            assert '~3 GiB VRAM' in tts.get_active_text()
            assert '~5 GiB VRAM' in llm.get_active_text()
            assert 'exceeds GPU' in llm.get_active_text()
            assert llm.get_child().has_css_class('error')
            assert not tts.get_child().has_css_class('error')
            capture(dialog, output/(name+'-models-4gb.png'))
            llm.set_active_id('Qwen/Qwen3-0.6B')
            assert not llm.get_child().has_css_class('error')
            llm.set_active_id('Qwen/Qwen3-1.7B')
        assert colors[0] > colors[1] + 1, colors
        choices = [w for w in walk(notebook.get_nth_page(1)) if isinstance(w, Gtk.ComboBoxText)]
        assert len(choices) == 2
        choices[0].set_active_id('Auto')
        choices[1].set_active_id('English')
        save = next(w for w in walk(dialog) if isinstance(w, Gtk.Button) and w.get_label() == 'Save settings')
        save.emit('clicked')
        pump()
        config = load_settings()
        assert config['document_language'] == 'Auto'
        assert config['voice_language'] == 'English'
        assert config['llm'] == 'Qwen/Qwen3-1.7B'
        assert config['tts'] == 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
        assert config == app.config
        manager.set_color_scheme(Adw.ColorScheme.DEFAULT)
        app.window.destroy()
        app.quit()
        print('Live light/dark themes and independent language settings passed.')


if __name__ == '__main__':
    main()
