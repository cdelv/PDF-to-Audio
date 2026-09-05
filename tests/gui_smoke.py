"""Exercise native widgets, drag/drop, settings, and a real worker from the GUI."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import App, Gdk, Gio, GLib, Gtk, ROOT


def main():
    app = App()
    app.set_application_id('io.github.pdftoaudio.Test')
    app.register(None)
    app.activate()
    app.config['output'] = str(ROOT/'test-output/gui-audio')
    context = GLib.MainContext.default()

    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            while context.pending():
                context.iteration(False)
            time.sleep(0.01)

    path = ROOT/'test-output/Long notes.md'
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        'Reading gives us a way to travel without leaving home. '
        'A good book can carry us across an ocean, introduce us to a new friend, or explain an idea we have never considered. '
        'Sometimes we have time to sit beside a window and turn the pages slowly. '
        'At other times, we want to keep listening while we walk through the park or make a cup of tea.\n\n'
        'This little application turns written documents into audio on your own computer. '
        'It reads the sentences in order and keeps the words together when it divides a longer document into passages. '
        'Each passage uses the same reference voice, and the finished recording joins them into one file. '
        'You can add several documents to the queue and let them finish while you do something else.\n\n'
        'When the recording is ready, press the play button to listen. '
        'The text and the individual audio passages are also saved, so you can inspect the result at any time. '
        'A new way to enjoy your reading is now only a few clicks away.'
    )
    value = Gdk.FileList.new_from_array([Gio.File.new_for_path(str(path)), Gio.File.new_for_path(str(ROOT/'test-output/First.pdf'))])
    assert app.on_drop(None, value, 0, 0)
    assert len(app.rows) == 2
    app.remove(app.rows[-1])
    assert len(app.rows) == 1
    app.add_files([str(path)])
    assert len(app.rows) == 1  # Duplicate files are ignored.
    app.add_files([str(ROOT/'test-output/First.pdf')])
    app.settings()
    pump(0.5)
    windows = Gtk.Window.get_toplevels()
    for i in range(windows.get_n_items()):
        window = windows.get_item(i)
        if window.get_title() == 'Voice & settings':
            window.destroy()
    app.start(None)
    assert app.process is not None
    assert not app.start_button.get_sensitive()
    pump(1)
    try:
        paintable = Gtk.WidgetPaintable.new(app.window)
        snapshot = Gtk.Snapshot.new()
        paintable.snapshot(snapshot, app.window.get_width(), app.window.get_height())
        node = snapshot.to_node()
        texture = app.window.get_renderer().render_texture(node, None)
        texture.save_to_png(str(ROOT/'test-output/window.png'))
        print('Saved window screenshot.', flush=True)
    except Exception as error:
        print('Screenshot unavailable:', error, flush=True)
    deadline = time.monotonic() + 600
    while app.process is not None and time.monotonic() < deadline:
        pump(0.1)
    assert app.process is None, 'Worker timed out'
    assert app.rows[0]['audio'], app.status.get_text()
    assert Path(app.rows[0]['audio']).is_file()
    assert app.rows[0]['play'].get_sensitive()
    assert app.rows[0]['files'].get_sensitive()
    assert all(row['audio'] for row in app.rows), app.status.get_text()
    print('GUI conversion passed:', app.rows[0]['audio'], flush=True)
    # Start another job and cancel during loading, exercising process cleanup.
    app.clear(None)
    app.add_files([str(path)])
    app.start(None)
    pump(0.5)
    app.cancel(None)
    deadline = time.monotonic() + 12
    while app.process and time.monotonic() < deadline:
        pump(0.1)
    assert app.process is None, 'Cancellation left a running worker'
    assert app.start_button.get_sensitive()
    print('Cancellation passed.', flush=True)
    app.window.destroy()
    app.quit()


if __name__ == '__main__':
    main()
