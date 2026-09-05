"""Linux appearance-portal adapter; Windows/macOS use Qt's native appearance."""
from PySide6.QtCore import QObject, Slot, SLOT
from PySide6.QtDBus import QDBusConnection, QDBusInterface, QDBusMessage, QDBusPendingCallWatcher, QDBusVariant
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


class SystemTheme(QObject):
    def __init__(self, application):
        super().__init__(application)
        self.application = application
        self.original = application.palette()
        self.scheme = 0
        self.accent = self.original.color(QPalette.ColorRole.Highlight)
        bus = QDBusConnection.sessionBus()
        self.settings = QDBusInterface('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop',
                                     'org.freedesktop.portal.Settings', bus, self)
        self.settings.setTimeout(2000)
        bus.connect('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop',
                    'org.freedesktop.portal.Settings', 'SettingChanged', self,
                    SLOT('changed(QString,QString,QDBusVariant)'))
        for key in ('color-scheme', 'accent-color'):
            watcher = QDBusPendingCallWatcher(self.settings.asyncCallWithArgumentList('Read', ['org.freedesktop.appearance', key]), self)
            watcher.finished.connect(lambda result, setting=key: self.read(result, setting))

    def read(self, watcher, key):
        reply = watcher.reply()
        if reply.type() != QDBusMessage.MessageType.ErrorMessage and reply.arguments():
            self.changed('org.freedesktop.appearance', key, reply.arguments()[0])
        watcher.deleteLater()

    @Slot(str, str, QDBusVariant)
    def changed(self, namespace, key, value):
        if namespace != 'org.freedesktop.appearance':
            return
        while isinstance(value, QDBusVariant):
            value = value.variant()
        if key == 'color-scheme' and value in (0, 1, 2):
            self.scheme = value
        elif key == 'accent-color' and isinstance(value, (tuple, list)) and len(value) == 3:
            self.accent = QColor.fromRgbF(*value)
        else:
            return
        self.apply()

    def apply(self):
        palette = QPalette(self.original)
        if self.scheme:
            dark = self.scheme == 1
            roles = {
                'Window': ('#252525', '#f6f6f6'), 'Base': ('#303030', '#ffffff'),
                'AlternateBase': ('#282828', '#eeeeee'), 'Button': ('#363636', '#e8e8e8'),
                'WindowText': ('#eeeeee', '#222222'), 'Text': ('#eeeeee', '#222222'),
                'ButtonText': ('#eeeeee', '#222222'), 'Mid': ('#555555', '#cccccc'),
                'Midlight': ('#444444', '#dddddd'), 'Light': ('#666666', '#ffffff'),
                'Dark': ('#191919', '#aaaaaa'), 'ToolTipBase': ('#363636', '#ffffff'),
                'ToolTipText': ('#eeeeee', '#222222'),
            }
            for role, colors in roles.items():
                palette.setColor(getattr(QPalette.ColorRole, role), QColor(colors[0 if dark else 1]))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
                palette.setColor(QPalette.ColorGroup.Disabled, role, QColor('#858585'))
        palette.setColor(QPalette.ColorRole.Highlight, self.accent)
        palette.setColor(QPalette.ColorRole.Link, self.accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#111111' if self.accent.lightnessF() > 0.65 else '#ffffff'))
        self.application.setPalette(palette)
