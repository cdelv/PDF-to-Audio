"""Small, dependency-free document planning and app configuration helpers."""
import json
import os
from pathlib import Path
import re
import shutil
import sys
from html.parser import HTMLParser

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
APP_ID = "io.github.pdftoaudio.Desktop"
RUNTIME = Path(sys.executable) if getattr(sys, 'frozen', False) else ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
if sys.platform == 'win32':
    DATA = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'PDF to Audio'
    CONFIG = DATA / 'settings.json'
elif sys.platform == 'darwin':
    DATA = Path.home() / 'Library/Application Support/PDF to Audio'
    CONFIG = DATA / 'settings.json'
else:
    DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "pdf-to-audio"
    CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "pdf-to-audio/settings.json"


def worker_command():
    if getattr(sys, 'frozen', False):
        return [str(RUNTIME), '--worker']
    return [str(RUNTIME), '-I', '-u', str(ROOT / 'worker.py')]
TEXT_TYPES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv"}
SUPPORTED = TEXT_TYPES | {".pdf"}
LANGUAGES = ["English", "Chinese", "French", "German", "Italian", "Japanese",
             "Korean", "Portuguese", "Russian", "Spanish"]
SECTION_CHARS = 100_000
SPEECH_CHARS = 700
# Only sentence ends and paragraph breaks. Decimal points and common abbreviations
# are not split points. An oversized unpunctuated sentence must be edited by the user.
BOUNDARY = re.compile(r'[。！？][\"\u201d\u2019\')\]」』]*|[.!?][\"\u201d\u2019\')\]]*(?=\s|$)|\n[ \t]*\n+')
ABBREVIATION = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|[A-Z])\.$', re.I)


def split_text(text, limit):
    """Lossless, bounded splitting. Never cut a word or unfinished sentence."""
    if limit < 1:
        raise ValueError("Chunk size must be positive.")
    if not text.strip():
        return []
    ends = []
    for match in BOUNDARY.finditer(text):
        if match.group().endswith(".") and ABBREVIATION.search(text[max(0, match.end()-12):match.end()]):
            continue
        ends.append(match.end())
    ends.append(len(text))  # Preserve an unpunctuated final sentence.
    chunks, start, best = [], 0, 0
    for end in sorted(set(ends)):
        if end - start > limit:
            if best <= start:
                raise ValueError(
                    f"A sentence or paragraph exceeds {limit:,} characters near character {start:,}. "
                    "Add a sentence ending or a blank line there and try again."
                )
            chunks.append(text[start:best])
            start = best
            if end - start > limit:
                raise ValueError(f"A sentence or paragraph exceeds {limit:,} characters near character {start:,}. Add punctuation or a blank line.")
        best = end
    if start < len(text):
        chunks.append(text[start:])
    return chunks


def cleanup_chunks(text, limit=2000):
    """Prefer sentences; tolerate malformed PDF text before it has punctuation."""
    try:
        return split_text(text, limit)
    except ValueError:
        chunks = []
        while len(text) > limit:
            end = text.rfind("\n", 0, limit + 1)
            if end < limit // 2:
                end = text.rfind(" ", 0, limit + 1)
            if end < limit // 2:
                end = limit
            chunks.append(text[:end])
            text = text[end:]
        if text:
            chunks.append(text)
        return chunks


def speech_plan(text):
    return [dict(section=s, text=part) for s, section in enumerate(split_text(text, SECTION_CHARS), 1)
            for part in split_text(section, SPEECH_CHARS) if part.strip()]


def omit_tables(text):
    """Drop structured table data before PDF narration; retain HTML captions."""
    class Captions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inside = False
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag == "caption":
                self.inside = True

        def handle_endtag(self, tag):
            if tag == "caption":
                self.inside = False
                self.parts.append("\n\n")

        def handle_data(self, data):
            if self.inside:
                self.parts.append(data)

    def caption(match):
        parser = Captions()
        parser.feed(match.group())
        return "\n\n" + "".join(parser.parts).strip() + "\n\n"

    text = re.sub(r"<table\b[^>]*>.*?</table\s*>", caption, text, flags=re.I | re.S)
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        # Require a Markdown separator row, so ordinary prose with pipes stays.
        if i + 1 < len(lines) and "|" in lines[i] and "|" in lines[i + 1]:
            cells = lines[i + 1].strip().strip("|").split("|")
            if len(cells) >= 2 and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells):
                i += 2
                while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                    i += 1
                result.append("\n")
                continue
        result.append(lines[i])
        i += 1
    return "".join(result)


def plain_text(text):
    """Remove common Markdown presentation without calling a language model."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Markdown hard line breaks are intentional boundaries (e.g. author lists).
    text = re.sub(r"[ \t]{2,}\n", "\n\n", text)
    text = re.sub(r"(?m)^\s*```[^\n]*\n?", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+)", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = re.sub(r"(\*\*|__|`)", "", text)
    return text.strip()


def defaults():
    return dict(output=str(Path.home() / "Music/PDF to Audio"),
                voice=str(DATA / "assets/voice.wav"),
                transcript=str(DATA / "assets/transcript.txt"),
                prompt=str(DATA / "assets/prompt.txt"),
                llm="Qwen/Qwen3-1.7B", tts="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                voice_language="Auto", document_language="Auto", device="auto")


def load_settings():
    assets = DATA / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("voice.wav", "transcript.txt", "prompt.txt"):
        source, target = ROOT / "assets" / name, assets / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    config = defaults()
    if CONFIG.exists():
        config.update(json.loads(CONFIG.read_text()))
    # The old setting forced every document to English by default. Original-
    # language narration now detects each file independently, including upgrades.
    config.pop("language", None)
    config.pop("python", None)
    return config


def save_settings(config):
    config = {key: value for key, value in config.items() if key != "python"}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temp = CONFIG.with_suffix(".tmp")
    temp.write_text(json.dumps(config, indent=2) + "\n")
    temp.replace(CONFIG)
