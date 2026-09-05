"""Crash-safe conversion checkpoints; reuse only matching inputs and settings."""
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fingerprint(source, config):
    from model_store import MODELS
    value = dict(version=1, source=digest(source))
    for key in ('llm', 'tts', 'voice_language', 'document_language'):
        value[key] = config.get(key)
    for key in ('llm', 'tts'):
        value[key + '_revision'] = MODELS.get(config.get(key), {}).get('revision')
    for key in ('voice', 'transcript', 'prompt'):
        value[key] = digest(config[key]) if config.get(key) else None
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def open_job(folder, source, config, resume=False):
    path = folder / 'job.json'
    expected = fingerprint(source, config)
    if resume:
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('version') != 1 or value.get('fingerprint') != expected:
            raise ValueError('Cannot resume: the document, voice, model, language, or cleanup prompt changed. Restore the original settings or start a new conversion.')
        return value
    value = dict(version=1, fingerprint=expected, source=str(source.resolve()), status='preparing')
    atomic_json(path, value)
    return value


def valid_audio(path):
    import soundfile as sf
    try:
        with sf.SoundFile(path) as audio:
            if audio.frames <= 0 or audio.channels != 1 or audio.samplerate <= 0:
                return False
            # Read the complete file to detect damaged/truncated recordings.
            return sum(len(block) for block in audio.blocks(blocksize=65536)) == audio.frames
    except (OSError, RuntimeError):
        return False
