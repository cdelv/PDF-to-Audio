"""Crash-safe checkpoints; model changes affect only unfinished work."""
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


def identity(source, config):
    value = dict(version=2, source=digest(source))
    for key in ('voice_language', 'document_language'):
        value[key] = config.get(key)
    for key in ('voice', 'transcript', 'prompt'):
        value[key] = digest(config[key]) if config.get(key) else None
    return value


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fingerprint(source, config):
    return identity_hash(identity(source, config))


def legacy_matches(saved, inputs, config):
    """Validate old model-bound hashes before upgrading; never bypass asset checks."""
    from model_store import MODELS
    llms = dict.fromkeys([config.get('llm'), *(name for name in MODELS if '-TTS-' not in name)])
    voices = dict.fromkeys([config.get('tts'), *(name for name in MODELS if '-TTS-' in name)])
    for llm in llms:
        for tts in voices:
            old = dict(inputs, version=1, llm=llm, tts=tts,
                       llm_revision=MODELS.get(llm, {}).get('revision'),
                       tts_revision=MODELS.get(tts, {}).get('revision'))
            if saved == identity_hash(old):
                return True
    return False


def open_job(folder, source, config, resume=False):
    path = folder / 'job.json'
    inputs = identity(source, config)
    expected = identity_hash(inputs)
    if resume:
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('version') == 1 and legacy_matches(value.get('fingerprint'), inputs, config):
            value.update(version=2, fingerprint=expected)
            atomic_json(path, value)
        if value.get('version') != 2 or value.get('fingerprint') != expected:
            raise ValueError('Cannot resume: the document, voice, language, or cleanup prompt changed. Restore the original settings. For an older checkpoint made with a custom model, also restore its original model once to upgrade it.')
        return value
    value = dict(version=2, fingerprint=expected, source=str(source.resolve()), status='preparing')
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
