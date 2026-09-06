"""Local inference worker. JSON events on stdout; library diagnostics on stderr."""
import contextlib
import gc
import json
import os
import re
from pathlib import Path
import signal
import sys
import tempfile
import traceback
import weakref

# Isolated Python ignores user site-packages, PYTHONPATH, and the working
# directory. Only this app's own modules are added to its private runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import SUPPORTED, cleanup_chunks, omit_tables, plain_text, speech_plan
from languages import check_cleanup_language, detect_language, resolve_language
from hardware import batch_size, virtual_metal
from model_store import local_model
from checkpoints import atomic_json, digest, open_job, valid_audio

# Downloads use the separate, pinned HTTP downloader, never Hub inference APIs.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# PyTorch must see this before its first import. Unsupported Metal operators
# may run on CPU; supported model operations remain on the Apple GPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


class Cancelled(BaseException):
    pass


def cancel(*_):
    raise Cancelled()


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def model_options(config):
    import torch
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and not virtual_metal() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select CPU or Automatic in Settings.")
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple Metal is unavailable. Select CPU or Automatic in Settings.')
    if device == 'mps' and virtual_metal():
        raise ValueError('The virtual Apple GPU cannot run Qwen reliably. Select CPU or use a physical Mac for Metal.')
    if device not in ('cpu', 'mps') and not device.startswith('cuda'):
        raise ValueError('Unknown processor. Select CPU, CUDA, Apple Metal, or Automatic in Settings.')
    torch.set_num_threads(os.cpu_count() or 1)
    # Local cache only: document conversion never needs an Internet connection.
    dtype = torch.float32 if device in ("cpu", "mps") else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return dict(device_map=device, dtype=dtype,
                attn_implementation="sdpa", local_files_only=True)


class Cleaner:
    def __init__(self, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = local_model(config["llm"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_options(config))
        self.prompt = Path(config["prompt"]).read_text(encoding="utf-8").strip()
        if not self.prompt:
            raise ValueError("The cleanup prompt is empty. Edit it in Settings.")

    def close(self):
        self.model = self.tokenizer = None

    def clean(self, text, progress, language="Auto", checkpoint=None):
        pieces = cleanup_chunks(omit_tables(text))
        cleaned = []
        self.warnings = []
        saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint and checkpoint.exists() else []
        for i, part in enumerate(pieces):
            progress(i, len(pieces))
            if i < len(saved) and saved[i]['source'] == part:
                cleaned.append(saved[i]['text'])
                self.warnings.extend(saved[i]['warnings'])
                continue
            saved = saved[:i]
            start = len(self.warnings)
            cleaned.append(self.clean_part(part, language))
            saved.append(dict(source=part, text=cleaned[-1], warnings=self.warnings[start:]))
            if checkpoint:
                atomic_json(checkpoint, saved)
        return "\n\n".join(cleaned)

    def clean_part(self, part, language, depth=0):
        import torch
        try:
            # A detection hint must never become a translation instruction.
            language_rule = "Keep each passage in its original language. Do not translate prose into a different language."
            source_language = detect_language(part)
            if source_language in ("Auto",) or len(part.split()) < 30:
                source_language = language
            if source_language != "Auto":
                language_rule += f" The detected source language is {source_language}."
            messages = [{"role": "system", "content": self.prompt + "\n\n" + language_rule},
                        {"role": "user", "content": "Document excerpt:\n<document>\n" + part + "\n</document>"}]
            chat = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            budget = min(2048, max(512, len(self.tokenizer.encode(part)) * 2 + 128),
                         self.model.config.max_position_embeddings - input_length)
            if budget < 512:
                raise ValueError("The cleanup prompt and excerpt exceed the model context. Shorten the prompt.")
            with torch.inference_mode():
                result = self.model.generate(**inputs, max_new_tokens=budget, do_sample=True,
                                             temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
            tokens = result[0, input_length:]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if len(tokens) >= budget and int(tokens[-1]) not in eos:
                raise ValueError("Text cleanup reached its output limit. No truncated narration was accepted.")
            narration = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not narration or "<think>" in narration:
                raise ValueError("Qwen returned no usable narration. Review the extracted Markdown and prompt.")
            check_cleanup_language(source_language, narration)
            # Catch obvious summaries/hallucinations, not stylistic edits. Keep
            # the original if the small model cannot safely clean an excerpt.
            source_words = re.findall(r"\w{4,}", part.lower())
            output_words = re.findall(r"\w{4,}", narration.lower())
            if len(source_words) > 40 and len(output_words) < len(source_words) * 0.45:
                raise ValueError("Cleanup omitted too much source text.")
            if len(output_words) > 20 and sum(w in set(source_words) for w in output_words) < len(output_words) * 0.55:
                raise ValueError("Cleanup added too much text not present in the source.")
            return narration
        except ValueError as error:
            # Invalid configuration must remain actionable, not trigger retries.
            if "model context" in str(error):
                raise
            if depth < 2 and len(part) > 400:
                return "\n\n".join(self.clean_part(piece, language, depth + 1)
                                   for piece in cleanup_chunks(part, max(200, len(part) // 2)))
            self.warnings.append(f"Original excerpt retained after uncertain cleanup: {error}\nExcerpt: {part[:160].strip()}")
            repaired = re.sub(r"(?<=\w)-\n(?=\w)", "", part)
            return plain_text(re.sub(r"(?<!\n)\n(?!\n)", " ", repaired))


def serial_audio_decoder(tokenizer):
    # Do not create tokenizer -> wrapper -> bound method -> tokenizer cycles.
    decode = weakref.WeakMethod(tokenizer.decode)

    def decode_serial(codes):
        waves, sample_rate = [], None
        for code in codes:
            audio, rate = decode()([code])
            if len(audio) != 1 or (sample_rate is not None and rate != sample_rate):
                raise ValueError('Decoded speech passages have incompatible audio outputs.')
            waves.extend(audio)
            sample_rate = rate
        if sample_rate is None:
            raise ValueError('There are no speech tokens to decode.')
        return waves, sample_rate

    return decode_serial


class Speaker:
    def __init__(self, config):
        import numpy as np
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        audio, rate = sf.read(config["voice"], dtype="float32")
        if not 2 <= len(audio) / rate <= 30:
            raise ValueError("Use a voice sample between 2 and 30 seconds long.")
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
            raise ValueError("The voice sample is silent or contains invalid audio.")
        transcript = Path(config["transcript"]).read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError("Add the exact voice sample transcript in Settings.")
        self.voice_language = resolve_language(transcript, config.get("voice_language", "Auto"), "Voice sample transcript")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.model = Qwen3TTSModel.from_pretrained(local_model(config["tts"]), **model_options(config))
        self.prompt = self.model.create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=transcript)
        tokenizer = self.model.model.speech_tokenizer
        tokenizer.decode = serial_audio_decoder(tokenizer)
        import torch
        device = self.model.model.device
        memory = (torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else
                  torch.mps.recommended_max_memory() if device.type == 'mps' else None)
        self.batch_size = batch_size(memory)
        # Qwen infers the reference language from ref_audio/ref_text. It exposes
        # only a target-text language argument, never a separate reference tag.
        # The public audio API discards token counts. Check the underlying result
        # before decoding, so hitting a generation cap cannot silently truncate speech.
        generate = weakref.WeakMethod(self.model.model.generate)

        def checked_generate(*args, **kwargs):
            result = generate()(*args, **kwargs)
            if any(len(codes) >= kwargs["max_new_tokens"] - 1 for codes in result[0]):
                raise ValueError("Speech reached its generation limit. Shorten the sentence and retry.")
            return result

        self.model.model.generate = checked_generate

    def close(self):
        self.prompt = self.model = None

    def speak(self, text, language="Auto"):
        return self.speak_batch([dict(text=text, language=language)])[0]

    @staticmethod
    def finish_wave(wave, rate):
        import numpy as np
        wave = np.asarray(wave, dtype=np.float32)
        if len(wave) == 0 or not np.isfinite(wave).all() or np.max(np.abs(wave)) < 0.0001:
            raise ValueError("The speech model produced empty, silent, or invalid audio.")
        # A tiny ramp at the outer edges suppresses clicks without overlapping words.
        n = min(int(rate * 0.004), len(wave) // 2)
        if n:
            wave[:n] *= np.linspace(0, 1, n)
            wave[-n:] *= np.linspace(1, 0, n)
        return wave

    def speak_batch(self, passages):
        waves, rate = self.model.generate_voice_clone(
            text=[p['text'].strip() for p in passages], language=[p['language'] for p in passages],
            voice_clone_prompt=self.prompt, max_new_tokens=2048)
        if len(waves) != len(passages):
            raise ValueError('Speech batch returned an incorrect number of recordings.')
        return [(self.finish_wave(wave, rate), rate) for wave in waves]


def stitch(paths, target):
    """Stream lossless passages to disk; memory use stays independent of book size."""
    import soundfile as sf
    if not paths:
        raise ValueError("There are no audio passages to join.")
    first = sf.info(paths[0])
    temporary = target.with_name(target.stem + ".partial.flac")
    with sf.SoundFile(temporary, "w", samplerate=first.samplerate, channels=1,
                      format="FLAC", subtype="PCM_16") as output:
        for path in paths:
            with sf.SoundFile(path) as source:
                if source.samplerate != first.samplerate or source.channels != 1:
                    raise ValueError("Audio passages have incompatible formats.")
                for block in source.blocks(blocksize=65536):
                    output.write(block)
    temporary.replace(target)


def run_batch(config, files, emit, resume=None):
    with contextlib.ExitStack() as locks:
        return _run_batch(config, files, emit, resume or {}, locks)


def _run_batch(config, files, emit, resume, locks):
    import soundfile as sf
    from filelock import FileLock
    output = Path(config["output"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    ready, failed, completed = [], 0, 0
    cleaner = None
    try:
        for index, filename in enumerate(files):
            folder = None
            try:
                source = Path(filename)
                if not source.is_file() or source.suffix.lower() not in SUPPORTED:
                    raise ValueError("Select an existing PDF, TXT, Markdown, RST, CSV, or LOG file.")
                existing = resume.get(filename)
                folder = Path(existing).resolve() if existing else Path(tempfile.mkdtemp(prefix=source.stem[:80] + "-", dir=output))
                locks.enter_context(FileLock(str(folder / '.conversion.lock'), timeout=0))
                job = open_job(folder, source, config, resume=bool(existing))
                emit("progress", index=index, fraction=0.02, message="Reading document", folder=str(folder))
                if job['status'] in ('ready', 'complete'):
                    if digest(folder / 'passages.json') != job['plan_digest']:
                        raise ValueError('Saved narration plan changed or is damaged. Start a new conversion.')
                    plan = json.loads((folder / 'passages.json').read_text(encoding='utf-8'))
                    ready.append((index, folder, plan))
                    emit('progress', index=index, fraction=0.25, message='Resuming saved narration', folder=str(folder))
                    continue
                if source.suffix.lower() == ".pdf":
                    from pdf_input import extract_pdf
                    text = extract_pdf(source)
                    (folder / "source.md").write_text(text, encoding="utf-8")
                    if not text.strip():
                        raise ValueError("This PDF contains no extractable text. Scanned PDFs need OCR first.")
                else:
                    text = source.read_text(encoding="utf-8-sig")
                readable = plain_text(omit_tables(text))
                if not readable:
                    raise ValueError("This document has no readable text outside tables.")
                language = resolve_language(readable, config.get("document_language", "Auto"))
                if cleaner is None:
                    emit("progress", index=index, fraction=0.03, message="Loading Qwen3 text cleanup")
                    cleaner = Cleaner(config)
                text = cleaner.clean(text, lambda n, total: emit("progress", index=index,
                    fraction=0.05 + 0.2 * n / total, message=f"Cleaning {language} text · {n+1}/{total}"), language,
                    checkpoint=folder / 'cleanup.json')
                if cleaner.warnings:
                    (folder / "warnings.txt").write_text(
                        "Some excerpts were retained in their original wording because model cleanup was unreliable. "
                        "Review narration.txt for layout artifacts and unstructured table data.\n\n"
                        + "\n\n".join(cleaner.warnings), encoding="utf-8")
                text = plain_text(text)
                if not text:
                    raise ValueError("This document has no readable text.")
                (folder / "narration.txt").write_text(text, encoding="utf-8")
                plan = speech_plan(text)
                for passage in plan:
                    passage["language"] = language
                atomic_json(folder / 'passages.json', plan)
                job.update(status='ready', plan_digest=digest(folder / 'passages.json'))
                atomic_json(folder / 'job.json', job)
                ready.append((index, folder, plan))
                emit("progress", index=index, fraction=0.25, message=f"Text ready · {language} · {len(plan)} passages")
            except Exception as error:
                failed += 1
                emit("error", index=index, message=str(error), folder=str(folder) if folder else "")
                traceback.print_exc(file=sys.stderr)
    finally:
        if cleaner is not None:
            cleaner.close()
        del cleaner
        release_gpu()
    speaker = None
    try:
        for index, folder, plan in ready:
            try:
                parts = folder / 'passages'
                parts.mkdir(exist_ok=True)
                paths = [parts / f'{number+1:06d}.flac' for number in range(len(plan))]
                pending = [number for number, path in enumerate(paths) if not valid_audio(path)]
                if pending and speaker is None:
                    emit("progress", index=index, fraction=0.25, message="Loading Qwen3 voice")
                    speaker = Speaker(config)
                if speaker is not None:
                    atomic_json(folder / 'languages.json', dict(document_language=plan[0]['language'], voice_language=speaker.voice_language))
                done = len(plan) - len(pending)
                while pending:
                    count = speaker.batch_size
                    numbers = pending[:count]
                    emit('progress', index=index, fraction=0.25 + 0.7 * done / len(plan),
                         message=f'Speaking · {done}/{len(plan)} complete · batch of {len(numbers)}')
                    batch = [plan[number] for number in numbers]
                    rendered = speaker.speak_batch(batch)
                    for number, (wave, rate) in zip(numbers, rendered, strict=True):
                        path = paths[number]
                        temporary = path.with_suffix('.partial.flac')
                        sf.write(temporary, wave, rate, subtype='PCM_16')
                        temporary.replace(path)
                        done += 1
                    pending = pending[len(numbers):]
                    del rendered, wave
                    # Keep the weights/voice prompt; return unused batch working
                    # memory to the driver instead of retaining a growing pool.
                    release_gpu()
                emit("progress", index=index, fraction=0.96, message="Joining audio")
                target = folder / "audio.flac"
                stitch(paths, target)
                job = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
                job['status'] = 'complete'
                atomic_json(folder / 'job.json', job)
                completed += 1
                warning = " · Review warnings.txt" if (folder / "warnings.txt").exists() else ""
                emit("done", index=index, fraction=1.0, message=f"Ready to listen · {plan[0]['language']}{warning}", audio=str(target), folder=str(folder))
            except Exception as error:
                failed += 1
                emit("error", index=index, message=str(error), folder=str(folder))
                traceback.print_exc(file=sys.stderr)
    finally:
        if speaker is not None:
            speaker.close()
        del speaker
        release_gpu()
    emit("finished", completed=completed, failed=failed)


def main():
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    channel = sys.stdout

    def emit(event, **data):
        channel.write(json.dumps(dict(event=event, **data)) + "\n")
        channel.flush()

    try:
        request = json.loads(sys.stdin.readline())
        with contextlib.redirect_stdout(sys.stderr):
            if request.get('ping'):
                emit('models_ready', message='Worker connection verified.')
            elif 'download_models' in request:
                from model_store import ensure_models
                ensure_models(request['download_models'], emit)
                emit('models_ready', message='Models are ready. Narration works offline.')
            else:
                os.environ['HF_HUB_OFFLINE'] = '1'
                run_batch(request["config"], request["files"], emit, request.get('resume'))
    except Cancelled:
        emit("cancelled", message="Cancelled. Completed passages are kept in the output folder.")
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit("fatal", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
