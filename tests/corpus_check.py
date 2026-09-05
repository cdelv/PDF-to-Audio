"""Read-only PDF corpus regression; outputs stay in test-output, not the source folder."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import BOUNDARY, ROOT, cleanup_chunks, load_settings, plain_text, speech_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stage', choices=['extract', 'clean', 'speak'], default='extract')
    parser.add_argument('--only')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sample', action='store_true', help='Clean/speak excerpts rather than full documents')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--cap-gib', type=float)
    args = parser.parse_args()
    output = ROOT / 'test-output/corpus'
    output.mkdir(parents=True, exist_ok=True)
    config = load_settings()
    if args.small:
        config['llm'] = 'Qwen/Qwen3-0.6B'
    if args.cap_gib:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.cap_gib * 2**30 / total)
    paths = sorted(args.directory.glob('*.pdf'))
    if args.only:
        paths = [path for path in paths if path.name == args.only]
    if args.limit:
        paths = paths[:args.limit]
    engine = None
    results = []
    try:
        for path in paths:
            start = time.monotonic()
            record = dict(file=path.name, stage=args.stage, sample=args.sample)
            folder = output / path.stem
            folder.mkdir(exist_ok=True)
            try:
                if args.stage == 'extract':
                    from pdf_input import extract_pdf
                    from languages import resolve_language
                    text = extract_pdf(path)
                    assert text.strip(), 'No extracted text'
                    (folder / 'source.md').write_text(text)
                    record.update(chars=len(text), language=resolve_language(text), excerpts=len(cleanup_chunks(text)))
                elif args.stage == 'clean':
                    from worker import Cleaner
                    from languages import resolve_language
                    if engine is None:
                        engine = Cleaner(config)
                    text = (folder / 'source.md').read_text()
                    if args.sample:
                        text = cleanup_chunks(text)[0]
                    narration = plain_text(engine.clean(text, lambda *_: None, resolve_language(text)))
                    name = 'sample.txt' if args.sample else 'narration.txt'
                    (folder / name).write_text(narration)
                    plan = speech_plan(narration)
                    record.update(chars=len(narration), passages=len(plan), warnings=engine.warnings,
                                  model=config['llm'])
                else:
                    import soundfile as sf
                    from worker import Speaker, stitch
                    from languages import resolve_language
                    if engine is None:
                        engine = Speaker(config)
                    name = 'sample.txt' if args.sample else 'narration.txt'
                    text = (folder / name).read_text()
                    plan = speech_plan(text)
                    if args.sample:
                        plan = plan[:1]
                        # A short, sentence-ended audio sample, not an audiobook.
                        passage = plan[0]['text']
                        end = next((m.end() for m in BOUNDARY.finditer(passage) if m.end() >= 100), len(passage))
                        plan[0]['text'] = passage[:end]
                    parts = []
                    language = resolve_language(text)
                    for index, passage in enumerate(plan):
                        wave, rate = engine.speak(passage['text'], language)
                        target = folder / f'{"sample" if args.sample else "part"}-{index:04d}.flac'
                        sf.write(target, wave, rate)
                        parts.append(target)
                    target = folder / ('sample.flac' if args.sample else 'audio.flac')
                    stitch(parts, target)
                    record.update(passages=len(plan), seconds=sf.info(target).duration)
                record['ok'] = True
            except Exception as error:
                record.update(ok=False, error=str(error))
            record['elapsed'] = round(time.monotonic() - start, 2)
            if args.stage != 'extract':
                import torch
                record['peak_reserved_gib'] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            results.append(record)
            print(json.dumps(record), flush=True)
            suffix = '-sample' if args.sample else ''
            (output / f'{args.stage}{suffix}-results.json').write_text(json.dumps(results, indent=2))
    finally:
        del engine
    return int(any(not result['ok'] for result in results))


if __name__ == '__main__':
    sys.exit(main())
