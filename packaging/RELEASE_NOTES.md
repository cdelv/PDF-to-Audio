## PDF to Audio 0.2.0 — preview

Convert PDFs and text documents into local, voice-cloned narration.

- Automatic speech batching on compatible GPUs, with memory-aware fallback.
- Elapsed conversion timer, saved queues, and resumable cleanup/audio generation.
- Default 0.6B models download during first-launch setup; optional 1.7B models download from Settings.

### Install

- **Ubuntu 24.04 or newer:** open the `.deb` with your software installer.
- **Fedora 44:** open the `.rpm` with your software installer.
- **Windows x64:** run `PDF-to-Audio-Windows-x64-Setup.exe`.
- **macOS Apple Silicon:** open the `.dmg` and drag PDF to Audio into Applications.

Then open PDF to Audio and let its setup download the two default models (about 4 GB). No Python installation is required.

These first native installers contain **CPU-only inference dependencies**. GPU acceleration and batching are available in the CUDA-enabled source installation; CUDA-enabled native packaging is not included in this preview. The native Windows/macOS files are unsigned and macOS is not notarized, so the OS may display a security warning or require explicit approval through its security settings. Do not disable system security globally.

Scanned PDFs need OCR first. Review important narration for extraction/cleanup errors. Resume applies to conversions started with this version and requires unchanged document, voice, model, language, and cleanup prompt. Completed passages are kept; an interrupted batch is regenerated.
