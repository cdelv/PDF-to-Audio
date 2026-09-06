## PDF to Audio 0.2.0

Convert PDFs and text documents into local, voice-cloned narration.

- Six-passage speech batches on CPU and GPU; GPUs below 4 GiB use one passage.
- Both models can use all CPU threads, including during GPU inference.
- Cleanup and speech models run in separate stages and unload between them; audio decoding is sequential to reduce peak VRAM.
- Elapsed conversion timer, saved queues, and resumable cleanup/audio generation.
- Original-language narration for mixed-language document batches, with scientific cleanup instructions for acronyms, equations, and figure/table captions.
- Every file goes through Qwen3 cleanup before narration; PDFs first go through text extraction. Table contents are omitted while captions are preserved.
- First-launch setup shows download percentage and transferred bytes, with resumable runtime downloads.
- Default 0.6B models download during first-launch setup; optional 1.7B models download from Settings.

### Install

- **Ubuntu 24.04 or newer:** open the `.deb` with your software installer.
- **Fedora 44:** open the `.rpm` with your software installer.
- **Windows x64:** run `PDF-to-Audio-Windows-x64-Setup.exe`.
- **macOS Apple Silicon:** open the `.dmg` and drag PDF to Audio into Applications.

Then open PDF to Audio and let its setup download its private runtime and the two default models (about 4 GB for models, plus runtime dependencies). No Python installation or terminal commands are required. Internet is required for setup, not narration.

The same Linux/Windows installer supports **CPU and NVIDIA CUDA 12.8**. Setup detects an available NVIDIA driver and downloads CUDA dependencies; choosing CUDA in Settings also installs them if missing. Installing a driver later does not require reinstalling the app. CPU mode works without GPU drivers. A compatible host NVIDIA driver is required for GPU acceleration and is not installed by the app.

The Apple Silicon installer includes **CPU and experimental Apple Metal (PyTorch MPS)** support. Automatic mode selects Metal on supported physical GPUs, using float32 and CPU fallback for unsupported operators. Both models release their GPU caches between stages. CPU narration is tested; Metal hardware inference is not yet validated on a physical Mac. Qwen failed on the hosted runner's Apple Paravirtual GPU, so virtual GPUs now use CPU and explicit Metal selection reports a clear error. AMD and Intel GPU acceleration on Linux/Windows is not supported.

Windows/macOS files are unsigned and macOS is not notarized, so the OS may display a security warning or require explicit approval through its security settings. Do not disable system security globally.

Release validation: Linux and Windows installers are tested automatically. Final macOS installer validation is deferred to manual testing; the preceding macOS build passed CPU narration, but failed Metal inference on the virtual GPU.

Scanned PDFs need OCR first. Review important narration for extraction/cleanup errors. Resume applies to conversions started with this version and requires unchanged document, voice, model, language, and cleanup prompt. Completed passages are kept; an interrupted batch is regenerated.
