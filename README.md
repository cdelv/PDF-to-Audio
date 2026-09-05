# PDF to Audio

A native Linux desktop app that turns documents into locally generated speech. Drop in several PDFs or text files, choose a voice sample, and create an audio file for each document. No browser, server, or cloud inference is required.

## Install

The intended end-user installer is **PDF-to-Audio.flatpak**. Flatpak packaging and clean-container installation tests are currently being validated; do not treat the source checkout as a finished one-click installer yet.

Once a bundle is available:

1. Make sure Flatpak support is installed through your distribution's Software application. GNOME Software and KDE Discover may require their Flatpak plugin.
2. Double-click **PDF-to-Audio.flatpak**, then select **Install**. Allow the Software application to download the GNOME runtime if needed.
3. Open **PDF to Audio** from the application menu.

The bundle is designed to contain its private Python environment, inference dependencies, CUDA user-space libraries, four selectable Qwen models, and a sample voice. No pip commands, environment editing, model accounts, or API keys are needed by the end user. Models make the download substantially larger than the GUI itself. Internet access is needed to install missing Flatpak runtimes; narration runs offline.

**An NVIDIA host driver is required for GPU acceleration.** Flatpak can supply matching user-space driver extensions, but it cannot install or replace the host's kernel driver. Neither can an ordinary Docker container. Without a compatible driver, Automatic uses the CPU. Install the host driver using your distribution's supported driver manager; a reboot or Secure Boot enrollment may be required. See [NVIDIA's prerequisites](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [Flatpak driver extensions](https://docs.flatpak.org/en/latest/extension.html).

The package targets x86-64 Linux. GNOME, KDE, and COSMIC can run the GTK/libadwaita interface. Window identity follows the Wayland and X11 desktop-entry conventions. GNOME and KDE sessions have not yet been live-tested here.

## Use

1. Drag files into the window or choose **Choose files**. Multiple PDFs and mixed file types can share one batch.
2. Select an output folder.
3. Optionally open **Settings** to choose a voice, edit its transcript or the PDF cleanup instructions, select languages, or choose models.
4. Press **Create audio**. **Play** opens the recording; **Files** opens its output folder.

Each document gets a separate, uniquely named output directory and an `audio.flac` recording. Cancellation keeps completed passages. A failed document does not stop the rest of the batch. Retrying starts a new output directory; automatic resume is not implemented.

## Features

- PDFs are extracted through Microsoft MarkItDown using its PDFMiner backend, with layout handling for scientific-paper columns and embedded text. Qwen3 prepares the result for narration.
- TXT, Markdown, RST, CSV, and LOG files go directly to speech after basic Markdown cleanup. These files must be UTF-8; they bypass PDF cleanup instructions.
- PDF cleanup preserves existing figure, chart, image, diagram, and table captions. It is instructed to omit table contents, not turn rows and values into prose. Structured Markdown/HTML tables are filtered before cleanup; unstructured table recognition still depends on the model.
- Documents stay in their original languages. Automatic detection operates independently for each file in a multilingual batch.
- The voice recording/transcript language is independent of the document language. The recording determines the cloned voice and accent; there is no separate reference-language conditioning argument in Qwen's Base API.
- A voice sample, its exact transcript, and an editable cleanup prompt are ordinary bundled assets. Voice samples should contain 2–30 seconds of clear speech.
- The window and settings follow system light/dark preferences, including changes while the app is open.
- Settings show estimated VRAM beside each model. Models exceeding the detected card's capacity are marked in red. Custom model paths show an unknown estimate.

Supported speech languages: English, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Detection can be ambiguous for short text; a manual language choice corrects detection without requesting translation. Mixed-language passages within one document are not guaranteed to have ideal pronunciation.

## Hardware and memory

For a **4 GB NVIDIA card**, select the **0.6B cleanup model** and the **0.6B Base speech model**, with Processor set to **auto**. Models load sequentially, not together. Automatic chooses CPU when the selected model's estimated requirement exceeds available GPU memory. Older CUDA cards without bfloat16 support use float16.

| Model | Estimated VRAM |
| --- | --- |
| Qwen3-0.6B cleanup | 2 GiB |
| Qwen3-1.7B cleanup | 5 GiB |
| Qwen3-TTS-12Hz-0.6B-Base | 3 GiB |
| Qwen3-TTS-12Hz-1.7B-Base | 6 GiB |

These are conservative working estimates, not guarantees. Other GPU applications, voice length, generation length, and driver overhead affect memory use. CPU fallback uses system RAM and is slower; 16 GB system RAM is a practical starting point for the larger models. A physical 4 GB GPU has not been tested; allocation-capped tests on the development GPU are documented separately.

## Output and limitations

Output folders retain `source.md` for PDFs, `narration.txt`, `passages.json`, `languages.json`, numbered FLAC passages, and the final `audio.flac`.

PDF cleanup uses small excerpts and bounded generation. Uncertain cleanup is retried in smaller excerpts. If it remains unreliable, the original excerpt is retained and `warnings.txt` explains what needs review; the completed row points out the warning. This avoids silently accepting truncated, translated, or obviously invented narration. Retained excerpts can still contain layout artifacts or unstructured table data.

Scanned PDFs require OCR first. The app does not include OCR or image understanding and cannot recover captions that extraction misses. Scientific equations, complex tables, headers, references, and multi-column layouts may need review. The small cleanup model can omit or alter details; always review important material in `narration.txt`.

Documents are planned in sections of at most 100,000 characters, then much smaller speech passages of at most 700 characters. Speech boundaries use sentence endings or paragraph breaks, not arbitrary character cuts. An overlong sentence without an acceptable boundary produces an error rather than silently cutting it. Audio is joined by streaming from disk, with short edge fades to reduce clicks. Generated prosody can still vary between passages.

Qwen3-TTS has token-based context limits, **not a 500,000-character speech capacity**. The 100,000-character sections are planning units, not individual model calls. See [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Build the Flatpak — maintainers

End users should receive the built `.flatpak` file, not run these commands. A build machine needs Python 3, Flatpak, internet access, and substantial free disk space for runtimes, CUDA wheels, models, the build tree, and the bundle.

```sh
git clone https://github.com/cdelv/PDF-to-Audio.git
cd PDF-to-Audio
python3 tools/build_flatpak.py
```

The script installs the GNOME 49 build/runtime dependencies noninteractively, creates the private environment inside the build sandbox, installs pinned top-level inference packages, checks their dependencies, fetches pinned model revisions, and exports `dist/PDF-to-Audio.flatpak`. It never installs host kernel drivers. A read-only existing Hugging Face cache can be reused with `--reuse-model-cache /path/to/hub`; the resulting bundle contains independent copies, not links to that cache.

For unattended installation of a built bundle on a machine with Flatpak:

```sh
flatpak install --user --noninteractive -y dist/PDF-to-Audio.flatpak
flatpak run io.github.pdftoaudio.Desktop
```

The sandbox has no inference-time network permission. App files and the private environment are mounted read-only. It uses the GNOME runtime, GPU-device access, document read access, and Music-folder write access; file selection uses desktop portals. Flatpak keeps settings under `~/.var/app/io.github.pdftoaudio.Desktop/`. Removing the package does not require editing system Python.

### Clean-container installation test

The Dockerfile starts with a minimal Debian installation, not the developer's Python environment. Use Docker or the Docker-compatible Podman engine. The test mounts only the bundle and test scripts, not host models, Python packages, or GPU devices.

```sh
podman build -f packaging/Dockerfile.test -t pdf-to-audio-install-test .
podman run --rm --security-opt seccomp=unconfined \
  -v "$PWD/dist:/bundle:ro,z" -v "$PWD/packaging:/test:ro,z" \
  pdf-to-audio-install-test
```

The relaxed container seccomp policy permits nested Flatpak/bubblewrap namespaces; it does not grant host GPU access. The test installs noninteractively, checks bundled models and dependencies offline, runs short CPU cleanup and voice cloning, and launches the GUI in a virtual display. It does not emulate a complete GNOME/KDE desktop, Secure Boot, or a driver-free GPU.

## Development and testing

For an already configured source checkout, `python3 install.py` registers it in the application menu. It is not a fresh-machine dependency installer. The GUI uses distribution GTK 4, libadwaita, and PyGObject; inference uses `.venv/bin/python` in isolated mode. The interpreter path is internal and is not editable in Settings.

```sh
.venv/bin/python -m unittest discover -s tests -v
/usr/bin/python3 tests/appearance_smoke.py
.venv/bin/python tests/smoke.py
/usr/bin/python3 tests/gui_smoke.py
.venv/bin/python tests/multilingual_smoke.py
.venv/bin/python tests/corpus_check.py /path/to/papers --stage extract
.venv/bin/python tests/corpus_check.py /path/to/papers --stage clean --sample --small --cap-gib 3
.venv/bin/python tests/corpus_check.py /path/to/papers --stage speak --sample --cap-gib 3
```

Corpus tests write only into ignored `test-output/corpus`, leaving the input PDFs untouched. `--sample` tests excerpts, not complete audiobooks. Omit it for whole-document conversion. `--cap-gib` caps PyTorch allocations; it is not physical GPU emulation and does not include every driver allocation.
