# PDF to Audio

<img src="assets/icon.svg" width="72" alt="PDF to Audio icon">

[![Installation tests](https://github.com/cdelv/PDF-to-Audio/actions/workflows/install-tests.yml/badge.svg)](https://github.com/cdelv/PDF-to-Audio/actions/workflows/install-tests.yml)
[![Downloads](https://img.shields.io/github/v/release/cdelv/PDF-to-Audio?include_prereleases&label=download)](https://github.com/cdelv/PDF-to-Audio/releases)
![Platforms: Linux, Windows, macOS](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)

Turn PDFs and text documents into spoken audio on your own computer. Drop files into the window, choose an output folder, and press **Create audio**.

- Convert multiple PDFs, TXT files, or Markdown documents in one batch.
- Narrate in each document's original language using a supplied or custom voice sample.
- Clean every document with Qwen3 before narration, preserving figure and table captions while omitting table contents. PDFs first go through text extraction.
- Follow your desktop's light/dark appearance and manage voices, languages, and models through the GUI.
- Track elapsed time, cancel safely, and resume saved conversions—even after closing the app. Cleanup and speech use batches of six on CPU and GPU, except GPUs with 4 GiB of VRAM or less, which process one passage at a time. Both models can use all CPU threads, including during GPU inference.

Documents and voice recordings stay on your computer. The two default **0.6B models** download during setup—about **4 GB** combined—and then work offline. Optional **1.7B models** download only when selected and saved in Settings. Scanned PDFs need OCR first; complex documents may need a review of the extracted text.

## Installation

Download the installer for your desktop below. No Python, Git, or terminal setup is required. Open the app after installing and let setup finish downloading its private runtime and the two default models. Allow several GB of free space for the runtime in addition to the models; CUDA needs more space than CPU.

One installer per OS supports all of the app's backends for that platform: **CPU and NVIDIA CUDA on Linux/Windows; CPU and Apple Metal on macOS**. Automatic mode selects an available GPU. You can also choose the processor in Settings; missing CUDA dependencies download automatically. Setup never modifies system Python or installs host GPU drivers.

### Linux

- **Ubuntu 24.04 or newer, x86-64:** download the [DEB installer](https://github.com/cdelv/PDF-to-Audio/releases/download/v0.2.0/PDF-to-Audio-Linux-amd64.deb).
- **Fedora 44, x86-64:** download the [RPM installer](https://github.com/cdelv/PDF-to-Audio/releases/download/v0.2.0/PDF-to-Audio-Linux-x86_64.rpm).

Open the downloaded package with your software installer, then open **PDF to Audio** from the application menu. GNOME, KDE, and COSMIC use the same app.

#### NVIDIA / CUDA support

CPU mode needs no GPU driver. NVIDIA acceleration needs a host driver compatible with **CUDA 12.8**. The app downloads the CUDA runtime libraries; you do **not** need the full CUDA Toolkit. If your GPU already works with a compatible driver, leave it installed.

**Ubuntu (apt):** install Ubuntu's driver-selection tool, then let it choose the driver for your GPU. See the [Ubuntu driver guide](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/).

```sh
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

**Fedora 44, x86-64 (dnf):** for Turing or newer GPUs, including GeForce RTX cards, use NVIDIA's repository and open kernel driver. Do not mix this with an existing RPM Fusion or manual driver installation; use your existing provider's update instructions instead. See the [NVIDIA Fedora guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/fedora.html).

```sh
sudo dnf install dnf5-plugins kernel-devel-matched kernel-headers
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install nvidia-open
sudo reboot
```

Save your work before rebooting. Secure Boot may require enrolling a signing key; follow your distribution's instructions rather than disabling it. After reboot, `nvidia-smi` should list your GPU. Other Fedora releases and older GPUs need their matching driver instructions.

### Windows

Download and run the [Windows x64 installer](https://github.com/cdelv/PDF-to-Audio/releases/download/v0.2.0/PDF-to-Audio-Windows-x64-Setup.exe), then open **PDF to Audio** from Start. NVIDIA acceleration requires a compatible NVIDIA driver; CPU mode needs none.

The installer is unsigned, so Windows may show a security warning. Verify that you downloaded it from this repository before approving it; do not disable system security.

### macOS

Download the [Apple Silicon DMG](https://github.com/cdelv/PDF-to-Audio/releases/download/v0.2.0/PDF-to-Audio-macOS-arm64.dmg), open it, and drag **PDF to Audio** into **Applications**. Intel Macs are not included in this release.

The private PyTorch runtime includes experimental **Apple Metal (MPS)** acceleration, selected automatically on supported physical GPUs. No separate GPU driver is needed. CPU narration is tested; Metal inference is not yet validated on a physical Mac. Virtual Apple GPUs use CPU because Qwen failed on the test runner's paravirtual GPU. CPU is always selectable in Settings.

The app is not notarized. macOS may require approval in **System Settings → Privacy & Security** after your first attempt to open it. Only approve the download from this repository; do not disable Gatekeeper globally.
