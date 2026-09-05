# PDF to Audio

<img src="assets/icon.svg" width="72" alt="PDF to Audio icon">

[![Installation tests](https://github.com/cdelv/PDF-to-Audio/actions/workflows/install-tests.yml/badge.svg)](https://github.com/cdelv/PDF-to-Audio/actions/workflows/install-tests.yml)
[![Downloads](https://img.shields.io/github/v/release/cdelv/PDF-to-Audio?include_prereleases&label=download)](https://github.com/cdelv/PDF-to-Audio/releases)
![Platforms: Linux, Windows, macOS](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)

Turn PDFs and text documents into spoken audio on your own computer. Drop files into the window, choose an output folder, and press **Create audio**.

- Convert multiple PDFs, TXT files, or Markdown documents in one batch.
- Narrate in each document's original language using a supplied or custom voice sample.
- Clean PDF text with Qwen3, with instructions to preserve figure and table captions while omitting table contents.
- Follow your desktop's light/dark appearance and manage voices, languages, and models through the GUI.
- Track elapsed time, cancel safely, and resume saved conversions—even after closing the app. Larger GPUs batch speech passages automatically.

Documents and voice recordings stay on your computer. The two default **0.6B models** download during setup—about **4 GB** combined—and then work offline. Optional **1.7B models** download only when selected and saved in Settings. Scanned PDFs need OCR first; complex documents may need a review of the extracted text.

## Installation

**One-click installers are not published yet.** For now, use the source-install instructions below with **Python 3.12** and **Git** installed. Dependencies stay in the app's private `.venv`; the first launch downloads any missing default models.

### Linux

The planned installer is a `.flatpakref` file. Until it is available, open a terminal and run:

```sh
git clone https://github.com/cdelv/PDF-to-Audio.git
cd PDF-to-Audio
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python install.py
.venv/bin/python app.py
```

After registration, you can open **PDF to Audio** from the application menu. On Ubuntu, creating the environment may require `sudo apt install python3.12-venv`.

#### NVIDIA / CUDA support

CPU mode needs no GPU driver. For NVIDIA acceleration, install a compatible host driver and run this from the app folder:

```sh
.venv/bin/python -m pip install -r requirements-cuda.txt
```

You do **not** need the full CUDA Toolkit. The Python dependencies supply the CUDA runtime libraries, but the app cannot install the host's kernel driver.

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

A native Windows installer is still in development. With Python 3.12 and Git installed, open **PowerShell** and run:

```powershell
git clone https://github.com/cdelv/PDF-to-Audio.git
cd PDF-to-Audio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe app.py
```

For NVIDIA acceleration, install a current compatible NVIDIA driver and replace `requirements-cpu.txt` with `requirements-cuda.txt` in the installation command. Otherwise, the app uses the CPU.

### macOS

A native Mac installer is still in development. The current setup has been tested on **Apple Silicon**; Intel Macs are not yet validated. With Python 3.12 and Git installed, open **Terminal** and run:

```sh
git clone https://github.com/cdelv/PDF-to-Audio.git
cd PDF-to-Audio
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

macOS currently uses CPU narration; Apple GPU acceleration is not yet supported by the app.
