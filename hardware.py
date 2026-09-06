"""Conservative single-document VRAM estimates; no inference imports in the GUI."""
import subprocess
import ctypes
import sys
from functools import cache


@cache
def virtual_metal():
    """Apple's virtual display GPU can advertise MPS but cannot run Qwen reliably."""
    if sys.platform != 'darwin':
        return False
    try:
        result = subprocess.run(['/usr/sbin/system_profiler', 'SPDisplaysDataType'],
                                capture_output=True, text=True, check=True, timeout=10)
        return 'paravirtual' in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False

# Includes weights, short-context generation, and a working-memory margin.
MODEL_VRAM = {
    "Qwen/Qwen3-0.6B": 2.0,
    "Qwen/Qwen3-1.7B": 5.0,
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": 3.0,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": 6.0,
}


def batch_size(gpu_bytes=None):
    """Six passages everywhere, except GPUs with less than 4 GiB."""
    return 1 if gpu_bytes is not None and gpu_bytes < 4 * 2**30 else 6


def gpu_memory():
    """Return the first NVIDIA GPU's total GiB, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=3)
        return float(result.stdout.splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        # The driver library can be available without the nvidia-smi utility.
        try:
            cuda = ctypes.CDLL('nvcuda.dll' if sys.platform == 'win32' else 'libcuda.so.1')
            cuda.cuInit.argtypes = [ctypes.c_uint]
            cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            cuda.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
            device, memory = ctypes.c_int(), ctypes.c_size_t()
            if cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(device), 0) == 0:
                if cuda.cuDeviceTotalMem_v2(ctypes.byref(memory), device) == 0:
                    return memory.value / 2**30
        except (OSError, AttributeError):
            pass
        return None


def model_label(name, capacity):
    estimate = MODEL_VRAM.get(name)
    if estimate is None:
        return name + " — VRAM unknown", False
    exceeds = capacity is not None and estimate > capacity
    suffix = " · exceeds GPU" if exceeds else ""
    return f"{name} — ~{estimate:g} GiB VRAM{suffix}", exceeds
