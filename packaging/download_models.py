"""Unattended native setup: download only the defaults unless explicitly selected."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_store import DEFAULT_MODELS, ensure_models


if __name__ == '__main__':
    ensure_models(sys.argv[1:] or DEFAULT_MODELS,
                  lambda event, **data: print(data['message'], flush=True))
