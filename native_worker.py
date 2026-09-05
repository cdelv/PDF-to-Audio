"""Console helper for a windowed frozen GUI; preserves JSON pipes on Windows."""
import sys

if __name__ == '__main__':
    if '--check' in sys.argv or '--self-test' in sys.argv:
        from native_checks import check
        check(inference='--self-test' in sys.argv)
    elif '--setup-models' in sys.argv:
        from model_store import DEFAULT_MODELS, ensure_models
        ensure_models(DEFAULT_MODELS, lambda event, **data: print(data['message'], flush=True))
    else:
        from worker import main
        sys.exit(main())
