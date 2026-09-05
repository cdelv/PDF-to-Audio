"""Offline text-language detection, independent of the reference voice."""
import re
from core import LANGUAGES

CODES = dict(en="English", es="Spanish", fr="French", de="German", it="Italian",
             pt="Portuguese", ru="Russian", ja="Japanese", ko="Korean",
             **{"zh-cn": "Chinese", "zh-tw": "Chinese"})


def detect_language(text):
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    # Profiles ship with the package. No model, network, or GPU is used.
    DetectorFactory.seed = 0
    sample = text[:12000]
    if sum(c.isalpha() for c in sample) < 20:
        return "Auto"
    # The statistical profiles can confuse Chinese and Korean on short text.
    # Their writing systems, and Japanese kana, provide stronger evidence.
    letters = sum(c.isalpha() for c in sample)
    if len(re.findall(r'[\u3040-\u30ff]', sample)) >= max(2, letters * 0.1):
        return "Japanese"
    if len(re.findall(r'[\uac00-\ud7af]', sample)) > letters * 0.5:
        return "Korean"
    if len(re.findall(r'[\u3400-\u9fff]', sample)) > letters * 0.75:
        return "Chinese"
    try:
        result = detect_langs(sample)[0]
    except LangDetectException:
        return "Auto"
    # Short/ambiguous text falls back to the speech model's native Auto mode.
    if result.prob < 0.9:
        return "Auto"
    return CODES.get(result.lang, result.lang)


def resolve_language(text, requested="Auto", subject="Document"):
    if requested not in ["Auto", *LANGUAGES]:
        raise ValueError(f"Unknown {subject.lower()} language: {requested}.")
    # A document override corrects detection; it is never a translation target.
    if requested != "Auto" and subject == "Document":
        return requested
    detected = detect_language(text)
    if requested != "Auto":
        if detected != "Auto" and detected != requested:
            raise ValueError(f"{subject} appears to be {detected}, but Settings specifies {requested}. "
                             "Choose its original language or Automatic; language settings do not translate text.")
        return requested
    if detected not in ["Auto", *LANGUAGES]:
        raise ValueError(f"{subject} appears to use language code '{detected}', which Qwen3-TTS does not support. "
                         "If detection is incorrect, choose the original language in Settings.")
    return detected


def check_cleanup_language(source_language, cleaned):
    detected = detect_language(cleaned)
    if source_language != "Auto" and detected != "Auto" and source_language != detected:
        raise ValueError(f"PDF cleanup changed the language from {source_language} to {detected}. "
                         "Review the cleanup prompt; documents must keep their original language.")
