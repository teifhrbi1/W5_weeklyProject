import re
from pathlib import Path

TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL_RE = re.compile(r"\u0640+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NON_ARABIC_SYMBOLS_RE = re.compile(r"[^\u0600-\u06FF\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

def remove_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text)

def remove_tatweel(text: str) -> str:
    return TATWEEL_RE.sub("", text)

def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)

def remove_non_arabic_symbols(text: str) -> str:
    return NON_ARABIC_SYMBOLS_RE.sub(" ", text)

def normalize_spaces(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()

def clean_arabic_text(text: str) -> str:
    text = str(text)
    text = remove_urls(text)
    text = remove_tashkeel(text)
    text = remove_tatweel(text)
    text = remove_non_arabic_symbols(text)
    text = normalize_spaces(text)
    return text

def load_stopwords(path: Path) -> set[str]:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w:
            words.add(w)
    return words

def remove_stopwords(text: str, stopwords: set[str]) -> str:
    tokens = str(text).split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = normalize_spaces(text)
    return text
