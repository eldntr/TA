from functools import lru_cache
from lingua import Language, LanguageDetectorBuilder
from phonemizer.backend import EspeakBackend

class Phonemizer:
    def __init__(self):
        self.languages = [Language.ENGLISH, Language.INDONESIAN]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.backend_en = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        self.backend_id = EspeakBackend(language='id', preserve_punctuation=True, with_stress=True)

    @lru_cache(maxsize=100_000)
    def detect_lang(self, word: str) -> str:
        result = self.detector.detect_language_of(word)
        if result is None:
            return "id"
        return "en" if result == Language.ENGLISH else "id"

    @lru_cache(maxsize=100_000)
    def __call__(self, word: str):
        lang = self.detect_lang(word)
        backend = self.backend_en if lang == "en" else self.backend_id
        phon = backend.phonemize([word], strip=True)[0]
        return phon

phonemizer_instance = Phonemizer()
