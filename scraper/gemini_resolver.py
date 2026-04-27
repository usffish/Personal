"""
gemini_resolver.py
==================
Uses the Gemini API to resolve the correct URL slug or identifier for a movie
on Metacritic, Letterboxd, and OMDb/IMDb when the local slug-guessing and
search fallbacks have already failed.

The resolver is intentionally narrow: it only asks Gemini for a slug string,
never for scores.  All score data still comes from the original sources.

Usage
-----
    from scraper.gemini_resolver import GeminiResolver

    resolver = GeminiResolver(api_key="YOUR_GEMINI_KEY")

    # Returns a slug string or None
    slug = resolver.resolve_metacritic_slug("Nirvana the Band the Show the Movie")
    slug = resolver.resolve_letterboxd_slug("Nirvana the Band the Show the Movie")
    imdb_id = resolver.resolve_imdb_id("Nirvana the Band the Show the Movie")

Environment variable
--------------------
    GEMINI_API_KEY — used when no api_key is passed to GeminiResolver().
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import so the module can be imported even when google-generativeai is
# not installed — callers that never instantiate GeminiResolver won't break.
_genai = None


def _import_genai():
    global _genai
    if _genai is not None:
        return _genai
    try:
        import google.generativeai as genai  # type: ignore
        _genai = genai
        return _genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for Gemini slug resolution. "
            "Install it with: pip install google-generativeai"
        ) from exc


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_METACRITIC_PROMPT = """\
I need the exact URL slug that Metacritic uses for the movie "{title}".

Metacritic movie pages follow this pattern:
  https://www.metacritic.com/movie/<slug>/

Rules Metacritic uses for slugs:
- All lowercase
- Spaces replaced with hyphens
- Leading articles ("the", "a", "an") are sometimes dropped
- Special characters and punctuation removed
- Accented characters converted to ASCII equivalents

Examples:
  "The Dark Knight"          -> dark-knight
  "Mulholland Drive"         -> mulholland-drive
  "Eternal Sunshine of the Spotless Mind" -> eternal-sunshine-of-the-spotless-mind
  "Se7en"                    -> se7en
  "2001: A Space Odyssey"    -> 2001-a-space-odyssey

Reply with ONLY the slug string, nothing else. No explanation, no URL, no quotes.
If you are not confident, reply with the word: unknown
"""

_LETTERBOXD_PROMPT = """\
I need the exact URL slug that Letterboxd uses for the movie "{title}".

Letterboxd film pages follow this pattern:
  https://letterboxd.com/film/<slug>/

Rules Letterboxd uses for slugs:
- All lowercase
- Spaces replaced with hyphens
- Special characters and punctuation removed
- Accented characters converted to ASCII equivalents
- Leading articles ("the", "a", "an") are kept (unlike Metacritic)
- When multiple films share a title, a year suffix is added: e.g. "the-fly-1986"
- Disambiguation suffixes like "-1" or "-2" are sometimes used

Examples:
  "The Dark Knight"          -> the-dark-knight
  "Mulholland Drive"         -> mulholland-drive
  "Nirvana the Band the Show the Movie" -> nirvana-the-band-the-show-the-movie
  "Se7en"                    -> se7en
  "2001: A Space Odyssey"    -> 2001-a-space-odyssey

Reply with ONLY the slug string, nothing else. No explanation, no URL, no quotes.
If you are not confident, reply with the word: unknown
"""

_IMDB_PROMPT = """\
I need the IMDb ID (tt-number) for the movie "{title}".

IMDb IDs follow this pattern: tt followed by 7 or 8 digits, e.g. tt0110912

Examples:
  "Pulp Fiction" (1994)      -> tt0110912
  "The Dark Knight" (2008)   -> tt0468569
  "Parasite" (2019)          -> tt6751668

Reply with ONLY the IMDb ID string (e.g. tt0110912), nothing else.
No explanation, no URL, no quotes.
If you are not confident, reply with the word: unknown
"""


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------

class GeminiResolver:
    """
    Resolves movie slugs/identifiers using the Gemini generative AI API.

    Instantiate once per process and reuse across all scraper calls.
    The model is loaded lazily on first use.
    """

    _MODEL_NAME = "gemini-2.5-flash-lite"  # fast, cheap, highest free tier limits

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key is required. Pass api_key= or set GEMINI_API_KEY."
            )
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is not None:
            return self._model
        genai = _import_genai()
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(self._MODEL_NAME)
        logger.debug("GeminiResolver: loaded model %s", self._MODEL_NAME)
        return self._model

    def _ask(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Gemini and return the stripped response text.

        Returns None on any API error or if the model replies "unknown".
        """
        try:
            model = self._get_model()
            response = model.generate_content(prompt)
            text = response.text.strip()
            logger.debug("GeminiResolver raw response: %r", text)
            if text.lower() == "unknown":
                return None
            return text
        except Exception as exc:
            logger.warning("GeminiResolver: API error — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public resolution methods
    # ------------------------------------------------------------------

    def resolve_metacritic_slug(self, title: str) -> Optional[str]:
        """
        Ask Gemini for the Metacritic URL slug for *title*.

        Returns a slug string (e.g. "dark-knight") or None.
        The returned value is validated to look like a plausible slug before
        being returned; garbage responses are discarded.
        """
        logger.info("GeminiResolver: resolving Metacritic slug for '%s'", title)
        prompt = _METACRITIC_PROMPT.format(title=title)
        slug = self._ask(prompt)
        return _validate_slug(slug)

    def resolve_letterboxd_slug(self, title: str) -> Optional[str]:
        """
        Ask Gemini for the Letterboxd URL slug for *title*.

        Returns a slug string (e.g. "the-dark-knight") or None.
        """
        logger.info("GeminiResolver: resolving Letterboxd slug for '%s'", title)
        prompt = _LETTERBOXD_PROMPT.format(title=title)
        slug = self._ask(prompt)
        return _validate_slug(slug)

    def resolve_imdb_id(self, title: str) -> Optional[str]:
        """
        Ask Gemini for the IMDb ID (tt-number) for *title*.

        Returns a string like "tt0110912" or None.
        """
        logger.info("GeminiResolver: resolving IMDb ID for '%s'", title)
        prompt = _IMDB_PROMPT.format(title=title)
        raw = self._ask(prompt)
        return _validate_imdb_id(raw)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_slug(value: Optional[str]) -> Optional[str]:
    """
    Return *value* if it looks like a valid URL slug, else None.

    A valid slug contains only lowercase letters, digits, and hyphens,
    is between 1 and 120 characters, and does not start or end with a hyphen.
    Rejects multi-word responses (spaces) and anything that looks like a URL.
    """
    if not value:
        return None
    # Strip surrounding whitespace and quotes the model might add
    value = value.strip().strip('"\'')
    # Reject if it contains spaces (model gave a sentence instead of a slug)
    if " " in value:
        logger.warning("GeminiResolver: slug response looks like prose, discarding: %r", value)
        return None
    # Reject if it looks like a full URL
    if value.startswith("http"):
        logger.warning("GeminiResolver: slug response is a URL, discarding: %r", value)
        return None
    # Must match slug pattern
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,118}[a-z0-9]?", value):
        logger.warning("GeminiResolver: slug response failed validation, discarding: %r", value)
        return None
    return value


def _validate_imdb_id(value: Optional[str]) -> Optional[str]:
    """
    Return *value* if it looks like a valid IMDb ID (tt + 7-8 digits), else None.
    """
    if not value:
        return None
    value = value.strip().strip('"\'')
    if re.fullmatch(r"tt\d{7,8}", value):
        return value
    logger.warning("GeminiResolver: IMDb ID response failed validation, discarding: %r", value)
    return None
