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
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Model configurations ordered by quality (best first) for cycling
# When one model hits its RPD limit, it moves to the next
# Format: (model_name, rpm, rpd)
# Note: Using models that work with the current google-generativeai library API
_GEMINI_MODELS = [
    ("gemini-3.1-flash-lite-preview", 4000, 150_000),  # Tier 1: 4K RPM, 150K RPD
    ("gemini-2.5-flash-lite",         4000, 999_999),  # Tier 1: 4K RPM, unlimited RPD
    ("gemini-3-flash-preview",        1000,  10_000),  # Tier 1: 1K RPM, 10K RPD
]

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
    
    Includes built-in rate limiting and automatic model cycling:
    - Uses best model first (gemini-3.1-flash-lite)
    - When a model hits its RPD limit, automatically switches to the next model
    - Cycles through models in order of quality
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key is required. Pass api_key= or set GEMINI_API_KEY."
            )
        
        # Model cycling state
        self._model_index = 0  # Start with best model
        self._models = _GEMINI_MODELS
        self._model = None  # lazy init
        
        # Per-model rate limiting: list of (minute_requests, day_requests) deques
        self._minute_requests = [deque() for _ in self._models]
        self._day_requests = [deque() for _ in self._models]
        
        # Warning throttling
        self._last_rpm_warning = 0
        self._last_rpd_warning = 0

    @property
    def _current_model_name(self) -> str:
        return self._models[self._model_index][0]

    @property
    def _current_rpm(self) -> int:
        return self._models[self._model_index][1]

    @property
    def _current_rpd(self) -> int:
        return self._models[self._model_index][2]

    def _get_model(self):
        if self._model is not None:
            return self._model
        genai = _import_genai()
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(self._current_model_name)
        logger.debug("GeminiResolver: loaded model %s", self._current_model_name)
        return self._model

    def _switch_to_next_model(self) -> bool:
        """
        Switch to the next available model.
        Returns True if switched, False if no more models available.
        """
        if self._model_index < len(self._models) - 1:
            self._model_index += 1
            self._model = None  # Force reload of model
            logger.warning(
                "GeminiResolver: switching to model %s (RPD limit reached on %s)",
                self._current_model_name,
                self._models[self._model_index - 1][0]
            )
            return True
        return False

    def _wait_for_rate_limit(self) -> None:
        """
        Wait if necessary to respect RPM and RPD limits for current model.
        Uses a sliding window approach to track requests.
        """
        idx = self._model_index
        now = time.time()
        
        # Clean up old timestamps (older than 1 minute)
        while self._minute_requests[idx] and self._minute_requests[idx][0] < now - 60:
            self._minute_requests[idx].popleft()
        
        # Clean up old timestamps (older than 1 day)
        while self._day_requests[idx] and self._day_requests[idx][0] < now - 86400:
            self._day_requests[idx].popleft()
        
        # Check RPM limit
        if len(self._minute_requests[idx]) >= self._current_rpm:
            oldest = self._minute_requests[idx][0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                if now - self._last_rpm_warning > 60:
                    logger.warning(
                        "GeminiResolver: RPM limit reached (%d/min) on %s, waiting %.1fs",
                        self._current_rpm, self._current_model_name, wait_time
                    )
                    self._last_rpm_warning = now
                time.sleep(wait_time)
                now = time.time()
                while self._minute_requests[idx] and self._minute_requests[idx][0] < now - 60:
                    self._minute_requests[idx].popleft()
        
        # Check RPD limit - if hit, try switching to next model
        if len(self._day_requests[idx]) >= self._current_rpd:
            if self._switch_to_next_model():
                # New model has different limits, recurse to check its limits
                self._wait_for_rate_limit()
                return
            else:
                # All models exhausted, wait for the first model's day to reset
                oldest = self._day_requests[0][0]
                wait_time = 86400 - (now - oldest) + 1
                if now - self._last_rpd_warning > 3600:
                    logger.warning(
                        "GeminiResolver: ALL models at RPD limit, waiting %.0fs for reset",
                        wait_time
                    )
                    self._last_rpd_warning = now
                time.sleep(wait_time)
                # Reset to best model after waiting
                self._model_index = 0
                self._model = None
                return
        
        # Record this request
        self._minute_requests[idx].append(now)
        self._day_requests[idx].append(now)

    def _ask(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Gemini and return the stripped response text.
        Automatically cycles through models if rate limited.
        Returns None on any API error or if the model replies "unknown".
        """
        # Try each model in order
        for attempt in range(len(self._models)):
            # Wait for rate limits before making request
            self._wait_for_rate_limit()
            
            try:
                model = self._get_model()
                response = model.generate_content(prompt)
                text = response.text.strip()
                logger.debug("GeminiResolver: %s returned %r", self._current_model_name, text)
                if text.lower() == "unknown":
                    return None
                return text
            except Exception as exc:
                error_str = str(exc).lower()
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    logger.warning(
                        "GeminiResolver: rate limit error on %s, trying next model",
                        self._current_model_name
                    )
                    if self._switch_to_next_model():
                        continue
                    else:
                        # All models exhausted
                        logger.error("GeminiResolver: all models at rate limit")
                        return None
                logger.warning("GeminiResolver: API error on %s — %s", self._current_model_name, exc)
                return None
        
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
    # Must match slug pattern: starts and ends with alnum, hyphens only in middle
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        logger.warning("GeminiResolver: slug response failed validation, discarding: %r", value)
        return None
    # Check length
    if len(value) > 120:
        logger.warning("GeminiResolver: slug too long (%d chars), discarding: %r", len(value), value)
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
