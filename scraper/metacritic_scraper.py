"""
Metacritic scraper using requests + BeautifulSoup.
Fetches the critic review count and, when available, the Metascore for a film.

Metacritic film pages follow the pattern:
  https://www.metacritic.com/movie/<slug>/

When a film has only 1–3 critic reviews, Metacritic does not display an
aggregate Metascore.  In that case this module fetches the individual review
scores from the critic-reviews sub-page and averages them to produce a
synthetic Metascore.
"""

import json
import logging
import math
import re
import time
import unicodedata
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.metacritic.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

_MOVIE_URL = "https://www.metacritic.com/movie/{slug}/"
_REVIEWS_URL = "https://www.metacritic.com/movie/{slug}/critic-reviews/"
_SEARCH_URL = "https://www.metacritic.com/search/{query}/?category=2"  # category 2 = movies

# Metacritic only shows an aggregate Metascore once a film has at least this
# many critic reviews.  Below this threshold we average the individual scores.
_MIN_REVIEWS_FOR_AGGREGATE = 4


def _slugify(text: str) -> str:
    """Convert a title to a Metacritic-style URL slug (strips leading articles)."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Remove articles at the start (Metacritic sometimes drops "the", "a", "an")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text


def _slugify_with_article(text: str) -> str:
    """Slugify keeping leading articles."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text


def _fetch(url: str, retries: int = 3, backoff: float = 2.5, 
           rate_limiter=None, domain: str = "metacritic.com") -> Optional[BeautifulSoup]:
    """GET a URL and return a BeautifulSoup object, with retry logic and exponential back-off."""
    for attempt in range(retries):
        try:
            # Wait for rate limiter if provided
            if rate_limiter:
                rate_limiter.wait_if_needed(domain)
            
            resp = SESSION.get(url, timeout=15)
            
            # Check for rate limit errors
            if resp.status_code == 429 or resp.status_code == 503:
                logger.warning("Metacritic: Rate limit error HTTP %s for %s", resp.status_code, url)
                if rate_limiter:
                    rate_limiter.increase_delay(domain)
                # Wait longer before retry
                time.sleep(backoff * (2 ** attempt) * 2)
                continue
                
            if resp.status_code == 200:
                # Success - gradually decrease delay if using rate limiter
                if rate_limiter and attempt == 0:  # Only on first successful attempt
                    rate_limiter.decrease_delay(domain)
                return BeautifulSoup(resp.text, "lxml")
            if resp.status_code == 404:
                return None
            logger.warning("Metacritic: HTTP %s for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logger.warning("Metacritic: request error (%s) for %s", exc, url)
        # Only sleep between attempts, not after the last one
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    return None


def _extract_review_count(soup: BeautifulSoup) -> Optional[int]:
    """Extract the critic review count from a Metacritic movie page."""
    # Try JSON-LD structured data first
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0]
            agg = data.get("aggregateRating", {})
            count = agg.get("reviewCount") or agg.get("ratingCount")
            if count is not None:
                return int(count)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Fallback: HTML selectors for review count
    count_selectors = [
        "span[class*='count'] a",
        "div[class*='summary'] span.count a",
        "span.based_on",
    ]
    for sel in count_selectors:
        tag = soup.select_one(sel)
        if tag:
            nums = re.findall(r"\d+", tag.get_text())
            if nums:
                return int(nums[0])

    return None


def _extract_aggregate_score(soup: BeautifulSoup) -> Optional[int]:
    """
    Extract the aggregate Metascore from a Metacritic movie page.

    Returns None when the score is not present (e.g. fewer than 4 reviews).
    """
    # JSON-LD is the most reliable source
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0]
            agg = data.get("aggregateRating", {})
            value = agg.get("ratingValue")
            if value is not None:
                score = round(float(value))
                return max(0, min(100, score))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Fallback: look for a prominent score element in the HTML.
    # Metacritic renders the Metascore inside elements like:
    #   <span ...>95</span>  near text "METASCORE" or "Metascore"
    for tag in soup.find_all(string=re.compile(r"\bMetascore\b", re.IGNORECASE)):
        parent = tag.parent
        # Walk up a few levels looking for a sibling/child with a bare integer
        for _ in range(4):
            if parent is None:
                break
            nums = re.findall(r"\b(\d{1,3})\b", parent.get_text())
            for n in nums:
                val = int(n)
                if 0 <= val <= 100:
                    return val
            parent = parent.parent

    return None


def _extract_individual_scores(soup: BeautifulSoup) -> list:
    """
    Parse individual critic scores from a Metacritic critic-reviews page.

    Metacritic renders each score as text matching "Metascore N out of 100"
    (visible in the rendered page).  Returns a list of ints (0–100).
    """
    scores = []

    # Pattern seen in rendered HTML: "Metascore 88 out of 100"
    pattern = re.compile(r"Metascore\s+(\d{1,3})\s+out\s+of\s+100", re.IGNORECASE)
    for text in soup.find_all(string=pattern):
        for match in pattern.finditer(text):
            val = int(match.group(1))
            if 0 <= val <= 100:
                scores.append(val)

    return scores


def _search_for_slug(title: str, rate_limiter=None) -> Optional[str]:
    """Search Metacritic and return the slug of the best matching movie."""
    query = re.sub(r"\s+", "%20", title.strip())
    url = _SEARCH_URL.format(query=query)
    soup = _fetch(url, rate_limiter=rate_limiter, domain="metacritic.com")
    if soup is None:
        return None

    title_lower = title.lower().strip()

    # Search result links look like /movie/<slug>/
    for link in soup.select("a[href^='/movie/']"):
        href = link["href"]
        parts = href.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "movie":
            slug = parts[1]
            link_text = link.get_text(strip=True).lower()
            if link_text == title_lower:
                return slug

    # Fallback: first movie link
    first = soup.select_one("a[href^='/movie/']")
    if first:
        href = first["href"]
        parts = href.strip("/").split("/")
        if len(parts) >= 2:
            return parts[1]

    return None


def get_metacritic_data(title: str, year: Optional[int] = None, resolver=None, 
                        rate_limiter=None) -> dict:
    """
    Fetch critic review count and Metascore for a movie from Metacritic.

    When the film has 4+ reviews the aggregate Metascore is read directly from
    the movie page.  When it has 1–3 reviews Metacritic does not publish an
    aggregate, so this function fetches the critic-reviews sub-page, parses
    each individual score, and returns their rounded average.

    Args:
        title:    Movie title.
        year:     Optional release year (unused in URL construction but kept for
                  API compatibility).
        resolver: Optional GeminiResolver instance.  When all local slug
                  candidates and the site search have failed, the resolver is
                  asked for the correct slug as a last resort.

    Returns:
        dict with keys:
            review_count (int):         >= 0; 0 when not found or on error.
            metascore    (int | None):  0–100; None when unavailable.
    """
    result: dict = {"review_count": 0, "metascore": None}

    # Build candidate slugs to try
    slug_no_article = _slugify(title)
    slug_with_article = _slugify_with_article(title)

    slugs = [slug_no_article]
    if slug_with_article != slug_no_article:
        slugs.append(slug_with_article)

    soup = None
    matched_slug = None

    for slug in slugs:
        url = _MOVIE_URL.format(slug=slug)
        soup = _fetch(url, rate_limiter=rate_limiter, domain="metacritic.com")
        if soup is not None:
            matched_slug = slug
            break

    if soup is None:
        # Fall back to Metacritic site search
        logger.info("Metacritic: direct slug failed for '%s', trying search", title)
        matched_slug = _search_for_slug(title)
        if matched_slug:
            url = _MOVIE_URL.format(slug=matched_slug)
            soup = _fetch(url, rate_limiter=rate_limiter, domain="metacritic.com")

    if soup is None and resolver is not None:
        # Last resort: ask Gemini for the correct slug
        logger.info("Metacritic: site search failed for '%s', asking Gemini", title)
        gemini_slug = resolver.resolve_metacritic_slug(title)
        if gemini_slug:
            url = _MOVIE_URL.format(slug=gemini_slug)
            soup = _fetch(url, rate_limiter=rate_limiter, domain="metacritic.com")
            if soup is not None:
                matched_slug = gemini_slug
                logger.info("Metacritic: Gemini resolved slug '%s' for '%s'", gemini_slug, title)

    if soup is None:
        logger.warning("Metacritic: could not find page for '%s'", title)
        return result

    # --- review count ---
    count = _extract_review_count(soup)
    if count is not None:
        result["review_count"] = count

    # --- metascore ---
    if result["review_count"] >= _MIN_REVIEWS_FOR_AGGREGATE:
        # Aggregate score should be present on the main page
        score = _extract_aggregate_score(soup)
        if score is not None:
            result["metascore"] = score
        else:
            logger.debug(
                "Metacritic: aggregate score not found on main page for '%s' "
                "(%d reviews)",
                title,
                result["review_count"],
            )
    elif result["review_count"] > 0:
        # 1–3 reviews: average the individual scores from the reviews sub-page
        logger.info(
            "Metacritic: %d review(s) for '%s' — averaging individual scores",
            result["review_count"],
            title,
        )
        reviews_url = _REVIEWS_URL.format(slug=matched_slug)
        reviews_soup = _fetch(reviews_url, rate_limiter=rate_limiter, domain="metacritic.com")
        if reviews_soup is not None:
            scores = _extract_individual_scores(reviews_soup)
            if scores:
                avg = round(sum(scores) / len(scores))
                result["metascore"] = max(0, min(100, avg))
                logger.info(
                    "Metacritic: averaged %d individual score(s) → %d for '%s'",
                    len(scores),
                    result["metascore"],
                    title,
                )
            else:
                logger.warning(
                    "Metacritic: could not parse individual scores for '%s'", title
                )

    return result


def get_review_count(title: str, year: Optional[int] = None, resolver=None) -> int:
    """
    Backward-compatible wrapper around get_metacritic_data.

    Returns only the critic review count.  Prefer get_metacritic_data for new
    call sites so that the Metascore is also available.
    """
    return get_metacritic_data(title, year, resolver=resolver)["review_count"]
