"""
Letterboxd scraper using requests + BeautifulSoup.
Fetches the average community rating and number of ratings for a film.

Letterboxd film pages follow the pattern:
  https://letterboxd.com/film/<slug>/
where the slug is a URL-friendly version of the title.
"""

import json
import logging
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
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

_SEARCH_URL = "https://letterboxd.com/search/films/{query}/"
_FILM_URL = "https://letterboxd.com/film/{slug}/"


def _slugify(text: str) -> str:
    """Convert a title to a Letterboxd-style URL slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text


def _fetch(url: str, retries: int = 3, backoff: float = 2.0,
           rate_limiter=None, domain: str = "letterboxd.com") -> Optional[BeautifulSoup]:
    """GET a URL and return a BeautifulSoup object, with retry logic and exponential back-off."""
    for attempt in range(retries):
        try:
            # Wait for rate limiter if provided
            if rate_limiter:
                rate_limiter.wait_if_needed(domain)
            
            resp = SESSION.get(url, timeout=15)
            
            # Check for rate limit errors
            if resp.status_code == 429 or resp.status_code == 503:
                logger.warning("Letterboxd: Rate limit error HTTP %s for %s", resp.status_code, url)
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
            logger.warning("Letterboxd: HTTP %s for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logger.warning("Letterboxd: request error (%s) for %s", exc, url)
        # Only sleep between attempts, not after the last one
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    return None


def _parse_rating_from_soup(soup: BeautifulSoup) -> Optional[float]:
    """Extract the average rating from a Letterboxd film page."""
    # Primary: <meta itemprop="ratingValue" content="X">
    meta = soup.find("meta", itemprop="ratingValue")
    if meta and meta.get("content"):
        try:
            return max(0.0, min(5.0, float(meta["content"])))
        except ValueError:
            pass

    # Secondary: JSON-LD aggregateRating block
    script = soup.find("script", type="application/ld+json")
    if script:
        try:
            data = json.loads(script.string or "")
            agg = data.get("aggregateRating", {})
            val = agg.get("ratingValue")
            if val is not None:
                return max(0.0, min(5.0, float(val)))
        except (json.JSONDecodeError, TypeError):
            pass

    # Tertiary: <meta name="twitter:data2" content="3.85 out of 5">
    twitter_meta = soup.find("meta", attrs={"name": "twitter:data2"})
    if twitter_meta and twitter_meta.get("content"):
        content = twitter_meta["content"]
        match = re.match(r"([\d.]+)\s+out\s+of", content, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(5.0, float(match.group(1))))
            except ValueError:
                pass

    # Quaternary: <span class="average-rating"> or <span class="display-rating">
    for selector in ("span.average-rating", "span.display-rating"):
        tag = soup.select_one(selector)
        if tag:
            text = tag.get_text(strip=True)
            try:
                return max(0.0, min(5.0, float(text)))
            except ValueError:
                pass

    return None


def _parse_review_count_from_soup(soup: BeautifulSoup) -> Optional[int]:
    """Extract the number of ratings from a Letterboxd film page."""
    meta = soup.find("meta", itemprop="ratingCount")
    if meta and meta.get("content"):
        try:
            return int(meta["content"])
        except ValueError:
            pass

    script = soup.find("script", type="application/ld+json")
    if script:
        try:
            data = json.loads(script.string or "")
            agg = data.get("aggregateRating", {})
            count = agg.get("ratingCount")
            if count is not None:
                return int(count)
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _search_for_slug(title: str, rate_limiter=None) -> Optional[str]:
    """Search Letterboxd and return the slug of the best matching film."""
    query = re.sub(r"\s+", "+", title.strip())
    url = _SEARCH_URL.format(query=query)
    soup = _fetch(url, rate_limiter=rate_limiter, domain="letterboxd.com")
    if soup is None:
        return None

    # Search results list items contain links like /film/<slug>/
    results = soup.select("ul.results li.film-detail")
    if not results:
        # Fallback selector
        results = soup.select("li.film-detail")

    title_lower = title.lower().strip()
    for item in results:
        link = item.select_one("a[href^='/film/']")
        if not link:
            continue
        film_title_tag = item.select_one("h2.film-title, .film-title")
        film_title = film_title_tag.get_text(strip=True).lower() if film_title_tag else ""
        href = link["href"]  # e.g. /film/boogie-nights/
        slug = href.strip("/").split("/")[-1]

        if film_title == title_lower:
            return slug

    # If no exact match, return the first result's slug
    first = soup.select_one("a[href^='/film/']")
    if first:
        href = first["href"]
        return href.strip("/").split("/")[-1]

    return None


def _candidate_slugs(title: str, year: Optional[int] = None) -> list:
    """
    Return an ordered list of slug candidates to try for a given title.

    Letterboxd commonly uses:
      - plain slug:            the-dark-knight
      - slug with year:        the-dark-knight-2008
      - slug with suffix -1:   the-dark-knight-1
      - slug with suffix -2:   the-dark-knight-2
    """
    base = _slugify(title)
    candidates = [base]
    if year:
        candidates.append(f"{base}-{year}")
    # Disambiguation suffixes Letterboxd uses when titles clash
    candidates.append(f"{base}-1")
    candidates.append(f"{base}-2")
    return candidates


def get_letterboxd_data(title: str, year: Optional[int] = None, resolver=None, 
                        rate_limiter=None) -> dict:
    """
    Fetch Letterboxd average rating and rating count for a movie.

    Args:
        title:    Movie title.
        year:     Optional release year — used to try a year-suffixed slug first.
        resolver: Optional GeminiResolver instance.  When all local slug
                  candidates and the site search have failed, the resolver is
                  asked for the correct slug as a last resort.

    Returns:
        dict with keys: rating (float|None), rating_count (int|None), url (str|None)
    """
    result = {"rating": None, "rating_count": None, "url": None}

    # Try each slug candidate in order before falling back to search
    for slug in _candidate_slugs(title, year):
        url = _FILM_URL.format(slug=slug)
        soup = _fetch(url, rate_limiter=rate_limiter, domain="letterboxd.com")
        if soup is not None:
            result["url"] = url
            result["rating"] = _parse_rating_from_soup(soup)
            result["rating_count"] = _parse_review_count_from_soup(soup)
            return result

    # All direct slugs failed — fall back to Letterboxd search
    logger.info("Letterboxd: direct slugs failed for '%s', trying search", title)
    slug = _search_for_slug(title, rate_limiter=rate_limiter)
    if slug is not None:
        url = _FILM_URL.format(slug=slug)
        soup = _fetch(url, rate_limiter=rate_limiter, domain="letterboxd.com")
        if soup is not None:
            result["url"] = url
            result["rating"] = _parse_rating_from_soup(soup)
            result["rating_count"] = _parse_review_count_from_soup(soup)
            return result

    # Site search also failed — ask Gemini as a last resort
    if resolver is not None:
        logger.info("Letterboxd: site search failed for '%s', asking Gemini", title)
        gemini_slug = resolver.resolve_letterboxd_slug(title)
        if gemini_slug:
            url = _FILM_URL.format(slug=gemini_slug)
            soup = _fetch(url, rate_limiter=rate_limiter, domain="letterboxd.com")
            if soup is not None:
                logger.info("Letterboxd: Gemini resolved slug '%s' for '%s'", gemini_slug, title)
                result["url"] = url
                result["rating"] = _parse_rating_from_soup(soup)
                result["rating_count"] = _parse_review_count_from_soup(soup)
                return result

    logger.warning("Letterboxd: could not find '%s'", title)
    return result
