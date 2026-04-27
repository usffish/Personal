"""
scraper — data-fetching modules for Movie Score Scraper.

Public API:
    get_omdb_data        — Metascore + IMDB rating from the OMDb JSON API
    get_metacritic_data  — critic review count (+ Metascore fallback) from Metacritic
    get_letterboxd_data  — average community rating from Letterboxd
    GeminiResolver       — last-resort slug/ID disambiguation via Gemini API
"""

from scraper.letterboxd_scraper import get_letterboxd_data
from scraper.metacritic_scraper import get_metacritic_data
from scraper.omdb_client import get_omdb_data
from scraper.gemini_resolver import GeminiResolver

__all__ = ["get_omdb_data", "get_metacritic_data", "get_letterboxd_data", "GeminiResolver"]
