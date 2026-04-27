# Movie Score Engine

A Python CLI tool that aggregates film scores from OMDb, Metacritic, and Letterboxd, applies Bayesian-motivated weighting and min-max normalization, and produces a composite ranking — all written back to Excel without touching the original file.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-3776AB?style=flat-square&logo=python&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-3776AB?style=flat-square&logo=python&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![Hypothesis](https://img.shields.io/badge/Hypothesis-3776AB?style=flat-square&logo=python&logoColor=white)

---

## What it does

For every title in a personal Movies.xlsx watchlist, the tool:

1. Fetches **Metascore** and **IMDB rating** from the [OMDb API](http://www.omdbapi.com/)
2. Scrapes **critic review count** (and Metascore fallback) from Metacritic
3. Scrapes **average community rating** from Letterboxd
4. Normalises all three scores column-wide using min-max scaling
5. Computes a **review-count-weighted composite score** grounded in Bayesian statistics
6. Writes results to Movies_updated.xlsx, leaving the original file untouched

---

## Highlights

- **Three-pass pipeline** — fetch → normalise → composite. Normalisation is intentionally separated from the fetch loop because min-max scaling requires the full column to be known before any single value can be computed. When --smart-update skips stable movies, their existing raw scores are still included in the normalisation batch so the full distribution is always used.
- **Bayesian-motivated weighting** — Metacritic's contribution to the composite scales with its critic review count, not a fixed coefficient. A score backed by 80 reviews carries more weight than the same score backed by 4. This is grounded in Laplace's rule of succession (see [Theory](#composite-score--theory) below).
- **Dynamic denominator** — missing scores are dropped from both numerator and denominator rather than substituted with zeros, preserving the relative weighting of whichever sources are available.
- **Resilient scraping** — all HTTP fetches retry up to 3 times with exponential back-off. Per-movie failures are logged and skipped; the rest of the batch continues.
- **Data safety** — existing cell values are never overwritten by a missing result. The input workbook is never modified.
- **Property-based test suite** — correctness properties (normalisation bounds, composite formula, ZeroDivisionError safety, global anchor computation) are verified with [Hypothesis](https://hypothesis.readthedocs.io/) across hundreds of generated inputs.
- **Smart scheduling** — --smart-update tracks score stability over time and skips movies that haven't changed, reducing unnecessary network requests on repeat runs.

---

## Project Structure

```
.
├── update_scores.py          # Orchestrator, three-pass pipeline, CLI entry point
├── requirements.txt
├── Movies.xlsx               # Input watchlist (user-provided, not committed)
├── Movies_updated.xlsx       # Generated output (not committed)
├── scraper/
│   ├── omdb_client.py        # OMDb API client — Metascore + IMDB rating
│   ├── metacritic_scraper.py # Scrapes critic review count (+ Metascore fallback)
│   └── letterboxd_scraper.py # Scrapes average community rating
└── tests/
    ├── test_omdb_client.py
    ├── test_metacritic_scraper.py
    ├── test_letterboxd_scraper.py
    ├── test_orchestrator.py
    ├── test_normalisation_properties.py  # Property: normalise_column bounds
    ├── test_omdb_properties.py           # Property: OMDb parsing round-trip
    ├── test_composite_properties.py      # Property: formula correctness + safety
    ├── test_scraper_properties.py        # Property: review count, rating range, back-off
    └── test_orchestrator_properties.py   # Property: input unchanged, output columns
```

---

## Composite score — theory

### The problem: can you trust a 100% rating?

Consider three sellers offering the same product at the same price:

| Seller | Rating | Reviews |
|--------|--------|---------|
| A | 100% positive | 10 |
| B | 96% positive | 50 |
| C | 93% positive | 200 |

Most people instinctively distrust the 100% rating — it comes from so few reviews that it feels fragile. But how do you make that intuition *quantitative*?

This is the central question in [3Blue1Brown's series on Bayesian statistics](https://www.youtube.com/watch?v=8idr1WZ1A7Q), and it is the theoretical foundation for how this project weights the Metacritic score.

### Laplace's rule of succession

When you observe p positive reviews out of n total, your best estimate of the true underlying success rate is not p/n but:

```
(p + 1) / (n + 2)
```

You pretend there were two extra reviews — one positive, one negative — before seeing any data. This is **Laplace's rule of succession** (18th century). It encodes a Bayesian prior of genuine uncertainty: the more real data accumulates, the less those two phantom reviews matter.

Applied to the sellers above:

| Seller | Adjusted estimate | |
|--------|------------------|-|
| A | (10 + 1) / (10 + 2) = **91.7%** | |
| B | (48 + 1) / (50 + 2) = **94.2%** | ← best choice |
| C | (186 + 1) / (200 + 2) = **92.6%** | |

The seller with the highest raw percentage is not the best choice. The one with the most *evidence* behind a strong rating wins.

### Applying this to Metacritic

A Metascore is a weighted average of critic reviews. A score of 85 from 6 critics and a score of 85 from 80 critics are not equally trustworthy.

This project applies the same logic: rather than giving Metacritic a fixed weight, its contribution to the composite is **scaled by its critic review count**. A Metascore backed by 80 reviews carries 80× the influence of a single fixed-weight term; one backed by 4 reviews carries only 4×.

### Step 1 — Min-max normalisation

The three sources use incompatible scales (0–100, 0–10, 0–5). Before combining them, each column is rescaled to [0, 1]:

```
st.X[i] = (X[i] − min(X)) / (max(X) − min(X))
```

min and max are computed across the entire batch after all fetches complete — not per-movie. This is why the pipeline separates fetching from normalisation. The best movie in each column maps to 1.0, the worst to 0.0, and everything else falls proportionally in between.

### Step 2 — Review-count-weighted composite

```
TRUE = ((st.Metacritic × Reviews) + st.Letterboxd + Global_Max + Global_Min + st.IMDB)
       / (Reviews + 4)
```

| Term | Weight | Rationale |
|------|--------|-----------|
| st.Metacritic × Reviews | Reviews | Metacritic's influence scales with critical coverage |
| st.Letterboxd | 1 | Fixed unit weight |
| st.IMDB | 1 | Fixed unit weight |
| Global_Max | 1 | Batch anchor — highest normalised value across all columns |
| Global_Min | 1 | Batch anchor — lowest normalised value across all columns |

The denominator is Reviews + 4 (4 fixed-weight terms plus the variable Metacritic weight).

### Dynamic denominator

Missing scores are dropped from both numerator and denominator, not substituted with zeros.

| Condition | Effect on denominator |
|-----------|----------------------|
| Reviews == 0 | Base denominator is 4 (Metacritic term dropped) |
| st.Letterboxd is None | − 1 |
| st.IMDB is None | − 1 |
| Global anchors unavailable | − 2 |
| All terms missing | Return None |

---

## Setup

**Requirements:** Python 3.10+, a free [OMDb API key](https://www.omdbapi.com/apikey.aspx)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OMDb API key
export OMDB_API_KEY=your_key_here
```

---

## Usage

```bash
# Update all movies
python update_scores.py

# Process a single movie
python update_scores.py --movie "Boogie Nights"

# Test with a random sample of 10
python update_scores.py --limit 10

# Use a custom input file
python update_scores.py --input my_list.xlsx

# Skip recently-stable movies
python update_scores.py --smart-update

# Prompt for missing values when scraping fails
python update_scores.py --manual

# Adjust request delay (default 1.0s between sources)
python update_scores.py --delay 2.0
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| --input PATH | Movies.xlsx | Path to the input workbook |
| --output PATH | \<stem\>_updated.xlsx | Path for the output workbook |
| --api-key KEY | — | OMDb API key (overrides OMDB_API_KEY env var) |
| --limit N | — | Pick N movies at random |
| --movie TITLE | — | Process a single movie by exact title |
| --delay SECS | 1.0 | Seconds between requests to each source |
| --verbose | off | Enable debug-level logging |
| --smart-update | off | Skip recently-stable movies |
| --manual | off | Prompt for missing values interactively |

---

## Input format

Place your watchlist in Movies.xlsx in the project root. The workbook must have a column named **Movies** with one title per row. All other columns are optional — the script adds any missing output columns automatically.

---

## Output columns

| Column | Description |
|--------|-------------|
| Metacritic | Metascore (0–100) — Metacritic scrape, falls back to OMDb |
| st.Metacritic | Min-max normalised Metascore (0.0–1.0) |
| Reviews | Critic review count from Metacritic |
| Letterboxd | Average community rating (0.0–5.0) |
| st.Letterboxd | Min-max normalised Letterboxd rating (0.0–1.0) |
| IMDB | IMDB rating (0.0–10.0) from OMDb |
| st.IMDB | Min-max normalised IMDB rating (0.0–1.0) |
| TRUE | Weighted composite score (0.0–1.0, rounded to 2 dp) |
| LastUpdated | ISO date of last successful fetch (YYYY-MM-DD) |
| StableWeeks | Consecutive weeks the composite stayed within ±0.05 |

---

## Running the tests

```bash
# Full test suite
python -m pytest

# Verbose output
python -m pytest -v

# Property-based tests only
python -m pytest tests/test_normalisation_properties.py tests/test_composite_properties.py tests/test_scraper_properties.py tests/test_omdb_properties.py tests/test_orchestrator_properties.py
```

---

## Author

**Ismail Jhaveri** — [LinkedIn](https://www.linkedin.com/in/ismail-jhaveri-2021/) · [ismailj@usf.edu](mailto:ismailj@usf.edu)
