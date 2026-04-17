# Movie Score Scraper

A command-line tool that fetches critic and audience scores for a personal movie list and writes the results to a new Excel workbook.

For each movie in `Movies.xlsx` it retrieves:
- **Metascore** and **IMDB rating** from the [OMDb API](http://www.omdbapi.com/)
- **Metascore** and **critic review count** scraped from Metacritic (preferred over OMDb when available)
- **Average community rating** scraped from Letterboxd

It then applies min-max normalisation across the full dataset and computes a weighted composite score, writing everything to `Movies_updated.xlsx` without touching the original file.

---

## Requirements

- Python 3.10+
- A free OMDb API key — register at [omdbapi.com](https://www.omdbapi.com/apikey.aspx)

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

The OMDb API key can be provided in two ways (the CLI flag takes precedence):

```bash
# Option A — environment variable (recommended)
export OMDB_API_KEY=your_key_here

# Option B — CLI flag
python update_scores.py --api-key your_key_here
```

---

## Input file

Place your movie list in `Movies.xlsx` in the project root. The file must have a sheet with a column named **`Movies`** containing one title per row. All other columns are optional — the script will add any missing output columns automatically.

---

## Usage

```bash
# Update all movies (reads Movies.xlsx, writes Movies_updated.xlsx)
python update_scores.py

# Specify a custom input file
python update_scores.py --input my_list.xlsx

# Specify a custom output path
python update_scores.py --output results.xlsx

# Pick 10 random movies to process
python update_scores.py --limit 10

# Process a single movie by exact title
python update_scores.py --movie "Boogie Nights"

# Slow down requests (seconds between each source, default 1.0)
python update_scores.py --delay 2.0

# Enable debug logging
python update_scores.py --verbose

# Skip movies whose scores have been stable for a while
python update_scores.py --smart-update

# Prompt to enter scores manually when scraping fails
python update_scores.py --manual

# Combine flags
python update_scores.py --smart-update --manual --delay 2.0
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | `Movies.xlsx` | Path to the input workbook |
| `--output PATH` | `<input_stem>_updated.xlsx` | Path for the output workbook |
| `--api-key KEY` | — | OMDb API key (overrides `OMDB_API_KEY` env var) |
| `--limit N` | — | Pick N movies at random to process |
| `--movie TITLE` | — | Process a single movie by exact title |
| `--delay SECS` | `1.0` | Seconds to wait between requests to each source |
| `--verbose` | off | Enable debug-level logging |
| `--smart-update` | off | Skip recently-stable movies (see below) |
| `--manual` | off | Prompt for missing values when scraping fails |

---

## Data preservation

Existing values in the workbook are **never overwritten by a missing result**. If a scraper returns nothing for a field (or the whole movie fails to fetch), the cell keeps whatever value it had before. Use `--manual` to fill in gaps interactively instead of leaving them blank.

---

## Manual entry (`--manual`)

When `--manual` is set, after all network fetches complete the script pauses and prompts you for any values it couldn't find:

- **Partial data** — if one or more fields came back empty for a movie, only those fields are prompted.
- **Complete failure** — if the movie couldn't be fetched at all, all four fields are prompted.

Press **Enter** on any prompt to skip that field and keep the existing workbook value. If you skip every field for a failed movie, that movie is left unchanged.

```
  ── Manual entry for: Nirvana the Band the Show the Movie ──
  (Press Enter to skip a field and leave it unchanged)

  Metascore (0-100): 72
  IMDB rating (0.0-10.0): 7.4
  Critic review count (0+):
  Letterboxd rating (0.0-5.0): 3.9
```

---

## Smart update (`--smart-update`)

Tracks how stable each movie's composite score has been over time and skips movies that don't need refreshing yet:

- After each successful fetch, `LastUpdated` is set to today's date and `StableWeeks` is incremented if the composite score changed by ≤ 0.05, or reset to 0 if it changed more.
- On the next run with `--smart-update`, a movie is skipped if fewer than `StableWeeks × 7` days have passed since `LastUpdated`.
- Movies with any missing score are **always** fetched regardless of stability.

---

## Output columns

The output workbook contains the original data plus these columns:

| Column | Description |
|---|---|
| `Metacritic` | Metascore (0–100) — scraped from Metacritic, falls back to OMDb |
| `st.Metacritic` | Min-max normalised Metascore (0.0–1.0) |
| `Reviews` | Critic review count scraped from Metacritic |
| `Letterboxd` | Average community rating (0.0–5.0) from Letterboxd |
| `st.Letterboxd` | Min-max normalised Letterboxd rating (0.0–1.0) |
| `IMDB` | IMDB rating (0.0–10.0) from OMDb |
| `st.IMDB` | Min-max normalised IMDB rating (0.0–1.0) |
| `TRUE` | Weighted composite score (rounded to 2 dp) |
| `LastUpdated` | ISO date of last successful fetch (`YYYY-MM-DD`) |
| `StableWeeks` | Consecutive weeks the composite score stayed within ±0.05 |

### Composite formula

```
TRUE = ((st.Metacritic × Reviews) + st.Letterboxd + Global_Max + Global_Min + st.IMDB)
       / (Reviews + 4)
```

- `Global_Max` and `Global_Min` are the highest and lowest normalised values across all three score columns for the current batch.
- Missing scores are dropped from both numerator and denominator (dynamic denominator) rather than substituted with zeros.

---

## Syncing the updated file back to Movies.xlsx

After reviewing `Movies_updated.xlsx`, make it the new baseline so the next run builds on top of it (preserving `StableWeeks`, `LastUpdated`, and any scores already fetched):

```bash
# Optional: keep a backup of the original
cp Movies.xlsx Movies_backup.xlsx

# Promote the updated file
cp Movies_updated.xlsx Movies.xlsx
```

The next `python update_scores.py` will then read from the promoted file. If you're using `--smart-update`, the stability counters carry over correctly.

> **Note:** close the file in Excel before copying — the `~$Movies_updated.xlsx` lock file means Excel still has it open.

---

## How it works

The script runs a three-pass pipeline to ensure normalisation is always column-wide:

1. **Pass 1 — Fetch**: retrieve raw scores for every movie from all three sources.
2. **Manual entry** *(optional, `--manual`)*: prompt for any values that couldn't be fetched.
3. **Pass 2 — Normalise**: apply min-max normalisation across the full batch for each score column.
4. **Pass 3 — Composite**: compute global anchors, then calculate the composite score per movie.

Normalisation and composite calculation never happen inside the per-movie fetch loop.

---

## Running the tests

```bash
# Run the full test suite
python -m pytest

# Run a specific test file
python -m pytest tests/test_omdb_client.py

# Run with verbose output
python -m pytest -v

# Run only property-based tests
python -m pytest tests/test_normalisation_properties.py tests/test_composite_properties.py tests/test_scraper_properties.py tests/test_omdb_properties.py tests/test_orchestrator_properties.py
```

---

## Project structure

```
.
├── update_scores.py          # Orchestrator and CLI entry point
├── requirements.txt
├── Movies.xlsx               # Your input file (not included)
├── Movies_updated.xlsx       # Generated output (not committed)
├── scraper/
│   ├── omdb_client.py        # OMDb API client (Metascore + IMDB rating)
│   ├── metacritic_scraper.py # Scrapes critic review count and Metascore
│   ├── letterboxd_scraper.py # Scrapes average community rating
│   └── imdb_scraper.py       # Legacy — not used, do not extend
└── tests/
    ├── test_omdb_client.py
    ├── test_metacritic_scraper.py
    ├── test_letterboxd_scraper.py
    ├── test_orchestrator.py
    ├── test_normalisation_properties.py
    ├── test_omdb_properties.py
    ├── test_composite_properties.py
    ├── test_scraper_properties.py
    └── test_orchestrator_properties.py
```
