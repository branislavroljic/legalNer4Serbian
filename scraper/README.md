# Montenegro Court Scraper

A pure Playwright-based scraper for extracting court decisions from the Montenegro court system website (https://sudovi.me/sdvi/odluke).

## Features

- ✅ **SPA-friendly**: Handles single-page applications properly
- ✅ **New window handling**: Automatically captures and processes judgment pages that open in new windows
- ✅ **Cookie consent**: Automatically handles GDPR cookie dialogs
- ✅ **Advanced search**: Uses the "Napredna pretraga" (Advanced Search) functionality
- ✅ **Batch processing**: Extracts multiple court decisions in one run
- ✅ **File output**: Saves each judgment as a separate text file with metadata

## Installation

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Install Playwright browsers**:
   ```bash
   uv run playwright install
   ```

## Usage

Run the scraper:
```bash
uv run python montenegro_scraper.py
```

The scraper will:
1. Navigate to the Montenegro court decisions website
2. Handle cookie consent automatically
3. Perform an advanced search for "kriv je" (guilty)
4. Extract the first 100 search results (configurable)
5. Click on each result to open the judgment page
6. Extract the full judgment text
7. Save each judgment to a separate file in `../judgments/`

## Output

The scraper creates text files in the `../judgments/` directory named `judgment_[case_number].txt` containing:
- Court name
- Case number
- Date
- Full judgment text

Example: `../judgments/judgment_K_224_2011.txt`

## Configuration

You can modify the scraper behavior by editing `montenegro_scraper.py`:
- `self.search_term`: Change the search term (default: "kriv je")
- `self.max_results`: Change number of results to process (default: 100)
- `headless=False`: Set to `True` to run without visible browser
- `output_dir`: Change output directory (default: "../judgments")

## Requirements

- Python 3.12+
- Playwright
