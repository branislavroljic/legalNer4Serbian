# Legal NER for Serbian - Montenegro Court Analysis

A complete pipeline for scraping Montenegro court decisions and performing Named Entity Recognition (NER) analysis on Serbian legal texts.

## Project Overview

This project consists of three independent components:

1. **🕷️ Scraper** - Extracts court decisions from Montenegro court system
2. **📁 Judgments** - Shared data directory for scraped court decisions
3. **🧠 NER** - Named Entity Recognition analysis of legal texts

## Project Structure

```
legalNer4Serbian/
├── README.md                    # This overview
├── scraper/                     # 🕷️ Court decisions scraper
│   ├── montenegro_scraper.py    # Main scraper script
│   ├── pyproject.toml          # Scraper dependencies
│   └── README.md               # Scraper documentation
├── judgments/                   # 📁 Shared data directory
│   └── judgment_*.txt          # Scraped court decisions
└── ner/                        # 🧠 NER analysis project
    ├── pyproject.toml          # NER dependencies
    ├── README.md               # NER documentation
    ├── notebooks/              # Jupyter notebooks
    ├── src/                    # Python modules
    └── models/                 # Trained models
```

## Quick Start

### 1. Scrape Court Decisions

```bash
cd scraper
uv sync
uv run playwright install
uv run python montenegro_scraper.py
```

This will scrape court decisions and save them to `../judgments/`

### 2. Analyze with NER

```bash
cd ner
uv sync
uv run jupyter lab
```

This will start Jupyter Lab for NER analysis of the scraped judgments.

## Components

### 🕷️ Scraper
- **Purpose**: Extract court decisions from https://sudovi.me/sdvi/odluke
- **Technology**: Pure Playwright (SPA-friendly)
- **Output**: Text files in `../judgments/`
- **Features**: Cookie handling, new window management, batch processing

### 📁 Judgments Directory
- **Purpose**: Shared data storage for court decisions
- **Format**: Plain text files with metadata headers
- **Naming**: `judgment_[case_number].txt`
- **Content**: Court name, case number, date, full judgment text

### 🧠 NER Analysis
- **Purpose**: Extract legal entities from judgment texts
- **Technology**: spaCy, Transformers, scikit-learn
- **Input**: Files from `../judgments/`
- **Output**: Extracted entities, trained models, analysis reports

## Dependencies

Each component has its own `pyproject.toml`:
- **Scraper**: Playwright
- **NER**: Jupyter, spaCy, Transformers, pandas, scikit-learn

## Usage Workflow

1. **Scrape** → Run scraper to collect court decisions
2. **Analyze** → Use NER notebooks to extract legal entities
3. **Iterate** → Refine models and scrape more data as needed

## Requirements

- Python 3.12+
- uv package manager