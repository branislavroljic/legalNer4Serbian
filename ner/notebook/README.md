## Methods and Data for Named Entity Recognition (NER) on Serbian Legal Documents

This document presents the Data and Methods used to construct, train, and evaluate a Named Entity Recognition (NER) system for Serbian legal judgments. It is written for thesis-ready use, providing methodological justifications and implementation references to the project’s notebooks and shared modules.


## 1. Data

### 1.1 Web Scraping (Montenegro court decisions)
- Implementation: The repository implements a pure Playwright-based scraper (not Scrapy-Playwright) in `scraper/montenegro_scraper.py` with a companion cleaner in `scraper/css_cleaner.py`. The scraper targets the Montenegro court decisions portal at https://sudovi.me/sdvi/odluke.
- SPA interaction pattern: The site is a Single Page Application (SPA). The scraper uses Playwright to:
  1) Navigate to the decisions page and wait for network idle.
  2) Accept the GDPR cookie dialog when present.
  3) Switch to the “Napredna pretraga” (Advanced search) tab.
  4) Open the “Vrsta odluke” combobox and select “Presuda”.
  5) Fill the search input with the query string.
  6) Click the “Pretraga” button.
  7) Wait for results and iterate result anchors.
  8) For each result, capture the newly opened window, wait for load, extract the judgment article text, close the window, and return to results view.
- Search queries: The default implementation uses the decision type Presuda (judgment) and the keyword query "kriv je" (is guilty).
- Extraction targets: Judgment text is extracted from `article[data-v-7a7c2765]` (fallback to a generic `article` selector). Metadata such as case number and decision date are read from result cards before opening each judgment.
- Output: One text file per judgment is written to the configured output directory (default resolved to `../mlm`). Files contain textual content; additional metadata is logged in the console during scraping.
- Rationale: A browser-automation approach (Playwright) robustly handles SPA state, tabbed interfaces, dynamic dropdowns, and new-window navigation, which are brittle under pure HTTP/HTML scraping.

Note on tooling consistency: The thesis requirements mention “Scrapy Playwright”. This repository’s working implementation is a pure Playwright scraper. The description above documents the actual code for methodological fidelity.


### 1.2 Data Cleaning
- Objectives: Remove HTML/CSS artifacts, normalize whitespace, and restore paragraph structure to improve downstream tokenization and annotation quality.
- Regex-based routines:
  - `scraper/css_cleaner.py` defines a `CSSCleaner` class with compiled regex patterns to strip CSS class definitions, HTML tags, CSS properties/values/units, and known CSS keywords. Post-processing ensures single spacing and collapses multiple blank lines.
  - `montenegro_scraper.py` additionally includes text filtering utilities to drop lines that look like CSS/formatting artifacts (e.g., `.page`, color codes, typeface tokens), and heuristic paragraph breaks (adds blank lines after tokens like “PRESUDU”, “OKRIVLJENI”, “Kriv je”, etc.).
- Paragraph restoration: After cleaning, lines are re-joined with deliberate double newlines for sections (judgment headings, parties, operative part) to preserve rhetorical structure beneficial for sentence segmentation and entity annotation.
- Text normalization: Trim leading/trailing spaces, compress internal whitespace, and unify line breaks. Normalization is conservative to avoid altering legal terms or numeric expressions critical to later entity extraction.


### 1.3 Annotation Process
- Tooling: LabelStudio was used for manual annotation of entities in cleaned judgments. The export JSON is stored in `ner/annotations.json` (and variants). Exact LabelStudio source files can be downloaded via `ner/download_labelstudio_files.py` to avoid offset mismatches.
- Conversion to BIO: `ner/shared/data_processing.py` provides `LabelStudioToBIOConverter` to map LabelStudio spans into token-level BIO tags. It prioritizes `file_upload` entries from LabelStudio’s export to ensure byte-identical text for span offsets; falls back to local judgments if necessary.
- Entity inventory (implemented; total = 16):
  - Institutions: COURT, REGISTRAR
  - Participants: DEFENDANT, PROSECUTOR, JUDGE
  - Outcomes and case metadata: CRIMINAL_ACT, PROVISION, PROVISION_MATERIAL, PROVISION_PROCEDURAL, VERDICT, SANCTION, SANCTION_TYPE, SANCTION_VALUE, PROCEDURE_COSTS, DECISION_DATE, CASE_NUMBER
- Inter-annotator agreement (IAA):
  - Methodology: For production-quality annotation, span-level agreement should be quantified using entity-level F1 and Cohen’s κ (on span presence by type). Disagreements are adjudicated to a gold set. Overlapping/nested spans are resolved per guidelines (priority by legal salience; no crossing spans). While the repository focuses on conversion/consistency tooling, the thesis’ IAA procedure should document annotator counts, calibration exercises, agreement scores (per-entity and overall), and adjudication rules.


### 1.4 Corpus Statistics
- Size: 225 annotated judgments (see `ner/notebook/bertic.ipynb` logs: “Loaded 225 annotated documents … Converted 225 examples to BIO”).
- Tokens: 232,475 labeled tokens across all examples (same notebook logs).
- Splitting:
  - Main experiments use 5-fold cross-validation (K=5) with approximately 45 documents per fold.
  - After sliding windows (max_length=512, stride=128), per-fold chunk counts are around 1,845 training chunks and 500 validation chunks (illustrative values from notebook prints).
  - When a fixed split is needed (e.g., for certain baselines or non-CV reports), a 70/15/15 train/val/test split is used consistently with shared utilities.
- Entity distribution: The distribution is imbalanced, with PROVISION_* (material/procedural) relatively frequent and certain roles (e.g., REGISTRAR) sparse. Class-imbalance is addressed in training via class weights (see Methods).


### 1.5 End-to-End Pipeline Visualization

```mermaid
flowchart LR
    A[Scraping: Playwright on sudovi.me] --> B[Cleaning: CSS/HTML removal + paragraph restoration]
    B --> C[Annotation: LabelStudio (BIO conversion)]
    C --> D[Modeling: Tokenization + Sliding Windows]
    D --> E[Training/Evaluation: CV, metrics]
```


## 2. Methods (Models and Training)

Implementation sources: Notebooks in `ner/notebook/` (bertic.ipynb, bertic_class_weights.ipynb, bertic_crf.ipynb, xlm_r_bertic.ipynb, gliner_zero_shot.ipynb) and shared modules in `ner/shared/` (config.py, dataset.py, model_utils.py, evaluation.py, data_processing.py). Domain-adaptive pretraining (DAPT) is implemented in `ner/serbian_legal_ner_pipeline_dapt_mlm*.ipynb`.

### 2.1 Model Architectures

1) Multilingual BERT (mBERT) – baseline
- Rationale: Transformer-based contextual encoders capture long-range dependencies and subword morphology, crucial for inflected South Slavic languages. mBERT provides a multilingual baseline to contextualize the gains from domain/language specialization.
- Limitation: Limited exposure to Serbian legal domain, potentially weaker representation of legal jargon and citation patterns.

2) BERTić (BCSm-BERTić)
- Description: A South Slavic transformer pretrained on Bosnian/Croatian/Serbian/Montenegrin corpora. Better lexical coverage and subword segmentation for Serbian legal text.
- Expectation: Consistently outperforms mBERT on Serbian legal NER due to domain/language proximity and pretraining corpora.
- Implementation: `ner/notebook/bertic.ipynb` (5-fold CV pipeline) with shared utilities.

3) BERT + CRF (Conditional Random Field)
- Motivation: Sequence labeling benefits from structured decoding; a CRF layer enforces BIO transition constraints (e.g., I-X cannot follow O without B-X) and models label dependencies.
- Architecture: BERT encoder → token-level linear layer → CRF decoder (see `ner/notebook/bertic_crf.ipynb`, implemented with `torchcrf` and a custom `BertCrfForTokenClassification`).
- Effect: Typically improves entity boundary consistency and yields gains on span-level F1 for structured legal text.

4) XLM-R-BERTić (classla/xlm-r-bertic)
- Description: Leverages XLM-RoBERTa’s multilingual pretraining (SentencePiece) with Serbian specialization. Useful for cross-lingual generalization and robust subword coverage.
- Implementation: `ner/notebook/xlm_r_bertic.ipynb` (5-fold CV).

5) Domain-Adaptive Pretraining (DAPT) via MLM
- Motivation: For under-resourced domains/languages (e.g., Serbian legal), adapting the base encoder on unlabeled in-domain text via Masked Language Modeling (MLM) often yields larger NER gains than architectural tweaks.
- Objective: Standard MLM (random token masking with percentage p; prediction over masked tokens). Corpus: unlabeled judgments collected by the scraper (e.g., notebook log “Found 849 text files for MLM pretraining”).
- Process: `ner/serbian_legal_ner_pipeline_dapt_mlm*.ipynb` trains an MLM head starting from a base model, writes a domain-adapted encoder, and then fine-tunes NER with the same data pipeline.
- Implementation note: The provided DAPT notebooks demonstrate MLM for BERTić. The identical procedure applies to `classla/xlm-r-bertic` (with SentencePiece tokenizer); the methodological description and expected benefits transfer unchanged.


### 2.2 Training Methodology
- Tokenization:
  - BERT family (BERTić/mBERT): WordPiece; fast tokenizer enabled (`use_fast=True`).
  - XLM-R-BERTić: SentencePiece; loaded via AutoTokenizer.
- Sequence handling (long documents):
  - Sliding windows implemented in `ner/shared/dataset.py` (`tokenize_and_align_labels_with_sliding_window`).
  - Max length = 512, stride = 128. Special tokens added; subsequent subwords and padding receive `-100` labels to be ignored in loss.
- Optimization and regularization:
  - Optimizer/scheduler: HuggingFace Trainer with AdamW and a linear LR scheduler with warmup (`lr_scheduler_type="linear"`; warmup_steps≈500; weight_decay≈0.01). Default LR≈3e-5; epochs≈8; batch sizes per device≈4–8 depending on notebook.
  - Early stopping: Enabled via `EarlyStoppingCallback` (patience≈3) in `ner/shared/model_utils.py`.
  - Dropout: Default dropout from the respective pre-trained heads; no custom dropout layers beyond standard heads.
- Class imbalance mitigation:
  - Class weights computed from training labels (`sklearn.utils.compute_class_weight`) and applied in a custom `WeightedTrainer` with `CrossEntropyLoss(weight=...)` (see `ner/notebook/bertic_class_weights.ipynb`).
- Cross-validation:
  - 5-fold K-Fold CV (scikit-learn `KFold`) used across notebooks to ensure robust estimates and fair comparisons.
- Evaluation metrics:
  - Entity-level precision/recall/F1 (seqeval) as primary metrics; token-level accuracy reported for completeness.
  - Both macro and micro/global reporting can be produced; weighted averages are supported in `ner/shared/evaluation.py` for corpus-level summaries.
- Pipeline consistency:
  - The exact same preprocessing (cleaning assumptions), tokenization, sliding-window strategy, CV protocol, optimizer/scheduler settings, and evaluation functions are reused across model notebooks to enable like-for-like comparisons.


## 3. Figures and Tables

### 3.1 Process Flow Diagram (Data → Modeling)
```mermaid
flowchart LR
    S[Scraping] --> C[Cleaning]
    C --> A[Annotation]
    A --> P[Preprocessing (BIO, tokenization, windows)]
    P --> M[Model training]
    M --> V[Validation & Metrics]
```

### 3.2 Entity Types (definitions with examples)
- Institutions: COURT (e.g., “OSNOVNI SUD U NIKŠIĆU”), REGISTRAR (court registrar names/titles).
- Participants: DEFENDANT (okrivljeni), PROSECUTOR (tužilac), JUDGE (sudija/sudija–pojedinac).
- Outcomes/metadata: CRIMINAL_ACT (e.g., “laka tjelesna povreda”), PROVISION / PROVISION_MATERIAL (e.g., “čl. 152 st. 1 Krivičnog zakonika”), PROVISION_PROCEDURAL (e.g., ZKP provisions), VERDICT (operative decision text), SANCTION (generic mention), SANCTION_TYPE (e.g., “novčana kazna”), SANCTION_VALUE (e.g., “800 €” / “tri mjeseca”), PROCEDURE_COSTS, DECISION_DATE (e.g., “28.05.2019.”), CASE_NUMBER (e.g., “K 139/17”).

### 3.3 Model Comparison (characteristics)
- mBERT: Multilingual baseline; weaker domain adaptation for Serbian legal.
- BERTić: South Slavic pretraining; improved lexical/subword match for Serbian.
- BERT+CRF: Structured decoding improves BIO consistency; torchcrf-based.
- XLM-R-BERTić: XLM-R multilingual strength + Serbian specialization.
- DAPT (MLM): Domain-adapted encoder (implemented for BERTić; applicable to XLM-R-BERTić) trained on ~800–850 unlabeled judgments, then fine-tuned for NER.
- GLiNER variants (zero-shot): `urchade/gliner_multiv2.1`, `knowledgator/gliner-x-large`, `urchade/gliner_large-v2`, `modern-gliner-bi-large-v1.0` evaluated in `ner/notebook/gliner_zero_shot.ipynb`.

### 3.4 Corpus Statistics (indicative)
- Documents: 225 annotated judgments.
- Tokens: 232,475 labeled tokens.
- Cross-validation: K=5; ~45 docs/fold.
- Windowed chunks (approx per fold): train ≈ 1,845; val ≈ 500 (depends on document lengths and tokenizer).
- Entity distribution: Imbalanced; provisions and sanction-related labels more frequent than roles like REGISTRAR.


## 4. Methodological Justifications and Related Work
- Legal NLP and NER: Prior work shows domain specialization and structured decoding are important for legal text, which contains formulaic sections (e.g., provisions, verdicts) and long-distance dependencies (citations → outcomes).
- Low-resource/under-resourced languages: For Serbian and closely related South Slavic languages, multilingual models provide a strong start, but language- and domain-adapted pretraining (e.g., BERTić; DAPT) typically yields larger gains.
- Domain adaptation (DAPT/MLM): Empirical studies consistently find domain-adaptive MLM on unlabeled in-domain corpora improves downstream NER more than modest architectural changes. This is reinforced by our pipeline behavior where the domain-adapted encoder reduces OOV effects, improves segmentation around legal citations, and stabilizes learning for sparse labels.
- South Slavic NLP: BERTić’s pretraining regime yields better subword segmentation and richer contextualization for Serbian morphosyntax compared to generic multilingual encoders; XLM-R-BERTić complements this with broader multilingual generalization.


## 5. Implementation Notes and Pointers
- Notebooks (methods):
  - `ner/notebook/bertic.ipynb` – BERTić baseline with 5-fold CV.
  - `ner/notebook/bertic_class_weights.ipynb` – Class-weighted training for imbalance.
  - `ner/notebook/bertic_crf.ipynb` – BERT + CRF architecture and evaluation.
  - `ner/notebook/xlm_r_bertic.ipynb` – XLM-R-BERTić 5-fold CV.
  - `ner/notebook/gliner_zero_shot.ipynb` – GLiNER multi-model zero-shot evaluation.
- DAPT (MLM): `ner/serbian_legal_ner_pipeline_dapt_mlm*.ipynb` (BERTić-based DAPT), with logs indicating ~849 MLM documents.
- Shared modules (consistency):
  - `ner/shared/config.py` – entity inventory (16 types), default training args, paths.
  - `ner/shared/dataset.py` – BIO dataset, sliding-window tokenization/alignment.
  - `ner/shared/model_utils.py` – Trainer creation, metrics, early stopping, linear scheduler.
  - `ner/shared/evaluation.py` – plotting training curves, precision/recall/F1 summaries.
  - `ner/shared/data_processing.py` – LabelStudio → BIO conversion and export analysis.
- Scraping/Cleaning:
  - `scraper/montenegro_scraper.py` – Playwright SPA navigation, query selection (Presuda, “kriv je”), result iteration, judgment extraction.
  - `scraper/css_cleaner.py` – regex-based CSS/HTML artifact removal and paragraph restoration.


## 6. Reproducibility and Pipeline Consistency
- All comparative experiments reuse identical preprocessing (cleaning assumptions), tokenization, sliding window configuration (512/128), cross-validation protocol (K=5), Trainer settings (AdamW + linear scheduler; early stopping), and evaluation metrics. This ensures fairness in model comparisons (mBERT vs BERTić vs XLM-R-BERTić vs BERT+CRF vs DAPT-initialized encoders), aligning with good scientific practice for controlled ablations.

