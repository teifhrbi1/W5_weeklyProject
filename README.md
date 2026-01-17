# Arabic NLP Classification CLI Tool

A command-line interface (CLI) tool that runs an end-to-end Arabic text classification pipeline:
EDA → Preprocessing → Embedding (TF-IDF) → Training/Evaluation, with an optional one-line pipeline command.

## Project Structure

nlp-cli-tool/
- main.py
- commands/
  - eda.py
  - preprocessing.py
  - embedding.py
  - training.py
  - pipeline.py
- utils/
  - data_handler.py
  - arabic_text.py
  - stopwords_ar.txt
- outputs/
  - visualizations/
  - reports/
  - models/
  - embeddings/
- requirements.txt

## Requirements

- Python 3.10+ recommended (tested locally with newer Python versions).
- Install dependencies from requirements.txt.

## Setup

Create and activate a virtual environment, then install dependencies:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Verify:

python -c "import click, pandas, sklearn, matplotlib; print('OK')"

## CLI Overview

Show help:

python main.py --help

Commands:
- eda         Exploratory data analysis (distribution, histogram)
- preprocess  Text preprocessing (remove, stopwords, replace, all)
- embed       Embeddings (tfidf)
- train       Train and evaluate models (knn, lr, rf)
- pipeline    Run preprocess + tfidf + training in one command (bonus)

## Input Data Format

Your CSV should contain:
- A text column (e.g., description)
- A label column (e.g., class)

Example:

description,class
"هذا المنتج ممتاز",positive
"التوصيل سيء جدا",negative

## 1) EDA

### Class Distribution

python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar
python main.py eda distribution --csv_path data.csv --label_col class --plot_type pie

Outputs:
- Plot saved to outputs/visualizations/
- Console summary with counts and percentages

### Text Length Histogram

python main.py eda histogram --csv_path data.csv --text_col description --unit words
python main.py eda histogram --csv_path data.csv --text_col description --unit chars

Outputs:
- Plot saved to outputs/visualizations/
- Console stats: mean/median/std/min/max

## 2) Preprocessing

The preprocessing steps are implemented in utils/arabic_text.py and use a stopwords list in utils/stopwords_ar.txt.

### remove (cleaning)

Removes:
- URLs
- Tashkeel (diacritics)
- Tatweel
- Non-Arabic symbols (keeps Arabic letters and spaces)

python main.py preprocess remove --csv_path data.csv --text_col description --output cleaned.csv

### stopwords

Removes Arabic stopwords using utils/stopwords_ar.txt.

python main.py preprocess stopwords --csv_path cleaned.csv --text_col description --output no_stops.csv

### replace (normalization)

Normalizes common Arabic variants:
- أ/إ/آ → ا
- ى → ي
- ة → ه
- ؤ → و
- ئ → ي

python main.py preprocess replace --csv_path no_stops.csv --text_col description --output normalized.csv

### all (chain all preprocessing steps)

python main.py preprocess all --csv_path data.csv --text_col description --output final.csv

## 3) Embedding (TF-IDF)

Creates TF-IDF vectors and saves them as a joblib file containing:
- X (sparse matrix)
- vectorizer (TfidfVectorizer)

python main.py embed tfidf --csv_path final.csv --text_col description --max_features 5000 --output outputs/embeddings/tfidf.pkl

Quick check:

python -c "import joblib; d=joblib.load('outputs/embeddings/tfidf.pkl'); print(d.keys()); print(d['X'].shape)"

## 4) Training and Evaluation

Trains and evaluates selected models:
- knn
- lr (Logistic Regression)
- rf (Random Forest)

Generates:
- Markdown report in outputs/reports/
- Confusion matrix images in outputs/visualizations/
- Best model package in outputs/models/ (optional)

Run training:

python main.py train --csv_path final.csv --label_col class --embeddings_path outputs/embeddings/tfidf.pkl --test_size 0.2 --models knn --models lr --models rf --save_model outputs/models/best_model.pkl

Notes:
- For very small datasets, stratified splitting may be disabled automatically.
- KNN automatically adjusts k to not exceed the training set size.

## 5) One-Line Pipeline (Bonus)

Runs:
preprocess (all) → tfidf → simple training report/model outputs

python main.py pipeline --csv_path data.csv --text_col description --label_col class --max_features 5000 --test_size 0.2

Outputs are saved under outputs/ (embeddings/models/reports).

## Outputs

- outputs/visualizations/
  - EDA distribution/histograms
  - confusion matrices
- outputs/embeddings/
  - tfidf.pkl
- outputs/reports/
  - training_report_*.md
  - pipeline_report.md
- outputs/models/
  - best_model.pkl (from train)
  - pipeline_best_model.pkl (from pipeline)

## Full Workflow Example

python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar
python main.py preprocess all --csv_path data.csv --text_col description --output final.csv
python main.py embed tfidf --csv_path final.csv --text_col description --max_features 5000 --output outputs/embeddings/tfidf.pkl
python main.py train --csv_path final.csv --label_col class --embeddings_path outputs/embeddings/tfidf.pkl --test_size 0.2 --models knn --models lr --models rf --save_model outputs/models/best_model.pkl

## Troubleshooting

- "No such command": Ensure the command is registered in main.py.
- "Class has only 1 member": Use a larger dataset, or stratify will be disabled automatically.
- KNN error with small data: k is adjusted automatically in training implementation, but extremely small datasets may still produce unstable metrics.
