cd /Users/teif/nlp-cli-tool
cat > README.md << 'EOF'
# W5 Weekly Project — Arabic NLP Classification CLI Tool

## Summary
A production-style CLI tool that runs an end-to-end Arabic text classification pipeline:
EDA → Preprocessing → Embedding → Training/Evaluation
The tool is designed to work with any CSV dataset (any column names) as long as the user provides:
- --text_col (text column name)
- --label_col (label/rating column name)
It produces reproducible outputs (plots, embeddings, reports, models) under outputs/.

## Key Features
Core Requirements
- Click-based CLI with commands/subcommands.
- EDA:
  - Class distribution plots (bar/pie)
  - Text length histogram (words/chars) with descriptive stats
- Preprocessing:
  - Remove: URLs, tashkeel, tatweel, non-Arabic symbols, extra spaces
  - Stopwords removal (Arabic stopwords list stored locally)
  - Replace/normalize Arabic character variants
  - preprocess all chains all steps
- Embeddings:
  - TF-IDF vectors (saved with fitted vectorizer)
- Training/Evaluation:
  - Models: KNN, Logistic Regression, Random Forest
  - Metrics: Accuracy, Precision, Recall, F1 (weighted)
  - Confusion matrices saved as PNG
  - Markdown report saved with timestamp
  - Best model saved as joblib package

Bonus Features Implemented
- pipeline: one-command full workflow run (recommended for instructor testing)
- Wordcloud generation (EDA bonus)
- Additional embeddings:
  - embed bert (transformers)
  - embed sentence-transformer (sentence-transformers)

## Project Structure
nlp-cli-tool/
- main.py
- commands/
  - eda.py
  - preprocessing.py
  - embedding.py
  - training.py
  - pipeline.py
  - wordcloud_cmd.py
- utils/
  - data_handler.py
  - arabic_text.py
  - visualization.py
  - stopwords_ar.txt
- outputs/
  - visualizations/
  - reports/
  - models/
  - embeddings/
- requirements.txt
- README.md

## Installation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import click, pandas, sklearn, matplotlib; print('OK')"

## Input CSV Requirements (Instructor Testing)
Any CSV file is supported as long as:
- It contains a text column (e.g., review_description, text, description)
- It contains a label column (e.g., rating, class, sentiment)
The user must specify the exact column names via CLI flags.

Example (CompanyReviews.csv tested):
review_description,rating,company

## How to Grade in 60 Seconds (Recommended)
Run one command that executes the full pipeline and saves artifacts:
python main.py pipeline --csv_path CompanyReviews.csv --text_col "review_description" --label_col "rating" --test_size 0.3 --out_dir outputs/companyreviews_run
Expected outputs:
- outputs/companyreviews_run/reports/pipeline_report.md
- outputs/companyreviews_run/models/pipeline_best_model.pkl
- outputs/companyreviews_run/embeddings/tfidf.pkl

## CLI Commands
Show all commands:
python main.py --help

### 1) EDA
Class distribution:
python main.py eda distribution --csv_path data.csv --label_col label --plot_type bar
python main.py eda distribution --csv_path data.csv --label_col label --plot_type pie
Text length histogram:
python main.py eda histogram --csv_path data.csv --text_col text --unit words
python main.py eda histogram --csv_path data.csv --text_col text --unit chars
Outputs:
- outputs/visualizations/ (PNG plots)

### 2) Preprocessing
Run all preprocessing steps:
python main.py preprocess all --csv_path data.csv --text_col text --output outputs/final.csv
Individual steps (optional):
python main.py preprocess remove --csv_path data.csv --text_col text --output cleaned.csv
python main.py preprocess stopwords --csv_path cleaned.csv --text_col text --output no_stops.csv
python main.py preprocess replace --csv_path no_stops.csv --text_col text --output normalized.csv
Stopwords:
- Stopwords list is stored locally at utils/stopwords_ar.txt
- It can be swapped with an instructor-provided list without changing code

### 3) Embedding
TF-IDF:
python main.py embed tfidf --csv_path outputs/final.csv --text_col text --max_features 5000 --output outputs/embeddings/tfidf.pkl
Bonus embeddings:
python main.py embed bert --csv_path CompanyReviews.csv --text_col review_description --output outputs/embeddings/bert.pkl
python main.py embed sentence-transformer --csv_path CompanyReviews.csv --text_col review_description --output outputs/embeddings/sbert.pkl

### 4) Training and Evaluation
Train models using an embedding artifact:
python main.py train --csv_path outputs/final.csv --label_col label --embeddings_path outputs/embeddings/tfidf.pkl --test_size 0.2 --models knn --models lr --models rf --save_model outputs/models/best_model.pkl
Outputs:
- outputs/reports/training_report_<timestamp>.md
- outputs/visualizations/confmat_<model>_<timestamp>.png
- outputs/models/best_model.pkl

## Outputs
- outputs/visualizations/
  - EDA plots (distribution/histogram)
  - confusion matrices
  - wordclouds
- outputs/embeddings/
  - TF-IDF artifacts and optional dense embeddings
- outputs/models/
  - saved best models
- outputs/reports/
  - markdown reports

## Verified Robustness (Instructor-Style Testing)
The tool was validated on:
1) Different column names:
- Successful run on a dataset with review/category columns via pipeline.
2) Messy Arabic text:
- Successful run on text containing diacritics, tatweel, URLs, emojis, numbers.
3) Missing values + class imbalance:
- Successful run on datasets containing empty strings and NaNs.
- The pipeline disables stratified splitting when it is mathematically invalid.
4) Real dataset test:
- Successful run on CompanyReviews.csv (40,046 samples, 3 classes) producing:
  - TF-IDF (5000 features) and a Logistic Regression baseline report
  - Saved embeddings/model/report artifacts

## Notes / Troubleshooting
Stratify warning on small datasets:
Stratified splitting requires the test set to include at least one sample per class.
If test_size is too small for the number of classes, stratify is disabled automatically to prevent errors.
Recommendation:
- For small multi-class datasets, use --test_size 0.3 or higher.

Wordcloud Arabic rendering:
Arabic wordcloud output may require Arabic font + shaping/RTL handling depending on the environment.

## Rubric Mapping
Core Features:
- EDA distribution + histogram implemented with saved plots.
- Preprocessing remove + stopwords + replace + all implemented.
- Embedding: TF-IDF implemented (baseline requirement) + bonus dense embeddings.
- Training: metrics + report + confusion matrices + model saving.
- Clean modular structure: commands/ + utils/ separation.
- Documentation: reproducible commands for instructor evaluation.
EOF
