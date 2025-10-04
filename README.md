# Entity‑Resolution Pipeline

This project implements a full pipeline for **entity resolution / deduplication / record linkage**, including blocking, matching (rules + machine learning), clustering, and canonicalization to create “golden records.”

## 🧩 Project Structure

```
Entity‑resolution-pipeline/
│
├── data/
│   ├── clear_data.csv              # normalized input records
│   ├── pair_model.joblib           # (optional) trained matching model
│   └── pair_model_meta.json        # metadata: features, threshold
│
├── out/
│   ├── cand_pairs.csv              # candidate pairs after blocking
│   ├── pairs_pred.csv              # predicted matching pairs
│   ├── rows_with_entity_id.csv     # original rows annotated with entity IDs
│   └── entities.csv                # canonical (golden) records for each entity cluster
│
├── src/
│   ├── pipline.py                  # end‑to‑end pipeline (matching → clustering → canonicalization)
│   ├── rules.py                    # normalization, feature extraction, rule logic & utilities
│   ├── cluster.py                  # clustering logic (connected components, cluster metrics)
│   └── canonicalize.py             # canonicalization logic for merged entities
│
├── tests/                          # unit tests using pytest
├── *.ipynb                         # notebooks: blocking, modeling, EDA, etc.
├── README.txt                      # original README text file
├── .gitignore
└── requirements.txt
```

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/SzaboIvan711/Entity-resolution-pipeline.git
cd Entity-resolution-pipeline
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Usage

### Blocking → Matching → Clustering → Canonicalization

1. Ensure you have **candidate pairs** after blocking (e.g. via `blocking.ipynb`) saved to `out/cand_pairs.csv`.  
2. Run the pipeline:

```bash
python -m src.pipline
# or
python src/pipline.py
```

3. The script will:
   - Load normalized records (`data/clear_data.csv`)
   - Load candidate pairs
   - If a trained model (`pair_model.joblib`) + metadata exist, use it; otherwise fallback to rule-based matching
   - Perform clustering and canonicalization
   - Save outputs into `out/`: `pairs_pred.csv`, `rows_with_entity_id.csv`, `entities.csv`

### Optional: Train / Update Matching Model

Use `model.ipynb` to train a new model, save it to `data/pair_model.joblib` and `data/pair_model_meta.json`.  

## ✅ Tests

Run tests with:

```bash
pytest -v
```

## 🧠 Concepts & Approach

- **Blocking**: reduce search space by heuristics (e.g., name prefix, zip code)
- **Matching**: rule-based or ML model
- **Clustering**: treat pairs as graph edges → connected components
- **Canonicalization**: pick representative values (longest name, most frequent address, etc.)

## 📌 Notes

- Works best with pre-cleaned data.
- Modular design — swap matching or clustering components easily.
- Tune model threshold to balance precision vs recall.

## 📝 License

Add a license here (MIT, Apache 2.0, etc.)
