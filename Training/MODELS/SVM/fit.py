from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# ===== Load full data =====
import json
from pathlib import Path

TEXT_COL = "text"
LABEL_COL = "language"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_config.json"

def load_df(path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / path 
    print(f"Loading data from {path}...")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported data format: {path.suffix}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

train_df = load_df(cfg["train_path"])
test_df = load_df(cfg["test_path"])

label_map = {int(k): v for k, v in cfg["label_map"].items()}
label_map = {v: int(k) for k, v in label_map.items()}
train_df[LABEL_COL] = train_df[LABEL_COL].map(label_map).astype(int)
test_df[LABEL_COL] = test_df[LABEL_COL].map(label_map).astype(int)

# ===== SVM stuff =====
X_train, y_train = train_df[TEXT_COL].map(str).tolist(), train_df[LABEL_COL].tolist()
X_test, y_test = test_df[TEXT_COL].map(str).tolist(), test_df[LABEL_COL].tolist()

# print(len(set(X_train) & set(X_test))) # debugging: duplicates, should be 0

model = Pipeline([
    ("vec", TfidfVectorizer(
        analyzer="word",     # word n-grams
        ngram_range=(1,3),   # unigrams - trigrams
        min_df=2
    )),
    ("clf", LinearSVC())
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(classification_report(y_test, pred))