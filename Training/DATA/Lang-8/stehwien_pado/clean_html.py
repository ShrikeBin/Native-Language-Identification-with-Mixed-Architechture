import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv("all.csv")

def clean_text(text: str) -> str:
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub('<.*?>', '', str(text))
    return text

df['text'] = df['text'].progress_apply(clean_text)

df.to_csv("all_clean.csv", index=False)