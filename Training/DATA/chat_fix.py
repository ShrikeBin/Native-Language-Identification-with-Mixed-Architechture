import json
import re
import html
import pandas as pd
from collections import defaultdict

INPUT_FILE = "lang-8-20111007-2.0/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat"
OUTPUT_FILE = "Lang-8/chat/lang8_nli_clean.csv"
MIN_LENGTH = 40
MAX_PER_L1 = 5000   # max sentences per L1
TOP_L1_COUNT = 20   # keep top 20 L1s

def clean_text(text):
    """Remove HTML, links, emojis, and normalize whitespace."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'https?://\S+', '', text)  # remove links
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    if re.match(r'^[\W_]+$', text):  # only punctuation/emoji
        return ""
    return text

# Step 1: collect and clean all sentences
all_entries = []
l1_counts = defaultdict(int)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry[2] != "English":
            continue
        native_lang = entry[3]
        for sentence in entry[4]:
            sentence_clean = clean_text(sentence)
            if len(sentence_clean) < MIN_LENGTH:
                continue
            all_entries.append((sentence_clean, native_lang))
            l1_counts[native_lang] += 1

# Step 2: keep only top L1s
top_l1s = set(sorted(l1_counts, key=l1_counts.get, reverse=True)[:TOP_L1_COUNT])

# Step 3: balance dataset
l1_written = defaultdict(int)
data = []

for sentence, l1 in all_entries:
    if l1 not in top_l1s:
        continue
    if l1_written[l1] >= MAX_PER_L1:
        continue
    data.append({'text': sentence, 'native_language': l1})
    l1_written[l1] += 1

# Step 4: save as CSV with headers
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"Saved {len(df)} sentences to {OUTPUT_FILE}")
