import json
import pandas as pd

with open("journal_ids.txt", "r", encoding="utf-8") as f:
    journal_ids = set(f.readlines())

with open("../../lang-8-20111007-2.0/lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat", "r") as f:
    lines = f.readlines()

def clean_control_sequences(s: str) -> str:
    return ''.join(
        c for c in s
        if ord(c) >= 32 or c in '\n\r\t'
    )
data = [json.loads(clean_control_sequences(line)) for line in lines]

rows = []
for entry in data:
    if str(entry[0]) + "\n" not in journal_ids:
        continue
    learning_language = entry[2]
    if learning_language != "English":
        continue
    native_language = entry[3]
    learner_sentences = entry[4]
    for sentence in learner_sentences:
        rows.append({'native_language': native_language, 'text': sentence})

df = pd.DataFrame(rows)
df.to_csv("all.csv", index=False)