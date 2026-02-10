# go throught the file with a regex and extract journal ids
import re

with open("lang8_urls.txt", "r", encoding="utf-8") as f:
    entries = f.readlines()

journal_ids = set()
for entry in entries:
    match = re.search(r'journals/(\d+)', entry)
    if match:
        journal_id = match.group(1)
        journal_ids.add(journal_id)

with open("journal_ids.txt", "w", encoding="utf-8") as f:
    for journal_id in journal_ids:
        f.write(journal_id + "\n")