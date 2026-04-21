import random
import json
import re
import os
import urllib.request
from zxcvbn import zxcvbn
from collections import Counter

# ── Wordlist ──────────────────────────────────────────────────────────────────
WORDLIST_FILE = "password_words.txt"
WORDLIST_URLS = [
    "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/100k-most-used-passwords-NCSC.txt",
    "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10k-most-common.txt",
]

if not os.path.exists(WORDLIST_FILE):
    downloaded = False
    for url in WORDLIST_URLS:
        try:
            print(f"Trying {url}...")
            urllib.request.urlretrieve(url, WORDLIST_FILE)
            print("Download successful.")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed: {e}")
    if not downloaded:
        print("All downloads failed — falling back to words.txt")
        WORDLIST_FILE = "words.txt"

raw_words = set()
with open(WORDLIST_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        word = line.strip()
        base = re.sub(r'[^a-zA-Z]', '', word)
        if 4 <= len(base) <= 10:
            raw_words.add(base.lower())

words = list(raw_words)
print(f"Loaded {len(words)} base words")

# ── Leet substitution map ─────────────────────────────────────────────────────
LEET = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}

def leet(w):
    return ''.join(LEET.get(c, c) for c in w)

# ── Password generator ────────────────────────────────────────────────────────
def gen_candidate():
    w = random.choice(words)
    pattern = random.choices([
        "word",
        "word_num",
        "cap_word_num",
        "leet_word",
        "leet_word_num",
        "word_special",
        "cap_leet_num_special",
    ], weights=[5, 10, 15, 10, 15, 15, 30])[0]

    if pattern == "word":
        return w
    elif pattern == "word_num":
        return w + str(random.randint(100, 9999))
    elif pattern == "cap_word_num":
        return w.capitalize() + str(random.randint(100, 9999))
    elif pattern == "leet_word":
        return leet(w)
    elif pattern == "leet_word_num":
        return leet(w) + str(random.randint(10, 999))
    elif pattern == "word_special":
        return w + random.choice(['!', '@', '#', '$', '&'])
    elif pattern == "cap_leet_num_special":
        return leet(w.capitalize()) + str(random.randint(10, 99)) + random.choice(['!', '@', '#'])

# ── Crackability heuristic ────────────────────────────────────────────────────
LEET_REVERSE = {'@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's', '7': 't'}

def is_rule_crackable(pwd):
    base = ''.join(LEET_REVERSE.get(c, c) for c in pwd.lower())
    base = re.sub(r'[^a-z]', '', base)
    return len(base) >= 4

# ── Main generation loop ──────────────────────────────────────────────────────
ADVERSARIAL_TARGET = 10000
MAX_ATTEMPTS = 50000

dataset = []
adversarial_count = 0
attempts = 0

print(f"\nGenerating until {ADVERSARIAL_TARGET} adversarial passwords found...\n")

while adversarial_count < ADVERSARIAL_TARGET and attempts < MAX_ATTEMPTS:
    attempts += 1
    pwd = gen_candidate()

    if not pwd or len(pwd) < 4:
        continue

    try:
        result = zxcvbn(pwd)
        score = result['score']
        crackable = is_rule_crackable(pwd)

        entry = {
            "password": pwd,
            "zxcvbn_score": score,
            "zxcvbn_feedback": result['feedback']['warning'],
            "guesses_log10": round(result.get('guesses_log10', 0), 2),
            "is_adversarial_candidate": score >= 3 and crackable,
        }
        dataset.append(entry)

        if score >= 3 and crackable:
            adversarial_count += 1

        if attempts % 1000 == 0:
            rate = adversarial_count / attempts * 100
            print(f"Attempts: {attempts:>6} | Adversarial: {adversarial_count:>5}/{ADVERSARIAL_TARGET} | Hit rate: {rate:.1f}%")

    except Exception as e:
        print(f"Skipped '{pwd}': {e}")

# ── Save outputs ──────────────────────────────────────────────────────────────

# Full labeled dataset for analysis
with open("labeled_passwords.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Only the 10k adversarial passwords for GPT-2 training
adversarial_only = [e for e in dataset if e['is_adversarial_candidate']]
with open("adversarial_dataset_10k.txt", "w") as f:
    for entry in adversarial_only:
        f.write(f"<|pwd|> {entry['password']}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
score_dist = Counter(e['zxcvbn_score'] for e in dataset)
print(f"\n{'='*50}")
print(f"GENERATION SUMMARY")
print(f"{'='*50}")
print(f"Total attempts:          {attempts}")
print(f"Total passwords saved:   {len(dataset)}")
print(f"Score distribution:      {dict(sorted(score_dist.items()))}")
print(f"Adversarial (score>=3):  {adversarial_count} ({adversarial_count/len(dataset)*100:.1f}%)")
print(f"\nFiles saved:")
print(f"  labeled_passwords.json      — full dataset with scores and labels")
print(f"  adversarial_dataset_10k.txt — 10k adversarial passwords for training")