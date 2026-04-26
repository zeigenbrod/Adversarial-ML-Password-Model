import random
import json
import re
import os
import urllib.request
from zxcvbn import zxcvbn
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Wordlist setup
# ─────────────────────────────────────────────────────────────
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
        print("Falling back to words.txt")
        WORDLIST_FILE = "words.txt"


# ─────────────────────────────────────────────────────────────
# Word filter (keeps “human words”)
# ─────────────────────────────────────────────────────────────
VOWELS = set("aeiou")

def is_word_like(w):
    if len(w) < 4 or len(w) > 10:
        return False
    if not any(c in VOWELS for c in w):
        return False
    vowel_ratio = sum(c in VOWELS for c in w) / len(w)
    return vowel_ratio >= 0.25


raw_words = set()
with open(WORDLIST_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        word = line.strip()
        base = re.sub(r'[^a-zA-Z]', '', word)

        if is_word_like(base.lower()):
            raw_words.add(base.lower())

words = list(raw_words)
print(f"Loaded {len(words)} base words")


LEET = {
    'a': '@',
    'e': '3',
    'i': '1',
    'o': '0',
    's': '$'
}

def leet(w):
    out = []
    for c in w:
        if c in LEET and random.random() < 0.5:
            out.append(LEET[c])
        else:
            out.append(c)
    return ''.join(out)


# ─────────────────────────────────────────────────────────────
# Password generator 
# ─────────────────────────────────────────────────────────────
def gen_candidate():
    w = random.choice(words)

    pattern = random.choice([
        "plain",
        "leet",
        "leet_digit",
    ])

    if pattern == "plain":
        result = w

    elif pattern == "leet":
        result = leet(w)

    elif pattern == "leet_digit":
        result = leet(w)

    else:
        return None


    return result


# ─────────────────────────────────────────────────────────────
# Crackability heuristic 
# ─────────────────────────────────────────────────────────────
LEET_REVERSE = {'@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's'}

def is_rule_crackable(pwd):
    base = ''.join(LEET_REVERSE.get(c, c) for c in pwd.lower())
    base = re.sub(r'[^a-z]', '', base)
    return len(base) >= 4


# ─────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────
ADVERSARIAL_TARGET = 10000
MAX_ATTEMPTS = 500000

dataset = []
adversarial_count = 0
attempts = 0

print(f"\nGenerating until {ADVERSARIAL_TARGET} adversarial passwords...\n")

while adversarial_count < ADVERSARIAL_TARGET and attempts < MAX_ATTEMPTS:
    attempts += 1
    pwd = gen_candidate()

    if not pwd:
        continue

    try:
        result = zxcvbn(pwd)
        score = result['score']
        crackable = is_rule_crackable(pwd)

        entry = {
            "password": pwd,
            "zxcvbn_score": score,
            "guesses_log10": round(result.get('guesses_log10', 0), 2),
            "is_adversarial_candidate": score >= 3 and crackable,
        }

        dataset.append(entry)

        if score >= 3 and crackable:
            adversarial_count += 1

        if attempts % 1000 == 0:
            rate = adversarial_count / attempts * 100
            print(f"Attempts: {attempts} | Adversarial: {adversarial_count}/{ADVERSARIAL_TARGET} | Rate: {rate:.1f}%")

    except Exception:
        continue


# ─────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────
with open("labeled_passwords.json", "w") as f:
    json.dump(dataset, f, indent=2)

adversarial_only = [e for e in dataset if e['is_adversarial_candidate']]
with open("adversarial_dataset_10k.txt", "w") as f:
    for entry in adversarial_only:
        f.write(f"<|pwd|> {entry['password']}\n")


score_dist = Counter(e['zxcvbn_score'] for e in dataset)

print("\n==============================")
print("GENERATION SUMMARY")
print("==============================")
print(f"Attempts: {attempts}")
print(f"Saved: {len(dataset)}")
print(f"Adversarial: {adversarial_count}")
print(f"Score distribution: {dict(sorted(score_dist.items()))}")