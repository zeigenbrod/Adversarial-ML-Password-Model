import random
import json
from zxcvbn import zxcvbn

# load words
with open("words.txt") as f:
    words = [w.strip() for w in f if w.strip().isalpha() and len(w.strip()) > 3]

LEET = {'a':'@','e':'3','i':'1','o':'0','s':'$','t':'7'}

def leet(w):
    return ''.join(LEET.get(c, c) for c in w)

def gen_candidate():
    w = random.choice(words)
    # criteria for password/pattern
    pattern = random.choice([
        "word",
        "word_num",
        "cap_word_num",
        "leet_word",
        "leet_word_num",
        "word_special",
        "cap_leet_num_special",
    ])
    if pattern == "word":
        return w
    elif pattern == "word_num":   
        return w + str(random.randint(100,9999))
    elif pattern == "cap_word_num":  
        return w.capitalize() + str(random.randint(100,9999))
    elif pattern == "leet_word": 
        return leet(w) 
    elif pattern == "leet_word_num": 
        return leet(w) + str(random.randint(100,9999))
    elif pattern == "word_special":
        return w + random.choice(['!', '@', '#', '$', '&']) 
    elif pattern == "cap_leet_num_special":
        return leet(w.capitalize()) + str(random.randint(0,9999)) + random.choice(['!', '@', '#', '$', '&'])
    
# generate dataset and label w/ zxcvbn
dataset = []
target = 5000
attempts = 0

while len(dataset) < target and attempts < 50000:
    attempts += 1
    pwd = gen_candidate()
    if not pwd or len(pwd) < 4:
        continue
    result = zxcvbn(pwd)
    score = result['score']
    dataset.append({
        "password": pwd,
        "zxcvbn_score": score,
        "zxcvbn_feedback": result['feedback']['warning'],
        "guesses_log10": result.get('guesses_log10', 0),
        # Adversarial label: high zxcvbn score but rule-derivable = "fool"
        "is_adversarial_candidate": score >= 3,
    })

# Save full labeled dataset
with open("labeled_passwords.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Save plain text for GPT-2 training (all passwords — model learns all patterns)
with open("password_dataset.txt", "w") as f:
    for entry in dataset:
        f.write(f"<|pwd|> {entry['password']}\n")

# Print distribution
from collections import Counter
scores = Counter(e['zxcvbn_score'] for e in dataset)
fools = sum(1 for e in dataset if e['is_adversarial_candidate'])
print(f"Generated {len(dataset)} passwords in {attempts} attempts")
print(f"Score distribution: {dict(sorted(scores.items()))}")
print(f"Adversarial candidates (score >= 3): {fools} ({fools/len(dataset)*100:.1f}%)")