import random

# load words
with open("words.txt") as f:
    words = [w.strip() for w in f if w.strip().isalpha()]

def gen_password():
    w = random.choice(words)

    pattern = random.choice([
        "word",
        "word_num",
        "cap_word",
        "cap_word_num"
    ])

    if pattern == "word":
        return w

    elif pattern == "word_num":
        return w + str(random.randint(0, 9999))

    elif pattern == "cap_word":
        return w.capitalize()

    elif pattern == "cap_word_num":
        return w.capitalize() + str(random.randint(0, 9999))


# generate small test dataset
dataset_size = 100
dataset = []

for _ in range(dataset_size):
    pwd = gen_password()
    dataset.append(f"<|pwd|> {pwd}")

# save file
with open("password_dataset.txt", "w") as f:
    for line in dataset:
        f.write(line + "\n")

print(f"Generated {dataset_size} passwords.")
