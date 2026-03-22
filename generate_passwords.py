from transformers import GPT2Tokenizer, GPT2LMHeadModel
from zxcvbn import zxcvbn
import re

# trained GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-passwords")
model = GPT2LMHeadModel.from_pretrained("./gpt2-passwords")


# clean password
def clean_password(p):
    p = p.replace("<|pwd|>", "").strip()
    p = re.sub(r'[^a-zA-Z0-9]', '', p)
    return p[:16]


# generate password
def generate_password():
    inputs = tokenizer("<|pwd|>", return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=16,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_password(text)


# test passwords
def test_generated_passwords(n=20):
    for _ in range(n):
        p = generate_password()

        # skip empty or too short outputs, b/c zxcvbn will crash
        if not p or len(p) < 2:
            print("Skipped invalid password")
            print("-" * 40)
            continue

        try:
            result = zxcvbn(p)

            score = result['score']
            guesses = result['guesses']

            print(f"{p} → Score: {score}/4 | Guesses: {guesses}")

        
        except Exception:
            print(f"Skipped invalid password: {p}")

        print("-" * 40)


# run test
test_generated_passwords(20)
