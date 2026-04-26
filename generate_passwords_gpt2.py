from transformers import GPT2Tokenizer, GPT2LMHeadModel
from zxcvbn import zxcvbn
import re
import torch


# Load model + tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-passwords")
model = GPT2LMHeadModel.from_pretrained("./gpt2-passwords")

model.eval()


def clean_password(p):
    p = p.replace("<|pwd|>", "").strip()

    return p[:12]  # enforce adversarial length cap



# Generate password
def generate_password():
    inputs = tokenizer("<|pwd|>", return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=12,              
        do_sample=True,
        temperature=0.7,            
        top_k=40,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_password(text)




# Evaluate password
def evaluate_password(p):
    result = zxcvbn(p)
    return result["score"], result["guesses"]


# Main loop
def test_generated_passwords(n=20):
    for _ in range(n):

        # generate
        p = generate_password()

        
        # skip invalid outputs
        if not p or len(p) < 3:
            print("Skipped invalid password")
            print("-" * 40)
            continue

        try:
            score, guesses = evaluate_password(p)

            # ADVERSARIAL FILTER
            if score == 4 and guesses < 1e8:
                print(f"ADVERSARIAL: {p} → Score: {score}/4 | Guesses: {guesses}")
            else:
                print(f"{p} → Score: {score}/4 | Guesses: {guesses}")

        except Exception:
            print(f"Skipped invalid password: {p}")

        print("-" * 40)


# Run
test_generated_passwords(50)