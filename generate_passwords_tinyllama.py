from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from zxcvbn import zxcvbn
import torch

# Load base + tokenizer
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained("./tinyllama-lora-passwords")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,   
    device_map="auto"
)

base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Load LoRA
model = PeftModel.from_pretrained(
    base_model,
    "./tinyllama-lora-passwords",
    is_trainable=False
)

model.eval()

# Clean output
def clean_password(p):
    p = p.replace("<|pwd|>", "").strip()
    return p[:12]

# Generate password
def generate_password():
    inputs = tokenizer("<|pwd|>", return_tensors="pt").to(model.device)

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

def test_generated_passwords(n=25, output_file="generated_passwords.txt"):
    with open(output_file, "w") as f:
        for _ in range(n):
            p = generate_password()

            if not p or len(p) < 3:
                print("Skipped invalid password")
                print("-" * 40)
                continue

            try:
                score, guesses = evaluate_password(p)
                print(f"{p} → Score: {score}/4 | Guesses: {guesses}")
                print("-" * 40)
                f.write(p + "\n")
            except Exception:
                print(f"Skipped invalid password: {p}")
                print("-" * 40)

test_generated_passwords(50)
