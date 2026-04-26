from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# =========================
# Load dataset
# =========================
dataset = load_dataset("text", data_files={"train": "adversarial_dataset_10k.txt"})

# =========================
# Model name
# =========================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({"additional_special_tokens": ["<|pwd|>"]})

# =========================
# Model
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))

# =========================
# LoRA config
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # works for TinyLlama
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# Tokenization
# =========================
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=20,
    )

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

# =========================
# Data collator
# =========================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =========================
# Training args (Mac-safe)
# =========================
training_args = TrainingArguments(
    output_dir="./tinyllama-lora-passwords",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    report_to="none",

    fp16=False,   # IMPORTANT for Mac
    bf16=False
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# =========================
# Train
# =========================
trainer.train()

# save LoRA adapter
model.save_pretrained("./tinyllama-lora-passwords")
tokenizer.save_pretrained("./tinyllama-lora-passwords")

print("\nTraining complete. Saved to ./tinyllama-lora-passwords")