from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# loads dataset
dataset = load_dataset("text", data_files={"train": "adversarial_dataset_10k.txt"})

# load GPT-2 tokenizer and add prompt token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"additional_special_tokens": ["<|pwd|>"]})
tokenizer.pad_token = tokenizer.eos_token


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# tokenize 
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

# causal LM collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, not masked LM
)

# training settings
training_args = TrainingArguments(
    output_dir="./gpt2-passwords",
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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# train
trainer.train()

# save model and tokenizer to same folder so generate_passwords.py can load them
trainer.save_model("./gpt2-passwords")
tokenizer.save_pretrained("./gpt2-passwords")

print("\nTraining complete. Model saved to ./gpt2-passwords")
print("Run generate_passwords.py to sample from the model.")
