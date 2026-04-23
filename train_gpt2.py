from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# loads the dataset
dataset = load_dataset("text", data_files={"train": "adversarial_dataset_10k.txt"})

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# tokenize
def tokenize(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=16
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize, batched=True)

# training settings
training_args = TrainingArguments(
    output_dir="./mistral-passwords",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=100,
    save_steps=500,
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# train
trainer.train()

# save model & tokenizer to folder
trainer.save_model("./mistral-passwords")
tokenizer.save_pretrained("./mistral-passwords")