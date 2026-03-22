from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# loads the dataset
dataset = load_dataset("text", data_files={"train": "password_dataset.txt"})

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# pad token that is used to make different length passwords, gpt-2 wants to make same length passwords
tokenizer.pad_token = tokenizer.eos_token

# tokenize data
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

# load model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# training settings
training_args = TrainingArguments(
    output_dir="./gpt2-passwords",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# train
trainer.train()

# save model & tokenizer to folder
trainer.save_model("./gpt2-passwords")
tokenizer.save_pretrained("./gpt2-passwords")
