# **PART C:** Build a 🤗 Hugging Face Data Pipeline

from transformers import AutoTokenizer
from datasets import load_dataset

context_length = 40

# Tokenize the dataset

# Modelimde kullanacağım Tokenları Hugging Face Hub'dan indireceğim
# bu veri setini tokenization.py de hazırlamıştım
# huggingface-cli login kodunu terminale yaz
# kendi token-id'ni yapıştır
tokenizer = AutoTokenizer.from_pretrained("imelike/turkishReviews-ds-mini")
tokenizer

# Tokenize edeceğim temizlediğim, hazır veri setini Hugging Face Hub'dan indireceğim
# bu veri setini prepare.py de temizlemiştim
reviews_sample = load_dataset("imelike/turkishReviews-ds-mini")
reviews_sample

def tokenize(element):
    outputs = tokenizer(
        element["review"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = reviews_sample.map(
    tokenize, batched=True, remove_columns=reviews_sample["train"].column_names
)
tokenized_datasets


# Data Collator

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")

out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

for key in out:
    print(f"{key}: {out[key][0]}")


# Convert from Hugging Face Dataset to TensorFlow Dataset

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)

len(tf_train_dataset)



