# PART D:**  Train a 🤗 Hugging Face Causal Language Model (Transformer) from scratch

from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
tokenizer = AutoTokenizer.from_pretrained("imelike/turkishReviews-ds-mini")

context_length = 40

# Initializing a new Transformer Model
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


model = TFGPT2LMHeadModel(config)
model(model.dummy_inputs)  # Builds the model
model.summary()


# Log in to Hugging Face Hub
# huggingface-cli login kodunu terminale yaz


# Set up the optimizer

from transformers import create_optimizer
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)


# Compile the model

model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Train the model

from transformers.keras_callbacks import PushToHubCallback
callback = PushToHubCallback(output_dir="imelike/turkishReviews-ds-mini", tokenizer=tokenizer)
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3, callbacks=[callback])

model.push_to_hub("imelike/turkishReviews-ds-mini")