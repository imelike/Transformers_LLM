# **PART E:** Generate Reviews with a 🤗 Hugging Face Text Generation Pipeline

# ## Build a Pipeline

from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
from datasets import load_dataset

dataset = load_dataset("imelike/turkishReviews-ds", split="validation")
review_model = TFGPT2LMHeadModel.from_pretrained("imelike/turkishReviews-ds")
review_tokenizer = AutoTokenizer.from_pretrained("imelike/turkishReviews-ds")

pipe = pipeline(
    "text-generation", model=review_model, tokenizer=review_tokenizer, device=0

)

dataset
dataset['review'][:2]

prompts = ["Termikel Ankastre Ocağımız","Pegasus Ücret İadesi İçin"]

output0=pipe(prompts, num_return_sequences=1)[0][0]["generated_text"]
output1=pipe(prompts, num_return_sequences=1)[1][0]["generated_text"]

print("For prompt ", prompts[0], " the generated text is:")
print(output0)
print("For prompt ", prompts[1], " the generated text is:")
print(output1)


