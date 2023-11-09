# Transformers_LLM

**Description:** In this tutorial series, we will learn and use the 🤗 Hugging Face Transformer API  

* how to build and preprocess a Custom Dataset from a CSV file with the 🤗 Hugging Face Datasets API
* how to train a 🤗 Hugging Face Tokenizer from scratch with the 🤗 Hugging Face Tokenizer API
* how to train a Causal Language Transformer Model from scratch
* how to push (upload) a Model, Dataset, and Tokenizer to the 🤗 Hugging Face Hub
* how to download and use a Model, Dataset, and Tokenizer from the 🤗 Hugging Face Hub
* how to generate text using the 🤗 Hugging Face Text Generation Pipeline

(TR)
**Tanım:** 
* CSV dosyasından 🤗 Hugging Face'in kabul edeceği bir veri seti oluşturulacak, oluşturulan veri seti ön işleme yapılacak bu ön işleme için 🤗 Hugging Face'in Datasets API'si kullanılacak
* Hazır bir Tokenizer kullanılacak, sıfırdan Tokenizer oluşturulup eğitilecek bunun için 🤗 Hugging Face'in Tokenizer API'si kullanılacak
* GPT-2 tabanlı bir dil modeli(Transfomer) sıfırdan eğitilecek
* Eğitilen dil modeli, tokenizer, veri seti kendi 🤗 Hugging Face hesabımıza(hub'a) yüklenecek
* Yükledikten sonra 🤗 Hugging Face Hub'tan dil modelimizi, tokenizer'ımızı, veri setimizi indirip kullanacağız
* Dil modelimizi kendi veri setimizde ve Tokenizer'ımızda eğittikten sonra 🤗 Hugging Face'in Text Generation Pipeline'ını kullanarak veri metnini üreteceğiz

We will cover all these topics with sample implementations in **Python / TensorFlow / Keras** environment.

We will use a [Kaggle Dataset](https://www.kaggle.com/savasy/multiclass-classification-data-for-turkish-tc32?select=ticaret-yorum.csv) in which there are 32 topics and more than 400K total reviews.

At the end of this tutorial, we will be able to generate text using a GPT2 transformer model trained on a Turkish review dataset as below:

![image](https://github.com/imelike/Transformers_LLM/assets/128046415/3bc9ecb5-27a1-4d2a-91c1-5a84459f302e)

![download](https://github.com/imelike/Transformers_LLM/assets/128046415/92ce0f10-0b11-4558-a726-bbbf591341ed)

#

# References:
 * [Training a causal language model from scratch using  🤗 Hugging Face Transformers](https://huggingface.co/course/chapter7/6?fw=tf)

 * [Share a model to the 🤗 Hugging Face Hub](https://huggingface.co/docs/transformers/model_sharing)

 * [Share a dataset to the 🤗 Hugging Face Hub](https://huggingface.co/docs/datasets/upload_dataset)

 * [The 🤗 Hugging Face Datasets Library](https://huggingface.co/course/chapter5/1?fw=pt)

 * [The 🤗 Hugging Face Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

 * [The 🤗 Hugging Face Text Generation Pipeline](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/pipelines#transformers.TextGenerationPipeline)

 * [An open source Git extension for versioning large files](https://git-lfs.github.com/)
 
 * [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)




