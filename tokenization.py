'''
 **PART B:** 🤗 Hugging Face Tokenization: Use a Pre-trained Tokenizer or Train a New Tokenizer from scratch?

- **gpt2-turkish-cased** modelini eğitirken kullanılmış olan tokenizer'ı kendi modelimizde kullanalım.
Bu Tokenizer Türkçe corpus üzerinde eğitildiği için Türkçe'ye uygun tokenlar elde etmemiz daha kolay olur.

- Verilen metni Tokenize etmek için iki yöntem var:
 - 1) input string'in ilk n token'ını alırız. ör. token sayısını n=40 belirlemişsek ilk 40 token'ı alırız geri kalan cümle dikkate alınmaz.

 - 2) input string'in tüm tokenları alınır. ör. cümle 120 tokendan oluşuyorsa bundan 3 tane 40'lık sequence elde ederiz.

 !!! Hazırlayacağım dil modelini metin üretme(generate) için kullanacağım ve metnin Türkçe gramer kurallarına(noktalama işaretleri) uyması için 1. yöntemi kullanacağım.

 (tehlikeli!!) 2. yöntemde cümlenin ortasından başlayacağı şek. model eğitilebilir.
'''

import os
from datasets import load_dataset


# Tokenize edeceğim temizlediğim, hazır veri setini Hugging Face Hub'dan indireceğim
# bu veri setini prepare.py de temizlemiştim
reviews_sample = load_dataset("imelike/turkishReviews-ds-mini")
reviews_sample


from transformers import AutoTokenizer

context_length = 40

# Hugging Face'teki yukarıda dediğimiz modelin Tokenizer'ını kullanıyoruz
pretrained_tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")

outputs = pretrained_tokenizer(
    # imelike/turkishReviews-ds-mini train data'mızdaki review'a uyguluyoruz
    reviews_sample["train"][:2]["review"],
    truncation=True,  #belirlenen n tokendan sonrasını alma
    max_length=context_length,
    return_overflowing_tokens=False, ##belirlenen n tokendan sonrasını yeni bir sequence e ekleme
    return_length=True, #kaçlık token serisi ürettiği
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

# redrussianarmy/gpt2-turkish-cased modelindeki tokenizer sözlüğü
# 50258 kelime var
print("vocab_size: ", len(pretrained_tokenizer))

# kendi veri setimizdeki bir yorumun aldığımız modelin tokenizer'ında karşılık geldiği kelimelerin numaraları
txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = pretrained_tokenizer(txt)['input_ids']
print(tokens)


'''
**NOTE(TR)**:

- yukarıdaki sayıların, redrussianarmy/gpt2-turkish-cased tokenizer sözlüğünde karşılık geldiği kelimeler
- Sürat ==> sür + at,  Kargom==> kar + g + om olarak almış

 GPT-2'de kullanılan Tokenizer():
-  kelime bazlı da olabilir
-  subword(heceleme) de olabilir
-  harf bazlı da olabilir bu üç farklı şek. oalbilir

 
- burada genellikle kullanılan subword parçalama yapılmış, modelimiz bunu öğrenebilir fakat
- bizim veri setimize bu hazır pretrained-token uygun değil kendi modelinde başarılı olabilir ama burada anlamlı tokenlar üretemedi
- bunun sebebi türkçe sondan eklemeli bir dil, kendi veri setimdeki kelimelerin çok bozuk olduğunu gördü
- Bir model içinde bütün kelimeleri böyle subword öğrenmek zor o yüzden,
- çok buyuk Türkçe veri setinde eğitilmemiş tokenlar hariç kendi Tokenizer()ımız kendimiz eğitmemiz,
- modelin başarısı için daha iyi olur
'''

converted = pretrained_tokenizer.convert_ids_to_tokens(tokens)
print(converted)



###### TRAIN A NEW TOKENIZER ######

# Bu fonksiyon list comprehension yapıyor
# List comprehension her bir batch_size kadarını bizim train datamızdan alacak, bize parça parça döndürecek
# training_corpus bizim için artık data generator olcak
def get_training_corpus():
    batch_size = 1000
    return (
        reviews_sample["train"][i : i + batch_size]["review"]
        for i in range(0, len(reviews_sample["train"]), batch_size)
    )
training_corpus = get_training_corpus()


# 1 tane review aldığımda 1000 tane review geliyor çünkü batch_size'ı 1000 yaptım yukarda
# toplamda 3378 tane gedi train'de bu kadar veri vardı çünkü geri kalanı validation'dı
for reviews in get_training_corpus():
    print(len(reviews))

'''
# şimdi Tokenizer()'ı yaratacğız bu Tokenizer()'ı Hugging Face ile entegre edeceğim için hem modeliyle hem data setiyle,
# bu yüzden Hugging Face'in kurallarına göre Tokenizer() yaratacağız
# Hugging Face'te her şeyi standart API'de kullacağım için Keras, Tensorflow veya Python'da değil Hugging Face'in Tokenizer'ını kullanacağım

# bizim veri seti ilk 4000 reviewdan oluşuyor bunun için 52000 vocab_size çok ama gerçek uygulamalarda bu vocab_size iyi

# sıfırdan Tokenizer yaratırken birçok yapmam gereken ayar var(kullacağım model ile ilgili)
# bu ayarlar için, GPT-2'de çalıştığını bildiğim ve indirdiğim pretrained_tokenizer'ın ayarlarını kullanacağım
# train_new_from_iterator: ona verdiğim data generator'ı kullanarak istediği vocab_size'a göre, benim elimdeki tokenizer'ın konfigürasyonuna(pretrained_tokenize) göre yeniden Tokenizer yaratacak
# bunun güzelliği biz bu Tokenizer'ı GPT-2'de kullanacağız ve orada subword ayarları var bunları yapmaktan kurtulmuş oluyorum

# vocab size: token sayısı demek
# !!!! bir kelime birden fazla token'a bölünebilir

vocab_size = 52000
tokenizer = pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)
'''

vocab_size = 52000
tokenizer = pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)

# end of string token, bunun sıfır olduğuna emin olmamız lazım
tokenizer.eos_token_id

# 44208 token elde edilmiş (ilk 4000 review için)
tokenizer.vocab_size

# yeni Tokenizer'ımızı kullanarak aynı metinde deneyelim
txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = tokenizer(txt)['input_ids']
print(tokens)

converted = tokenizer.convert_ids_to_tokens(tokens)
print(converted)

# pretrained_tokenizer aynı metni 19 token la gösterirken
# benim tokenizer'ım aynı metni 12 tokenla gösterir,
# böylece uygun sırada cümle üretmek daha kolay
print(len(tokenizer.tokenize(txt)))
print(len(pretrained_tokenizer.tokenize(txt)))


###### SAVING THE TOKENIZER ######

# Kendi Tokenizer'ımı eğittikten sonra localde saklayacağım
#path="./"
#file_name="turkishReviews-ds-mini"
#tokenizer.save_pretrained(path+file_name)

# Tokenizer'ımızı localden indirelim
loaded_tokenizer = AutoTokenizer.from_pretrained("./turkishReviews-ds-mini")

# Kendi Tokenizer'ımı eğittikten sonra Hugging Face Hub'a yükleyelim
# huggingface-cli login kodunu terminale yaz
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer.push_to_hub("imelike/turkishReviews-ds-mini")

# kendi ürettiğimiz Tokenizer()'ı Hugging Face Hub'dan indirelim
downloaded_tokenizer = AutoTokenizer.from_pretrained("imelike/turkishReviews-ds-mini")


# tokenizer'ımızın doğru inip-inmediğini kontrol edelim
# üçü de aynıysa doğru çalışıyor
txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = tokenizer(txt)['input_ids']
print("trained tokenizer:", tokens)
tokens = loaded_tokenizer(txt)['input_ids']   #locale yüklenen
print("loaded tokenizer:", tokens)
tokens = downloaded_tokenizer(txt)['input_ids']
print("downloaded tokenizer:", tokens)

