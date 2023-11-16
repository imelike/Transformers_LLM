'''
**PART A:** Prepare a 🤗 Hugging Face Dataset from Data in CSV Format

- **TERMINALDEN CALISTIRILACAK**

- git config --global user.email "melikeakalan1@gmail.com"
- git config --global user.name "imelike"
- huggingface-cli login


- **VERI SETINI INDIRME**
aş. kmuatları terminalden sırasıyla çalıştır

- cd input terminalden çalıştır bunu
- kaggle datasets download -d savasy/multiclass-classification-data-for-turkish-tc3
- dir
- tar -xf multiclass-classification-data-for-turkish-tc32.zip
- del C:\ADEN\GITHUB-PROJECTS\Transformers_LLM\input\multiclass-classification-data-for-turkish-tc32.zip
-
- **Veri setini dataFrame'e dönüştürelim**
- data = pd.read_csv('ticaret-yorum.csv')
- pd.set_option('max_colwidth', 100)
- data.head(5)

'''
import os
import re
import pandas as pd
import tensorflow as tf

file_name = "ticaret-yorum.csv"
path = "C:\ADEN\GITHUB-PROJECTS\Transformers_LLM\input\\"

from datasets import load_dataset
reviews_dataset = load_dataset("csv", data_files= path+file_name)

reviews_dataset

reviews_dataset['train'][:2]

# tüm veri seti shuffle yapılır
# ilk 4000 yorumla transfomerlar eğitilecek(süre uzamasın diye)

reviews_sample = reviews_dataset["train"].shuffle(seed=42).select(range(4000))
reviews_sample

"""
 bu veri seti metin sınıflandırma için yapılmış o yüzden category değişkeni var
 ama biz bu verisetiyle dil modeli eğitip, metin üretimi yapacağız
 bu değişken de lazım değil
"""
reviews_sample = reviews_sample.remove_columns('category')
reviews_sample


reviews_sample = reviews_sample.rename_column(
    original_column_name="text", new_column_name="review"
)
reviews_sample

def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

reviews_sample = reviews_sample.map(compute_review_length)
# Inspect the first training example
reviews_sample[0]

reviews_sample.sort("review_length")[:3]

########## ONEMLI #################

# Duygu analizinde kısa cümleler de model için etkili olur ama
# Dil modeli eğitimini kötü etkileyebileceğinden 30 dan az kelime içeren yorumlar silinir
# 250 yorum silindi

reviews_sample = reviews_sample.filter(lambda x: x["review_length"] > 30)
print(reviews_sample.num_rows)

reviews_sample[:3]

# Veriden bir sample aldığımızda ..., Devamını oku gibi tekrar eden anlamsız tokenları siliyoruz
def remove_repeated(example):
    example["review"] = example["review"].replace('...', '')
    example["review"] = example["review"].replace(',"', '. ')
    example["review"] = example["review"].replace('!.', '.')
    example["review"] = example["review"].replace('!,', '. ')
    example["review"] = example["review"].replace('"', '')
    example["review"] = re.sub('([a-zA-Z0-9zığüşöçZİĞÜŞÖÇ]),([a-zA-Z0-9zığüşöçZİĞÜŞÖÇ])', '\\1. \\2', example["review"])

    return {"review": example["review"].replace('Devamını oku', '')}


# review_sample'da bulunan her bir yoruma yukarıdaki işlem uygulanır

reviews_sample = reviews_sample.map(remove_repeated)
reviews_sample[:3]


###### VALIDATION SETI OLUSTURMA #########
# Veri setimizde şuan sadece train var bunu train ve test olarak ayırmak istiyoruz
# ama model eğitirken teste ihtiyacım yok validation'a ihtiyacım var
# test kolonunu validation olarak isimlendiriyorum

reviews_sample = reviews_sample.train_test_split(train_size=0.9, seed=42)
# Rename the default "test" split to "validation"
reviews_sample["validation"] = reviews_sample.pop("test")

# bu bir sözlük, bu sözlükte train ve val o.ü iki ayrı dataset var
reviews_sample

for key in reviews_sample["train"][0]:
    print(f"{key.upper()}: {reviews_sample['train'][0][key]}")


# Şuan üzerinde Dil Modeli eğiteceğim veri setim hazır bunu Hugging Face Hub'a yükleyeceğim
reviews_sample.push_to_hub("imelike/turkishReviews-ds-mini")

# Yukarıda yükledğim veri setini Hugging Face Hub'dan indireceğim
downloaded_dataset = load_dataset("imelike/turkishReviews-ds-mini")
downloaded_dataset

