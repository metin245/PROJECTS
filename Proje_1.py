#################################################
# WIKI 1 - Metin önişleme ve Görselleştirme (NLP - Text Preprocessing & Text Visualization)
#################################################

###################f##############################
# Problemin Tanımı
#################################################
# Wikipedia örnek datasından metin ön işleme, temizleme işlemleri gerçekleştirip, görselleştirme çalışmaları yapmak.

#################################################
# Veri Seti Hikayesi
#################################################
# Wikipedia datasından alınmış metinleri içermektedir.

#################################################
# Gerekli Kütüphaneler ve ayarlar



# https://zeyrek.readthedocs.io/en/latest/


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# Datayı okumak
df = pd.read_csv("datasets/wiki200k.csv", index_col=0)
df.head()
df.shape
#################################################
# Görevler:
#################################################

# Görev 1: Metindeki ön işleme işlemlerini gerçekleştirecek bir fonksiyon yazınız.
# •	Büyük küçük harf dönüşümünü yapınız.
# •	Noktalama işaretlerini çıkarınız.
# •	Numerik ifadeleri çıkarınız.

def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    # Numbers
    text = text.str.replace('\d', '')
    return text

df["text"] = clean_text(df["text"])

# Görev 2: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri (ben, sen, de, da, ki ile vs) çıkaracak fonksiyon yazınız.

def remove_stopwords(text):
    stop_words = stopwords.words('turkish')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])

df.head()

# Görev 3: Metinde az tekrarlayan (1000'den az, 2000'den az gibi) kelimeleri bulunuz.

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]


# Görev 4: Metinde az tekrarlayan kelimeleri metin içerisinden çıkartınız. (İpucu: lambda fonksiyonunu kullanınız.)

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))


# Görev 5: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words)

# Görev 6: Lemmatization işlemini yapınız.

# df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Görev 7: Metindeki terimlerin frekanslarını hesaplayınız. (İpucu: Barplot grafiği için gerekli)

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# Görev 8: Barplot grafiğini oluşturunuz.

# Sütunların isimlendirilmesi
tf.columns = ["words", "tf"]
# 5000'den fazla geçen kelimelerin görselleştirilmesi
tf[tf["tf"] > 5000].plot.bar(x="words", y="tf")
plt.show()

# •	Kelimeleri WordCloud ile görselleştiriniz.

# kelimeleri birleştirdik
text = " ".join(i for i in df["text"])

# wordcloud görselleştirmenin özelliklerini belirliyoruz
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 9: Tüm aşamaları tek bir fonksiyon olarak yazınız.
# •	Metin ön işleme işlemlerini gerçekleştiriniz.
# •	Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# •	Fonksiyonu açıklayan 'docstring' yazınız.

df = pd.read_csv("datasets/wiki200k.csv", index_col=0)


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemleri yapar.

    :param text: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('turkish')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Terim Frekanslarının Hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sütunların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geçen kelimelerin görselleştirilmesi
        tf[tf["tf"] > 5000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimeleri birleştirdik
        text = " ".join(i for i in text)
        # wordcloud görselleştirmenin özelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)
