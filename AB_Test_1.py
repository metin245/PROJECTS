import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
# pip install statsmodels

import statsmodels.stats.api as sms

from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,  \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal

from statsmodels.stats.proportion import proportions_ztest
# GÖREV 1:  Veriyi Hazırlama ve Analiz Etme
# Adım 1:  ab_testing_data.xlsxadlı kontrol ve test grubu verilerinden oluşan
# veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız

df1 = pd.read_excel('C:/Users/hbbgn/PycharmProjects/measurement/ab_testing.xlsx',sheet_name='Control Group')
df2= pd.read_excel('C:/Users/hbbgn/PycharmProjects/measurement/ab_testing.xlsx',sheet_name='Test Group')



# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
df1.shape
df1.isnull().sum()
df1.describe().T

df2.shape
df2.isnull().sum()
df2.describe().T


# Adım 3: Analiz işleminden sonra concatmetodunu kullanarak kontrol ve test grubu
# verilerini birleştiriniz

df=pd.concat([df1,df2]).reset_index()

# GÖREV 2:  A/B Testinin Hipotezinin Tanımlanması
#H0 : M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
#H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

print ("Kontrol grubu ortalaması: %3f"%df1["Puchase"].mean(), \
    "Test grubu ortalaması: %3f"%df2["Purchase"].mean())
# GÖREV 3:  Hipotez Testinin Gerçekleştirilmesi

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# Varsayım Kontrolü
# 1. Normallik Varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.


test_stat, pvalue = shapiro(df1["Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p değeri 0.05 den büyük olduğu için HO reddedilemez.

test_stat, pvalue = shapiro(df2["Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
 # H0 reddedilemez

# 2. Varyans Homojenliği
test_stat, pvalue = levene((df1["Purchase"].dropna(), df2["Purchase"].dropna())

# homojendir.
# Varsayımlar ağlandığı için iki örneklem t testi (parametrik test)
test_stat,p_value=ttest_ind(df1["Purchase"],df2["Purchase"])
print("Test stat: %.4f  p-value= %.4f"%(test_stat,p_value))

# Ho: reddedilemez anlamlı bir farklılık yoktur.