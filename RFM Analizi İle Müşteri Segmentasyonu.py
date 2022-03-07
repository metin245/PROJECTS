
# Görev 1
# Adım 1: Online RetailII excelindeki2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’inkopyasını oluşturunuz.
import pandas as pd
df_ = pd.read_excel("C:/Users/hbbgn/PycharmProjects/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Adım 2: Veri setinin betimselistatistiklerini inceleyiniz.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df.describe().T
df.columns
df = df[(df['Quantity'] > 0)]

"""
Out[25]:
                count          mean          std  ...       50%       75%      max
Quantity     541910.0      9.552234   218.080957  ...      3.00     10.00  80995.0
Price        541910.0      4.611138    96.759765  ...      2.08      4.13  38970.0
Customer ID  406830.0  15287.684160  1713.603074  ...  15152.00  16791.00  18287.0
"""
# Adım3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?

df.isnull().sum()
"""
Out[27]:
Invoice             0
StockCode           0
Description      1454
Quantity            0
InvoiceDate         0
Price               0
Customer ID    135080
Country             0
dtype: int64

"""

# Adım4: Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.

df.dropna(inplace=True)
df.isnull().sum()
"""
Out[29]: 
Invoice        0
StockCode      0
Description    0
Quantity       0
InvoiceDate    0
Price          0
Customer ID    0
Country        0
dtype: int64

"""

# Adım5: Eşsiz ürün sayısı kaçtır?
df.head()
df["StockCode"].nunique()
df.shape

# Out[33]: 3684

# Adım6: Hangi üründen kaçar tane vardır?

df.groupby("StockCode").agg({"Quantity": lambda x:x.sum()})

# Adım7: En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız

df.groupby("StockCode").agg({'Invoice': lambda x: x.nunique()}).sort_values(by="Invoice", ascending=False).head()
"""
Out[61]: 
           Invoice
StockCode         
85123A        1978
22423         1704
85099B        1600
47566         1380
84879         1375
"""
# Adım 8: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.

df = df[~df['Invoice'].str.contains('C', na=False)]
df.describe().T


# Adım 9: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz


df['TotalPrice']= df['Quantity'] * df['Price']
df.groupby('Invoice').agg({'TotalPrice': lambda x:x.sum()})


"""
Out[67]: 
         TotalPrice
Invoice            
536365       139.12
536366        22.20
536367       278.73
536368        70.05
536369        17.85
             ...
581583       124.60
581584       140.64
581585       329.05
581586       339.20
581587       267.45
[18536 rows x 1 columns]
"""

# Görev 2:  RFM Metriklerinin Hesaplanması

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# Adım 2: Müşteri özelinde Recency, Frequencyve Monetarymetriklerini groupby, aggve lambdaile hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız
import datetime as dt

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate:(today_date - InvoiceDate.max()).days,
                               "Invoice": lambda num: num.nunique(),
                               "TotalPrice": lambda Total_Price:Total_Price.sum()})

# Adım4: Oluşturduğunuz metriklerin isimlerini  recency, frequency ve monetaryolarak değiştiriniz

rfm.columns=["recency","frequency","monetary"]
rfm[rfm['monetary'] >0]
"""
Out[79]: 
             recency  frequency  monetary
Customer ID                              
12346.0          326          1  77183.60
12347.0            3          7   4310.00
12348.0           76          4   1797.24
12349.0           19          1   1757.55
12350.0          311          1    334.40
              ...        ...       ...
18280.0          278          1    180.60
18281.0          181          1     80.82
18282.0            8          2    178.05
18283.0            4         16   2094.88
18287.0           43          3   1837.28
[4339 rows x 3 columns]
 """


# Görev 3:  RFM Skorlarının Oluşturulması ve Tek bir Değişkene Çevrilmesi

# Adım 1: Recency, Frequency ve Monetary metriklerini qcutyardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_scoreve monetary_scoreolarak kaydediniz.
# Adım 3: recency_scoreve frequency_score’utek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["RF_Score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


# Görev 4:  RF Skorunun Segment Olarak Tanımlanması
# Adım 2: Aşağıdaki seg_mapyardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
    }
rfm['segment'] = rfm['RF_Score'].replace(seg_map, regex=True)

"""
Out[85]: 
             recency  frequency  monetary recency_score monetary_score frequency_score RFM_Score              segment
Customer ID                                                                                                          
12346.0          326          1  77183.60             1              5               1        11          hibernating
12347.0            3          7   4310.00             5              5               5        55            champions
12348.0           76          4   1797.24             2              4               4        24              at_Risk
12349.0           19          1   1757.55             4              4               1        41            promising
12350.0          311          1    334.40             1              2               1        11          hibernating
              ...        ...       ...           ...            ...             ...       ...                  ...
18280.0          278          1    180.60             1              1               2        12          hibernating
18281.0          181          1     80.82             1              1               2        12          hibernating
18282.0            8          2    178.05             5              1               3        53  potential_loyalists
18283.0            4         16   2094.88             5              5               5        55            champions
18287.0           43          3   1837.28             3              4               4        34      loyal_customers
[4339 rows x 8 columns]
"""

# Görev 5:  Aksiyon Zamanı !
# Adım1:  Önemli gördüğünüz segmenti seçiniz. Bu üçsegmentihem aksiyon kararları açısından hemde segmentlerin yapısı açısından(ortalamaRFM değerleri) yorumlayınız.
# Adım2:  "LoyalCustomers" sınıfına ait customerID'leri seçerek excel çıktısını alınız.
rfm[['segment','recency','frequency','monetary']].groupby('segment').agg(['mean', 'count'])

"""
Out[30]: 
                        recency        frequency           monetary      
                           mean count       mean count         mean count
segment                                                                  
about_to_sleep        53.312500   352   1.161932   352   471.994375   352
at_Risk              153.785835   593   2.878583   593  1084.535297   593
cant_loose           132.968254    63   8.380952    63  2796.155873    63
champions              6.361769   633  12.417062   633  6857.963918   633
hibernating          217.605042  1071   1.101774  1071   488.643307  1071
loyal_customers       33.608059   819   6.479853   819  2864.247791   819
need_attention        52.427807   187   2.326203   187   897.627861   187
new_customers          7.428571    42   1.000000    42   388.212857    42
potential_loyalists   17.398760   484   2.010331   484  1041.222004   484
promising             23.421053    95   1.000000    95   290.913158    95

"""


# Champions
# Can't Loose
# Loyal Customers


rfm[rfm["segment"]== "cant_loose"].head()
rfm[rfm["segment"]== "cant_loose"].index

Loyal_df = pd.DataFrame()
Loyal_df["Loyal_Customer_Id"] = rfm[rfm["segment"] == "loyal_customers" ].index
Loyal_df["Loyal_Customer_Id"] = Loyal_df["Loyal_Customer_Id"].astype(int)
Loyal_df.to_excel('Loyal_Customers.xlsx')

