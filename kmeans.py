import pandas as pd
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

###############################################################
# RFM
###############################################################

def create_rfm(dataframe):
    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    return rfm


rfm = create_rfm(df)
rfm.head()

###############################################################
# K-Means Clustering
###############################################################

# Min - Max Scaler
scaler = MinMaxScaler()
segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])
segment_data.head()

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(segment_data)
elbow.show()
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(segment_data)
segment_data["clusters"] = kmeans.labels_
print(f"Number of cluster selected: {elbow.elbow_value_}")


################################
# Final Cluster'ların Oluşturulması
################################

segment_data = pd.DataFrame(scaler.fit_transform(rfm[["recency", "frequency", "monetary"]]),
                            index=rfm.index, columns=["Recency_n", "Frequency_n", "Monetary_n"])

kmeans = KMeans(n_clusters=6).fit(segment_data)
segment_data["clusters"] = kmeans.labels_



################################
# RFM ve K-Means Clusterlarının Birleştirilmesi
################################
segmentation = rfm.merge(segment_data, on="Customer ID")
seg_desc = segmentation[["segment", "clusters", "recency", "frequency", "monetary"]].groupby(["clusters", "segment"]).agg(["mean", "count"])
print(seg_desc)
segmentation.to_csv("segmentation.csv")

# Detaylı analiz için:
# https://datastudio.google.com/reporting/a7baf3c6-832e-4f03-b129-1bc5bd3919be/page/tvWVC


