import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
pd.set_option("display.max_columns",None)
 #pd.set_option("display.max_rows",None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x:  "%.5f" %x)

# veri setinin okutulması ve ürünün ortalama puanının hesaplanması

df = pd.read_csv("C:/Users/hbbgn/PycharmProjects/amazon_review.csv")
df.head()
df.shape

# Ortalama
df["overall"].mean()

# Tarihe göre ağırlıklı puan

a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

def time_based_weighted_avarage(dataframe,w1=28,w2=26,w3=24,w4=22):
   return dataframe.loc[dataframe["day_diff"] <= a, "overall"].mean() * w1 / 100 + \
    dataframe.loc[(dataframe["day_diff"] > a) & (dataframe["day_diff"] <= b), "overall"].mean() * w2 / 100 + \
    dataframe.loc[(dataframe["day_diff"] > b) & (dataframe["day_diff"] <= c), "overall"].mean() * w3 / 100 + \
    dataframe.loc[dataframe["day_diff"] > c, "overall"].mean() * w4 / 100

df.loc[df["day_diff"] <= a, "overall"].mean()
df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean()
df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean()
df.loc[df["day_diff"] > c, "overall"].mean()

time_based_weighted_avarage(df)

# GÖREV 2
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# score_pos_neg_diff
def score_up_down_diff(up, down):
   return up - down
df["score_up_down_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_up_down_diff",ascending=False).head(20)
# score_average_rating
def score_avarage_rating(up, down):
   if up + down == 0:
      return 0
   return up / (up + down)

df["score_avarage_rating"] = df.apply(lambda x: score_avarage_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_avarage_rating",ascending=False).head(20)
# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
   n = up + down
   if n == 0:
      return 0
   z = st.norm.ppf(1 - (1 - confidence) / 2)
   phat = 1.0 * up / n
   return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"]=df.apply(lambda x:wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

df.sort_values("wilson_lower_bound",ascending=False).head(20)