pip install mlxtend

# veri ön işleme
import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.width', 500)

# çıktının tek bir satırda olmasını sağlar
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
df_ = pd.read_excel("C:/Users/hbbgn/PycharmProjects/recommadation/online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()
df.columns
df.info()
df.head()
df.describe().T

df.isnull().sum()
df[df['StockCode'] == 'POST']
df.drop(df.index[df['StockCode'] == 'POST'], inplace=True)

df.shape
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")
    return dataframe
df = retail_data_prep(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 -1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


df_Gr = df[df['Country'] == "Germany"]
df_Gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
df_Gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_Gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_Gr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda  x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_product_df(dataframe, id= False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x : 1 if x > 0 else 0)

Gr_inv_pro_df = create_invoice_product_df(df_Gr, id=True)

frequent_itemsets = apriori(Gr_inv_pro_df, min_support=0.01,use_colnames=True)

frequent_itemsets.sort_values("support",ascending=False)
rules = association_rules(frequent_itemsets, metric= "support", min_threshold=0.01)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 21086)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)

sorted_rules = rules.sort_values("lift", ascending=False)

product_id = 22492
check_id(df_Gr,product_id)

recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])



recommendation_list[0]

def arl_recommender(rules_df,product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]
arl_recommender(rules, 22492, 1)