
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak


# 1. Veri Ön İşleme

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("Ödevler/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()
df.shape

from Utilities.data_preprocessing import retail_data_prep # Önceden yazdığımız fonksiyonları dışarıdan (Utilities'ten) import etme .
df = retail_data_prep(df)
df.head()
df.shape

# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)

df_gr = df[df["Country"] == "Germany"]
df_gr.head()

df_gr.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).head(50)

df_gr.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:15, 0:15] # Unstack descriptionu columnlara alır.

df_gr.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:15, 0:15] # Nan degerlerini 0 yaptık.

df_gr.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:15, 0:15] # Applymap ile tüm değerlerde gezerek 1'den büyük sayı varsa 1, yoksa 0 olarak kalsın.


def create_invoice_product_matrix(dataframe, stockcode=True): # Ürünlerin descriptionlarına göre invoice-product matrixi oluşturma.
    if stockcode:
        return dataframe.groupby(["Invoice", "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])['Quantity'].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0)

df_gr_matrix = create_invoice_product_matrix(df_gr)

def stockcode_check(dataframe, ID): # Dataframedeki StockCode ID'lerinin karşılığı olan descriptionları verir.
    description = dataframe[dataframe["StockCode"] == ID][["Description"]].values[0].tolist()
    print(description)

stockcode_check(df_gr, 21987) # ['PACK OF 6 SKULL PAPER CUPS']
stockcode_check(df_gr, 23235) # ['STORAGE TIN VINTAGE LEAF']
stockcode_check(df_gr, 22747) # ["POPPY'S PLAYHOUSE BATHROOM"]

# 3. Birliktelik Kurallarının Çıkarılması

# Her bir itemin support değerlerini elde ettik.
support_items = apriori(df_gr_matrix, min_support=0.01, use_colnames=True) # Min support alabilecek minimum support eşik değerini belirler.
support_items.sort_values("support", ascending=False).head(50)

# Confidence, lift değerlerini bulmak için yukarıda apriori ile bulduğumuz support değerlerini kullanarak association rule uyguladık.
rules = association_rules(support_items, metric='support', min_threshold=0.01)
rules.sort_values("support", ascending=False).head(50)

# antecedents: Önceki Ürün
# consequents: Sonraki Ürün
# antecedent support: Önceki ürünün tek başına gözükme olasılığı
# consequent support: Sonraki ürünün tek başına gözükme olasılığı
# support: Önceki ve Sonraki ürünün birlikte gözükme olasılığı
# confidence: Önceki ürün alındığında, sonraki ürünün alınma olasılığı
# lift: Önceki ürün alındığında, sonraki ürünün alınma olasılığının kaç kat arttığını gösteren metriktir.

# 4. Çalışmanın Scriptini Hazırlama

pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

def create_invoice_product_matrix(dataframe, stockcode=True): # Ürünlerin descriptionlarına göre invoice-product matrixi oluşturma.
    if stockcode:
        return dataframe.groupby(["Invoice", "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])['Quantity'].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0)

def stockcode_check(dataframe, ID): # Dataframedeki StockCode ID'lerinin karşılığı olan descriptionları verir.
    description = dataframe[dataframe["StockCode"] == ID][["Description"]].values[0].tolist()
    print(description)

def create_rules(dataframe, stockcode=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_matrix(dataframe, stockcode)
    support_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(support_items, metric="support", min_threshold=0.01)
    return rules

df_ = pd.read_excel("Ödevler/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

sorted_rules = rules.sort_values("lift", ascending=False)

# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

# Sepetteki bir ürünü seçelim.
StockCode = 22746
stockcode_check(df_gr,StockCode)

sorted_rules.head(50)

stockcode_check(df,22747) # ["POPPY'S PLAYHOUSE BATHROOM"]
stockcode_check(df,22745) # ["POPPY'S PLAYHOUSE BEDROOM "]
stockcode_check(df,22748) # ["POPPY'S PLAYHOUSE KITCHEN"]


# Recommendation için script yazalım.

def arl_recommender(rules_df, StockCode, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendations = []

    for k, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == StockCode:
                recommendations.append(list(sorted_rules.iloc[k]["consequents"]))

    recommendations = list(dict.fromkeys({item for item_list in recommendations for item in item_list})) # Tekrar eden itemleri tekilleştirme.
    return recommendations[:rec_count]


# İlk Önerme
stockcode_check(df_gr, 21987) # ['PACK OF 6 SKULL PAPER CUPS']
arl_recommender(rules, 21987, 3) # [23050, 22029, 20750]
stockcode_check(df_gr, 23050) # ['RECYCLED ACAPULCO MAT GREEN']
stockcode_check(df_gr, 22029) # ['SPACEBOY BIRTHDAY CARD']
stockcode_check(df_gr, 20750) # ['RED RETROSPOT MINI CASES']

# İkinci Önerme
stockcode_check(df_gr, 22716) # ['CARD CIRCUS PARADE']
arl_recommender(rules, 22716, 2) # 22029, 22037
stockcode_check(df_gr, 22029) # ['SPACEBOY BIRTHDAY CARD']
stockcode_check(df_gr, 23205) # ['CHARLOTTE BAG VINTAGE ALPHABET ']

stockcode_check(df_gr, 22549) # ['PICTURE DOMINOES']
arl_recommender(rules, 22549, 3) # 21668, 21671, 22728
stockcode_check(df_gr, 21668) # ['RED STRIPE CERAMIC DRAWER KNOB']
stockcode_check(df_gr, 21671) # ['RED SPOT CERAMIC DRAWER KNOB']
stockcode_check(df_gr, 22728) # ['ALARM CLOCK BAKELIKE PINK']
