# CLTV Prediction By using BG-NBD and Gamma Gamma

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Sales Forecasting
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması
# 7. Sonuçların Veri Tabanına Gönderilmesi

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency

# Gerekli Kütüphane ve Fonksiyonlar

# pip install lifetimes
# pip install sqlalchemy
# conda install -c anaconda mysql-connector-python
# conda install -c conda-forge mysql


from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Verinin Veri Tabanından Okunması

# Group - 2
# database: group_2
# user: group_2
# password: miuul
# host: 34.79.73.237
# port: 3306


# credentials.
creds = {'user': 'group_2',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_2'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)


pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)


retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()

df = retail_mysql_df.copy()


# Veri Ön İşleme


# Ön İşleme Öncesi
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

today_date = dt.datetime(2011, 12, 11)


# Lifetime Veri Yapısının Hazırlanması


# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] # satın alma başına ortalama kazanç!!

cltv_df = cltv_df[(cltv_df['frequency'] > 1)] # Frequency 1'den büyük olanları almak gerekir.

cltv_df = cltv_df[cltv_df["monetary"] > 0] # Monetary değerinin 0'dan büyük olması gerekir.

cltv_df["recency"] = cltv_df["recency"] / 7 # Haftalık bazında hesap yapıldığı için günlük değeri 7'ye bölüyoruz.

cltv_df["T"] = cltv_df["T"] / 7 # Haftalık bazında hesap yapıldığı için günlük değeri 7'ye bölüyoruz.

cltv_df.head()


# BG-NBD Modelinin Kurulması


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# BG-NBD Modelinin parametre değerleri bulundu.

# 6 ay içinde en çok satın alma beklediğimiz müşteriler

cltv_df["expected_purc_6_months"] = bgf.predict(4*6,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])



# 6 Ay içinde tüm Şirketin Beklenen Satış Sayısı

bgf.predict(4*6,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# Tahmin Sonuçlarının Değerlendirilmesi

plot_period_transactions(bgf)
plt.show()


# GAMMA-GAMMA Modelinin Kurulması


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# Gamma gamma modelinin parametreleri bulundu.

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])



# BG-NBD ve GG modeli ile CLTV'nin hesaplanması. (6 aylık CLTV Prediction)


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_clv", ascending=False).head()

# Farklı Zaman periyotlarından oluşan CLTV Analizi
# 1 aylık CLTV Analizi
cltv_1_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

# 12 aylık CLTV Analizi
cltv_12_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_1_month.sort_values(ascending=False).head(10)
cltv_12_month.sort_values(ascending=False).head(10)

# (?)(?)(?)(?)(?)(?)
# 2 CLTV değerleri arasında bir fark görülmedi. Çünkü BGNBD ve GAMMA GAMMA modellerinin parametreleri bir kitleye göre oluşturuldu ve hala o kitledeyiz.
# Bu yüzden bu kitle üzerinden kişi bazında oluşturulan CLTV değerleri bazında bir sıralama yapıldığında müşterilerin değer sıralamaları aynı olacaktır.
# Bu nedenle sıralamada bir değişiklik görülmesi beklenemez. Ancak tabiki 1 aylık ve 12 aylık periyotlardaki CLTV değerleri farklı olacaktır.


# CLTV'ye Göre Segmentlerin Oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

# Öncelikle ayırmış olduğumuz segmentlere baktığımız zaman A ve B'nin öncü olduğunu yani bizim için cltv değerlerinin yüksek olduğunu görebiliriz.
# Bu yüzden eğer bu segmentteki müşterilerimize odaklanacak olursak bu müşterilerimiz zaten bizler için en değerli kategoride olan müşterilerdir
# ve bizim onlara, onların da bizlere güvenerek bazı paylaşımlar yapmamız onları bize daha sadık bir müşteri haline getirebilir.
# Örnek vermek gerekirse müşterilerin geri bildirimlerine açık olup onların söylemlerine göre aksiyon almak bu noktada oldukça faydalı olacaktır.
# Başka bir örnek verecek olursak, müşteri ilişkilerinde hızlı ve kolay destekler sağlamak bu konuda bir önem arz edebilir.
# Çünkü müşteri bizle iletişime geçtiği vakit, müşteri bizden bir aksiyon beklemeye başlar ve bu süre arttıkça olumsuzluklar da artabilir.

# Ayrıca D segmenti görüldüğü üzere CLTV değerleri bizim için en düşük olan müşterilerimiz. Bunları bir üst segmente geçirmek amacımız olabilir.
# Yani CLTV değerlerinin artmasını sağlayabiliriz. Bunun için bir aksiyon alacak olursak da bu segmentteki müşterilerimize müşteri sadakat programı (customer loyalty program) uygulayabiliriz.
# Böylece müşterilerimizi elde tutabilir ve onların değerlerini arttırabiliriz. Müşteri sadakat programı olarak uygulanabilecek aksiyonlardan birisi müşterilerimizin
# upsell ya da cross-sell yapmalarını sağlamak olabilir. Bunun için müşterilere indirimler, hediye kartları vs gibi uygulamalar düzenlenebilir.


# Sonuçların Veri Tabanına Gönderilmesi

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

cltv_final.head()

cltv_final.to_sql(name='mehmetcan_yildirim', con=conn, if_exists='replace', index=False)
