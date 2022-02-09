
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.options.mode.chained_assignment = None
#pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder


from helpers.eda import *
from helpers.dataprep import *

df_ = pd.read_excel(r"C:\Users\MehmetCanYildirim\Desktop\Veri Bilimi Okulu\Dersler\9.Hafta\online_retail_II.xlsx")
df = df_.copy()
df.head()

check_df(df)

# Veri setinden null değerleri cıkaralım.
df.dropna(inplace=True)
df.isnull().any()


# Veri setindeki C ile başlayanları çıkaralım. (C iptal edilenleri göstermektedir.)
df = df[~df["Invoice"].str.contains("C",na=False)]
df.shape

# Toplam geliri ifade eden TotalPrice oluşturalım. Adet*Fiyat
df["TotalPrice"] = df["Quantity"] * df["Price"]

# RFM Metrikleri Oluşturma.

df["InvoiceDate"].max() # ('2010-12-09 20:01:00')
today_date = dt.datetime(2010,12,10)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
rfm.head()

rfm.columns = ["Recency","Frequency","Monetary"]
rfm = rfm[rfm["Monetary"]>0]

rfm.head()
rfm = rfm.reset_index()
rfm.head()

rfm_scaled = rfm[["Recency","Frequency","Monetary"]]
rfm_scaled.head()

# Monetary'deki büyük sayıların frequency ve recency'deki değerleri ezmesini engellemek için bu metrikleri scale edebiliriz.
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_scaled)
rfm_scaled = pd.DataFrame(rfm_scaled)
rfm_scaled.head()
rfm_scaled.columns = ["Recency","Frequency","Monetary"]


# K-Means

kmeans = KMeans(n_clusters=3)
k_fit = kmeans.fit(rfm_scaled)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
k_fit.inertia_

# Kümelerin Görselleştirilmesi

clusters = k_fit.labels_
centers = k_fit.cluster_centers_

plt.scatter(rfm_scaled.iloc[:, 0],
            rfm_scaled.iloc[:, 1],
            c=clusters,
            s=50,
            cmap="viridis")

plt.scatter(centers[:, 0],
            centers[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Optimum Küme Sayısını bulalım.
# Elbow yöntemi ile bulma.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfm_scaled)
elbow.show()

elbow.elbow_value_ # 6

# Final Model

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm_scaled)

rfm_scaled.loc[:,'Customer ID'] = rfm['Customer ID']
rfm_scaled.head()

rfm_scaled['cluster'] = kmeans.labels_
rfm_scaled['cluster'] = rfm_scaled['cluster'] + 1
rfm_scaled['cluster'].unique()

rfm_scaled[["Recency","Frequency","Monetary"]] = scaler.inverse_transform(rfm_scaled[["Recency","Frequency","Monetary"]])
rfm_with_Kmeans = rfm_scaled
rfm_with_Kmeans.head()

# Clusterların kaçar tane olduğu
rfm_with_Kmeans.groupby("cluster").agg({"cluster":"count"})

# Clusterlara göre RFM metriklerinin ortalama değerleri
rfm_with_Kmeans.groupby("cluster")["Recency","Frequency","Monetary"].mean()

# RFM Metriklerine göre 6 kümeye segmentlenen müşterilerin 3D Görselleştirilmesi.
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (25,25)
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig,
            rect = [0, 0, 0.95, 1],
            elev = 48,
            azim = 134)
plt.cla()
ax.scatter(rfm_with_Kmeans['Frequency'],rfm_with_Kmeans["Recency"], rfm_with_Kmeans["Monetary"],
           c = rfm_with_Kmeans['cluster'],
           s = 200,
           cmap = 'spring',
           alpha = 0.5,
           edgecolor = 'darkgrey')

ax.set_xlabel('Frequency', fontsize = 16)
ax.set_ylabel('Recency', fontsize = 16)
ax.set_zlabel('Monetary', fontsize = 16)

plt.show()






