
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.options.mode.chained_assignment = None
#pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Reading Dataset
path = r"C:\Users\MehmetCanYildirim\Desktop\Veri Bilimi Okulu\Dersler\pythonProject\Country-data.csv"
df = pd.read_csv(path)
df.head()
df.shape


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# Determining categoric, cardinal and numeric columns according to given threshold.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

                Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
                Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

                Parameters
                ------
                    dataframe: dataframe
                            Değişken isimleri alınmak istenilen dataframe
                    cat_th: int, optional
                            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
                    car_th: int, optinal
                            kategorik fakat kardinal değişkenler için sınıf eşik değeri

                Returns
                ------
                    cat_cols: list
                            Kategorik değişken listesi
                    num_cols: list
                            Numerik değişken listesi
                    cat_but_car: list
                            Kategorik görünümlü kardinal değişken listesi

                Examples
                ------
                    import seaborn as sns
                    df = sns.load_dataset("iris")
                    print(grab_col_names(df))


                Notes
                ------
                    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
                    num_but_cat cat_cols'un içerisinde.
                    Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_but_car , num_cols

# Data Preprocessing
# Outliers
# Determining low and up limit according to given quantile values.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low_limit, up_limit = outlier_thresholds(df, num_cols)


# Outliers check.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) |
                 (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, num_cols)


# Changing outlier variables given threshold values.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit),
                  variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit),
                  variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for i in num_cols:
    print(i, check_outlier(df, i))

# Missing Values
# Querying missing values.
df.isnull().any()

# Scaling
# Scaling numerical columns so that they do not dominate.
df_scaled = df[["child_mort","exports","health","imports","income",
                "inflation","life_expec","total_fer","gdpp"]]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_scaled)
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = ["child_mort","exports","health","imports",
                     "income","inflation","life_expec",
                     "total_fer","gdpp"]
df_scaled.head()


# K-Means Clustering Method
# Creating first model with random cluster number which is 3 and fitting last dataframe.
kmeans = KMeans(n_clusters=3)
k_fit = kmeans.fit(df_scaled)

# n_clusters: give cluster number in the model.
# cluster_centers_: give center of each clusters.
# labels_: give labels of each point.
# inertia_: give sum of squared distances (SSD) of samples to their closest cluster center.

k_fit.n_clusters, k_fit.cluster_centers_ , k_fit.labels_, k_fit.inertia_

# Visualizing of model.
clusters = k_fit.labels_
centers = k_fit.cluster_centers_

plt.scatter(df_scaled.iloc[:, 0],
            df_scaled.iloc[:, 1],
            c=clusters,
            s=50,
            cmap="viridis")

plt.scatter(centers[:, 0],
            centers[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Finding optimal cluster number by using Elbow method.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df_scaled)
elbow.show()

elbow.elbow_value_ # 7

# Final Model with Optimum k cluster.
kmeans = KMeans(n_clusters=elbow.elbow_value_)
kmeans = kmeans.fit(df_scaled)

df_scaled.loc[:,'country'] = df['country']
df_scaled.head()

df_scaled['cluster'] = kmeans.labels_
df_scaled['cluster'] = df_scaled['cluster'] + 1
df_scaled['cluster'].unique()

# Inverse scaling for each numerical columns.
df_scaled[["child_mort","exports","health","imports","income",
           "inflation","life_expec","total_fer","gdpp"]] \
    = scaler.inverse_transform(df_scaled[["child_mort","exports",
                                          "health","imports",
                                          "income","inflation",
                                          "life_expec","total_fer",
                                          "gdpp"]])
df_scaled.head()

final_df = df_scaled
final_df.head()

# It gives that the total numbers of observation for each clusters.
final_df.groupby("cluster").agg({"cluster":"count"})

# It gives that the average numbers of columns for each clusters.
final_df.groupby("cluster").agg(np.mean)

final_df[final_df["cluster"] == 5]

# Visualizing Clusters

clusters = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(final_df.iloc[:, 0],
            final_df.iloc[:, 1],
            c=clusters,
            s=50,
            cmap="viridis")

plt.scatter(centers[:, 0],
            centers[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()
