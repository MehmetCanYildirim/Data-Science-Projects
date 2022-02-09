
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Import Dataset

def load_df():
    df = pd.read_csv(r"C:\Users\MehmetCanYildirim\Desktop\Veri_Bilimi_Okulu\Dersler\6.Hafta\Ödevler\Ödevler\diabetes.csv")
    return df

df = load_df()
df.head()


# Genel resimi görme

df.describe().T
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


# Kategorik Değişkenlerin Analizi

cat_cols = [col for col in df.columns if df[col].dtypes == "O"] # Kategorik değişken yok.


# Numerik Değişkenlerin Analizi

num_cols = [col for col in df.columns if df[col].dtypes != 'O']

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Outcome"]]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True)


# Kategorik ve Numerik Değişkenlerin Genelleştirilmesi

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
        car_th: int, optional
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

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Hedef Değişkenin Sayısal Değişkenler ile Analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Outliers

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95): # Aykırı değişkenler
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low, up = outlier_thresholds(df,"Insulin")

def check_outlier(dataframe, col_name): # Aykırı değer var mı yok mu ? Check
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# Outliers değerlerine ulaşalım.

def grab_outliers(dataframe, col_name, index=False): # Outliers'ları verir.
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

insulin_index = grab_outliers(df,"Insulin", True) # 13, 228


# Bu değerleri baskılama yöntemi kullanarak threshold değerlerine atayalım.

def replace_with_thresholds(dataframe, variable): # Aykırı değerleri eşik değere atayalım.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col, check_outlier(df,col))


# Çok Değişkenli Aykırı Değer Analiz

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[10] # Grafikten bakarak 10 olabileceğini öngördüm.

df[df_scores < th].shape # 10 adet değişken aykırı gözüküyor. Az sayıda oldugu icin silmek bir yöntem olabilir.

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df.shape

df.drop(axis=0, labels=df[df_scores < th].index, inplace=True) # Sildik.

df.tail()


# Eksik değerler

df.isnull().values.any()


# Korelasyon Analizi

cor_matrix = df.corr().abs()

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)

drop_list = high_correlated_cols(df)

high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


# Eksik veri analizi

# Eksik değer sıfır çıktı ama insulin değişkeni içinde sıfırlar var ve bir insanın insülini sıfır cıkması imkansız.
df[df["Insulin"] == 0].shape # 368 adet

df["Insulin"].shape

df.loc[df["Insulin"] == 0, ["Insulin"]] = np.nan

df["Insulin"].unique()


# Eksik olarak atadığımız değerleri gözlemleyip, bunları düzenleyelim.

df.isnull().values.any() # Hayır gözlemlendi çünkü verisetindeki insulin değişkeni object oldu. Bunu numeriğe dönüştürelim

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False) # Yüzdelik olarak eksiklikler ne kadar ?

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns # # Verisetinde kaç eksiklik var, bunların yüzdeliği ne, hangi columnlar'da eksiklik var

missing_values_table(df)

# Veri setinde NaN değerleri Insulin değişkenin yaklaşık yarısına denk geliyor,
# Mean ile doldurulabilir.

df["Insulin"].describe(percentiles=[0.01,0.05, 0.1,0.2,0.3,0.4,0.6,0.7,.8,.9,.95,.99])

df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0) # Ortalama ile doldurma

missing_values_table(df) # None

df.head()


# Feature Engineering


df.groupby([df["Age"] > 35]).agg({"Pregnancies" : "count"})

df.loc[(df["Pregnancies"] >= 10) & (df["Age"] > 35), "Age_Pregnancies_Count"] = "risk_9"
df.loc[((df["Pregnancies"] >= 3) & (df["Pregnancies"] < 10)) & (df["Age"] > 35), "Age_Pregnancies_Count"] = "risk_8"
df.loc[(df["Pregnancies"] < 3) & (df["Age"] > 35), "Age_Pregnancies_Count"] = "risk_7"

df.loc[(df["Pregnancies"] >= 10) & ((df["Age"] > 25) & (df["Age"] <= 35)), "Age_Pregnancies_Count"] = "risk_6"
df.loc[((df["Pregnancies"] >= 3) & (df["Pregnancies"] < 10)) & ((df["Age"] > 25) & (df["Age"] <= 35)), "Age_Pregnancies_Count"] = "risk_5"
df.loc[(df["Pregnancies"] < 3) & ((df["Age"] > 25) & (df["Age"] <= 35)), "Age_Pregnancies_Count"] = "risk_4"

df.loc[(df["Pregnancies"] >= 10) & (df["Age"] <= 25), "Age_Pregnancies_Count"] = "risk_3"
df.loc[((df["Pregnancies"] >= 3) & (df["Pregnancies"] < 10)) & (df["Age"] <= 25), "Age_Pregnancies_Count"] = "risk_2"
df.loc[(df["Pregnancies"] < 3) & (df["Age"] <= 25), "Age_Pregnancies_Count"] = "risk_1"

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Oluşturmuş oldugumuz kategorik değişkenlerin frekansına bakalım.

for col in cat_cols:
    cat_summary(df, col)

df.head()

# One Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df,ohe_cols)

# Standartlaştırma
mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])
df.head()


# Model
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

rf_model.feature_importances_

X_train.columns


# sns.pairplot
