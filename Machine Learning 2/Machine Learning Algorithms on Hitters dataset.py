# Gerekli Kütüphanelerin Yüklenmesi.

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor

import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet,Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from helpers.dataprep import *
from helpers.eda import *


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Ders Öncesi Notlar/hitters.csv")
df.head()

# Genel Resim
check_df(df)
# Kategorik, numerik ve cardinal columnlar
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Kategorik columnların analizi
for col in cat_cols:
    cat_summary(df,col)

# Numerik değişkenlerin analizi
for col in num_cols:
    num_summary(df,col)

sns.distplot(df.Salary)
plt.show()

print(df.shape)
df = df[(df['Salary'] < 1350) | (df['Salary'].isnull())]  # Eksik değerleri de istiyoruz.
print(df.shape)

sns.distplot(df.Salary)
plt.show()

# High correlated columns
high_correlated_cols(df)
target_correlation_matrix(df)

# Missing Values (Ağaç temelli modellerde bakılmasına gerek yok.)
na_columns = missing_values_table(df, na_name=True)

# Outliers Analizi ve baskılama işlemi (Ağaç temelli modellerde yapılmasına gerek yok.)
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


def feature_eng_for_hitters(df):
    df.columns = [col.upper() for col in df.columns]

    # 86-87 sezonundaki yapmış olduğu oyunların kariyerlerindeki yeri.
    df["ATBAT_DIV_CAREER"] = df["ATBAT"] / df["CATBAT"]
    df["HITS_DIV_CAREER"] = df["HITS"] / df["CHITS"]
    df["HMRUN_DIV_CAREER"] = df["HMRUN"] / df["CHMRUN"]
    df["RUNS_DIV_CAREER"] = df["RUNS"] / df["CRUNS"]
    df["RBI_DIV_CAREER"] = df["RBI"] / df["CRBI"]
    df["WALKS_DIV_CAREER"] = df["WALKS"] / df["CWALKS"]

    # 86-87 sezonunda yaptığı isabetli vuruşların tümüne oranı.
    df["HITS_DIV_ATBAT"] = df["HITS"] / df["ATBAT"]
    # 86-87 sezonunda yaptığı en değerli vuruşların tümüne oranı.
    df["HMRUN_DIV_ATBAT"] = df["HMRUN"] / df["ATBAT"]
    # 86-87 sezonunda yaptığı vuruşların ne kadarını takıma sayı olarak kazandırdığı.
    df["RUNS_DIV_ATBAT"] = df["RUNS"] / df["ATBAT"]
    # 86-87 sezonunda yaptığı vuruşların ne kadarı karşı oyuncuyu hataya sürükledi.
    df["WALKS_DIV_ATBAT"] = df["WALKS"] / df["ATBAT"]

    # Oyuncunun oynadığı yıl başına kariyerindeki toplam vuruş sayısı
    df["CATBAT_DIV_YEARS"] = df["CATBAT"] / df["YEARS"]
    # Oyuncunun oynadığı yıl başına kariyerindeki toplam isabet sayısı
    df["CHITS_DIV_YEARS"] = df["CHITS"] / df["YEARS"]
    # Oyuncunun oynadığı yıl başına kariyerindeki toplam en değerli vuruş sayısı
    df["CHMRUN_DIV_YEARS"] = df["CHMRUN"] / df["YEARS"]
    # Oyuncunun oynadığı yıl başına kariyerindeki toplam takıma kazandırdığı sayı
    df["CRUNS_DIV_YEARS"] = df["CRUNS"] / df["YEARS"]
    # Oyuncunun oynadığı yıl başına kariyerindeki toplam karşı takıma hata yaptırdığı sayı
    df["CWALKS_DIV_YEARS"] = df["CWALKS"] / df["YEARS"]

    #  Oyuncunun tüm sezonlar yaptığı isabetli vuruşların tümüne oranı.
    df["CHITS_DIV_ATBAT"] = df["CHITS"] / df["CATBAT"]
    #  Oyuncunun tüm sezonlar yaptığı en değerli vuruşların tümüne oranı.
    df["CHMRUN_DIV_ATBAT"] = df["CHMRUN"] / df["CATBAT"]
    #  Oyuncunun tüm sezonlar yaptığı vuruşların ne kadarını takıma sayı olarak kazandırdığı.
    df["CRUNS_DIV_ATBAT"] = df["CRUNS"] / df["CATBAT"]
    # Oyuncunun tüm sezonlar yaptığı vuruşların ne kadarı karşı oyuncuyu hataya sürükledi.
    df["CWALKS_DIV_ATBAT"] = df["CWALKS"] / df["CATBAT"]

    # Player junior, middle, senior durumu.
    df.loc[(df["YEARS"] <= 3), "NEW_YEARS_LEVEL"] = "Junior_player"
    df.loc[(df["YEARS"] > 3) & (df["YEARS"] <= 7), "NEW_YEARS_LEVEL"] = "Middle_player"
    df.loc[(df["YEARS"] > 7), "NEW_YEARS_LEVEL"] = "Senior_player"

    # E ve W (East ve Weast), playerları buna göre sınırlayabiliriz.

    df.loc[(df["NEW_YEARS_LEVEL"] == "Junior_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Junior_player_East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Junior_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Junior_player_West"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Middle_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Middle_player_East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Middle_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Middle_player_West"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Senior_player") & (df["DIVISION"] == "E"), "NEW_DIV"] = "Senior_player_East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Senior_player") & (df["DIVISION"] == "W"), "NEW_DIV"] = "Senior_player_West"

    # Oyuncuların 86-87 sezonunuda oynadıkları lig ile yeni sezonda oynadıkları lig karşılaştırılarak, lig yükselip düşme durumlarını öğrenebiliriz.
    # A american ligi ve junior olarak, N ise National ligi ve senior olarak bilinir.
    df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_SEASON_PROGRESS"] = "Same-A"
    df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_SEASON_PROGRESS"] = "Same-N"
    df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_SEASON_PROGRESS"] = "Higher"
    df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_SEASON_PROGRESS"] = "Lower"

    # 86-87 sezonunda oyunucun yapmış olduğu vuruş başına asist oranı.
    df["ASSISTS_DIV_ATBAT"] = df["ASSISTS"] / df["ATBAT"]
    # 86-87 sezonunda oyunucun yapmış olduğu vuruş başına hata oranı.
    df["ERRORS_DIV_ATBAT"] = df["ERRORS"] / df["ATBAT"]
    return df

df = feature_eng_for_hitters(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#
# missing_values_table(df)
#
# df["HMRUN_DIV_CAREER"] = df["HMRUN_DIV_CAREER"].fillna(df["HMRUN_DIV_CAREER"].mean())
# df["RBI_DIV_CAREER"] = df["RBI_DIV_CAREER"].fillna(df["RBI_DIV_CAREER"].mean())
# df["WALKS_DIV_CAREER"] = df["WALKS_DIV_CAREER"].fillna(df["WALKS_DIV_CAREER"].mean())
#
# missing_values_table(df)
#
#
# for col in num_cols:
#     print(col, check_outlier(df, col))
#
# for col in num_cols:
#     replace_with_thresholds(df, col)
#
# for col in num_cols:
#     print(col, check_outlier(df, col))
#
#

# Rare Encoding
for col in cat_cols:
    cat_summary(df,col)

rare_analyser(df,'SALARY', cat_cols)

# Gözlendiği gibi rare encoding yapmaya gerek kalmadı, çünkü yeni oluşturulan değişkenlerde küçük oranlara sahip değerler yok.

# One Hot Encoding

df = one_hot_encoder(df,cat_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("SALARY") # Salary target değişkenimizi numerik columnlardan cıkaralım.
# Son kez yeni oluşturdugumuz değişkenlerin rare olma durumlarına bakalım ve gereksiz olanları verisetinden cıkarabiliriz.
rare_analyser(df,'SALARY',cat_cols)

# Değişkenlerimizin değerlerini scale edelim.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform((df[num_cols]))

# Modelleri Kuralım.
df.dropna(inplace=True)
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)


##########################
# BASE MODELS
##########################


def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X, y, test_size=0.2, random_state=46, classification=False)

##########################
# RANDOM FORESTS MODEL TUNING
##########################

rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

# rf_cv = GridSearchCv(rf_model, rf_params, cv=5, n_jobs=-1).fit(X_train,y_train)
# best_params = rf_cv.best_params_

best_params = {'max_depth': 5,
               'max_features': 6,
               'min_samples_split': 12,
               'n_estimators': 80}

rf_tuned_model = RandomForestRegressor(max_depth=5,
                                       max_features=6,
                                       min_samples_split=12,
                                       n_estimators=80,
                                       random_state=12).fit(X_train,y_train)

y_pred_train = rf_tuned_model.predict(X_train)
y_pred_test = rf_tuned_model.predict(X_test)

print('RF Tuned Model Train RMSE:', np.sqrt(mean_squared_error(y_train,y_pred_train))) # 134.51567847910465
print('RF Tuned Model Train RMSE:', np.sqrt(mean_squared_error(y_test,y_pred_test))) # 170.91719599939577

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_tuned_model,X)


# Başka bir Model ile deneyelim

lgbm_model = lightgbm.LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.02, 0.03, 0.1, 0.001],
               "n_estimators": [100, 250, 300, 350, 500, 1000],
               "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1]}

lightgbm_cv = GridSearchCV(lgbm_model,
                           lgbm_params,
                           cv=5,
                           n_jobs=-1).fit(X_train,y_train)

lightgbm_cv.best_params_
#{'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}

lgbm_tuned_model = lightgbm.LGBMRegressor(colsample_bytree=0.5,learning_rate=0.01,n_estimators=500,random_state=12).fit(X_train,y_train)

y_pred_train = lgbm_tuned_model.predict(X_train)
y_pred_test = lgbm_tuned_model.predict(X_test)

print("LGBM Tuned Model Train RMSE :", np.sqrt(mean_squared_error(y_train,y_pred_train))) # 101.65820697412688
print("LGBM Tuned Model Test RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_test))) # 172.04075649052416

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_tuned_model,X)
