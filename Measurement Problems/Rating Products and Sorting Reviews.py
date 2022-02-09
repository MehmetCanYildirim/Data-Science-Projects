

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\MehmetCanYildirim\Desktop\Veri_Bilimi_Okulu\Dersler\5.Hafta\DersOncesiNotlar\amazon_review.csv")
df

# Sorting Amazon Comments.

df["helpful_yes"].unique()
df["total_vote"].unique()
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df[df["helpful_yes"] ==1952] # 1952 helpful, 68 no helpful, 2020 total vote.


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["Wilson_lower_bound_score"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                       x["helpful_no"]), axis=1)

# "B007WTAJTO" ID'li elektronik ürün için ürün detay sayfasında görüntülenecek olan ilk 20 kullanıcı yorumu aşağıda sıralanmıştır.
WLB_sorted = df.sort_values("Wilson_lower_bound_score" ,ascending=False).head(20)



# Time-based average rating

df["day_diff"].describe(percentiles=[.2, .4, .6, .8]).T  # Maximum 1064 gün, minimum 1 gün, median 431.

# 5 farklı zaman diliminde gelen değerlendirmelerin zamana bağlı olarak ağırlıklı ortalamalarını alalım.

def time_based_weighted_average(df, w1=30, w2=25, w3=20, w4=15, w5=10):
    return df.loc[df["day_diff"] <= 248, "overall"].mean() * w1 / 100 + \
           df.loc[(df["day_diff"] > 248) & (df["day_diff"] <= 361), "overall"].mean() * w2 / 100 + \
           df.loc[(df["day_diff"] > 361) & (df["day_diff"] <= 497), "overall"].mean() * w3 / 100 + \
           df.loc[(df["day_diff"] > 497) & (df["day_diff"] <= 638), "overall"].mean() * w4 / 100 + \
           df.loc[df["day_diff"] > 638, "overall"].mean() * w5 / 100


time_based_weighted_average(df) # 4.6189
df["overall"].mean() # 4.5875

# 5 farklı zaman dilimine karşı gelen değerlendirmeler ile hesaplanan ağırlıklı ortalama, datasetinde verilen değerlendirmelere göre daha fazla çıktı.
# Çünkü zamana bağlı yapılan değerlendirmelerde yapılan yorumların zamanlarının, zamana bağlı olarak aynı ağırlığı vermesi beklenemez.
