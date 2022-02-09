# A/B Testing

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


control_df = pd.read_excel(r"C:\Users\MehmetCanYildirim\Desktop\Veri_Bilimi_Okulu\Dersler\5.Hafta\DersOncesiNotlar\ab_testing.xlsx")
control_df = control_df[control_df.columns[0:4]]
control_df.head()

test_df = pd.read_excel(r"C:\Users\MehmetCanYildirim\Desktop\Veri_Bilimi_Okulu\Dersler\5.Hafta\DersOncesiNotlar\ab_testing.xlsx", sheet_name='Test Group')
test_df = test_df[test_df.columns[0:4]]
test_df

control_df.isnull().sum()
test_df.isnull().sum()

# Kontrol  = Maximum Bidding
# Test = Average Bidding

# Yeni yöntem ile eski yöntemin verilerini karşılaştıralım.
control_df["Impression"].sum() # 4068457.96270
test_df["Impression"].sum() # 4820496.47030
# Reklam görüntüleme sayılarının bariz bir şekilde arttığı gözleniyor.

control_df["Impression"].mean() # 101711.44906769728
test_df["Impression"].mean() # 120512.41175753452

control_df["Click"].sum() # 204026.29490
test_df["Click"].sum() # 158701.99043
# Görüntülenen reklama tıklanma sayılarında bir azalma söz konusu.

control_df["Click"].mean() # 5100.657372577278
test_df["Click"].mean() # 3967.54976080602

# Peki, önerilen yeni teklif yönteminin satın alınan ürün sayısı açısından ve kazanılan para açısından bir etkisi oldu mu ?
control_df["Purchase"].sum() # 22035.76235
test_df["Purchase"].sum() # 23284.24386

control_df["Purchase"].mean() # 550.89405
test_df["Purchase"].mean() # 582.10609
# Tıklanan reklamlar sonrası satın alınan ürün sayısında artış gözlenmiş.

control_df["Earning"].sum() # 76342.73199
test_df["Earning"].sum() # 100595.62930

control_df["Earning"].mean() # 1908.56829
test_df["Earning"].mean() # 2514.89073
# Satın alınan ürünler sonrasında elde edilen kazanç da yeni önerilen teklif sisteminde artmış gözüküyor.


# Peki, acaba buradaki artışlar şans eseri mi ortaya çıktı ? Yoksa gerçekten yeni önerilen teklif sistemi bizim için değerli olup kullanılmalı mı?
# Bir başka bakış açısıyla acaba yapmış olduğum bu hesaplar acaba sadece buradaki 40 günlük veriye özel mi böyle çıktı ? Yoksa başka bir 40 günlük dilimde de aynı sonucu mu elde edicem?
# İşte bu soruların çözümü için AB Testi yapmalıyız ve bu sayede iki grup arasında oluşan bu farklılıkların şans eseri mi ortaya çıktığını yoksa gerçekten bu değişikliğin şirket için değerli mi olduğunu anlayabileceğiz.

# Burada odaklanabileceğimiz bir çok nokta var. Ama bir şirket sahibi olarak ilk tercihim kazançlar ya da satın almalar üzerindeki oluşan bu değişikliğin şans eseri mi ortaya çıktığını sorgulamak olurdu.
# Bu yüzden satın almalar üzerinde ortalamasal olarak bir AB testi uygulayacağım.

# Gelelim uygulamamıza.
# İlk olarak Varsayımlar oluşturup bunların kontrolünü sağlamalıyız.
# Adım 1: Varsayım Kontrolü
# a) Normallik varsayımı
# b) Varyans Homojenliği

# a) Normallik Varsayımı
# Normallik varsayımı için hipotezimizi kuralım.
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

# Control dataseti için normallik varsayımı.
t_stat, pvalue = shapiro(control_df["Purchase"])
print("Test_stat = %.4f, p-value = %.4f" % (t_stat, pvalue))
# p değerimiz 0.5891 çıktı. Yani alfa parametremiz olan 0.05'den küçük değil. O zaman deriz ki H0 hipotezimiz reddedilemedi.
# Yani H0: Normallik varsayımı sağlanmıştır.

# Test dataseti için normallik varsayımı.
t_stat, pvalue = shapiro(test_df["Purchase"])
print("Test_stat = %.4f, p-value = %.4f" % (t_stat, pvalue))
# p değerimiz 0.1541 çıktı. Bu da demek oluyor ki alfa parametremizden küçük değil. Yani H0 hipotezimiz reddedilemedi.
# Yani H0: Normallik varsayımı sağlanmıştır.

# b) Varyans Homojenliği Varsayımı
# Varyans homojenliği için hipotezimi kuralım.
# H0: Varyans homojenliği varsayımı sağlanmaktadır.
# H1: Varyans homojenliği varsayımı sağlanmamaktadır.

# Control ve test datasetimdeki purchase değişkenin varyans homojenliğine bakalım.
t_stat,pvalue = levene(control_df["Purchase"],
                       test_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (t_stat, pvalue))
# p değerimiz 0.1083 çıktı. Bu da demektir ki alfa parametremiz p value'den daha küçük değil. Yani H0 hipotezimiz reddedilemedi.
# Yani H0: Varyans Homojenliği varsayımı sağlanmaktadır.

# Adım 2: Hipotezi uygulama.
# Varsayımlar sağlandığı için bağımsız iki örneklem t testi (parametrik testi) uygulayalım.
# Hipotezimizi tekrar yazalım.
# Oluşturulan 2 farklı teklif sisteminin ortalama satın almalar arasında istatistiksel olarak anlamlı bir fark var mıdır yoksa yok mudur ?
# H0: Yoktur.
# H1: Vardır.

t_stat, pvalue = ttest_ind(control_df["Purchase"],
                           test_df["Purchase"],
                           equal_var=True)
print("Test stat = %.4f, p-value = %.4f" % (t_stat,pvalue))
# p değerimiz 0.3493 çıktı. O yüzden H0 reddedilemedi. H0 yani istatistiksel olarak anlamlı bir fark yoktur reddedilemediği için bir fark yoktur.
# Buradan çıkarılacak sonuç ise, yapılan 40 günlük incelemelerde bu iki teklifin ortalama satın alımları arasında % 95 güven aralığı çerçevesinde istatistiksel olarak anlamlı bir fark yoktur.


# Bu AB testi için kullandığımız testlere gelecek olursak;
# - İlk olarak varsayımlarımızın sağlanıp sağlanmadığına bakmamız gerekmektedir.
# Bunun için ilk olarak normallik varsayımına baktık ve bunu shapiro testi ile sağladık. Daha sonra ikinci varsayımımız olan varyans homojenliğine bakmamız gerekti. Bunu da levene testi ile kolayca gerçekleştirdik.
# - İkinci olarak bu varsayımlar sağlandığı için direkt olarak iki örneklem t testi (Parametrik testi) uyguladık. Bunu da ttest_ind fonksiyonu ile kolayca gerçekleştirdik. Çıkan sonuca göre, yorumumuzu yapıp anlamlı bir fark olup olmadığını öğrenmiş olduk.

# Not: Eğer ki normallik varsayımımız sağlanmasaydı direkt olarak =====> Mannwhitneyu testi (Non-parametrik test)
#      Varyans homojenliği varsayımı sağlanmıyorsa =====> Bağımsız iki örneklem t testine arguman girilir. (parametrik test)



# Gelelim müşterimize verilecek olan önerilere.
# Öncelikle çıkarımımızı hatırlayalım. Yapılan 40 günlük incelemelere bakıldığında, önceden kullanılan MaximumBidding teklif yöntemi ile yeni sunulan AverageBidding teklif yöntemi arasında ortalama satın alımlar açısından
# istatistiksel olarak anlamlı bir fark yoktur.

# O zaman hazırlanan bu yeni teklif yöntemi boşa mı tasarlandı?
# Tabi ki hemen bu şekilde yorumlayamayız. Çünkü verisetimize baktığımız zaman 40 günlük bir inceleme söz konusu.Yapılan bu 40 günlük incelemenin hangi koşullarda, hangi zaman diliminde yapıldığı, sonuçları değerlendirirken oldukça önemli olacaktır.
# Ayrıca 40 günlük yapılan bu inceleme oldukça az gözlem sayısından oluşmaktadır. Belki son gözlemlere yaklaşıldığında faydalı olacağı düşünülen bir örüntü yakalanabilir,
# ve eğer bu varsayımı göz önünde bulundurmadan değerlendirmemizi yaparak yeni teklif sunma yöntemini iptal edersek buradan bir fayda sağlayamayız.
# Bu yüzden daha fazla sayıda gözlem yapmak önerilebilir.





