
# Hybrid Recommender System: User-Based, Item-Based

import pandas as pd
pd.set_option('display.max_columns', 20)

# User-Movie Dataframe'ni oluşturma.
def create_user_movie_df():
    movie = pd.read_csv('DersSonrasıNotlar/movie.csv')
    rating = pd.read_csv('DersSonrasıNotlar/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
user_movie_df.head()

# Öneri yapılacak olan kullanıcıyı rastgele seçelim.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=0).values)

# kullanıcımız: 17800 id'ye sahip kişi.

# Öneri yapılacak olan kullanıcının izlediği filmleri belirleme

random_user_df = user_movie_df[user_movie_df.index == random_user] # user_movie_matrixinden random_user olarak belirlediğimiz kullanıcı için bütün filmleri seçtik.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() # Şimdi az önce oluşturduğumuz dataframe'den (seçtiğimiz kullanıcı için bütün filmler) seçtiğimiz kullanıcının izlediklerini yani NaN olmayanları seçelim.

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == 'Tommy Boy (1995)'] # Satırda kullanıcı, sütunda filmi seçerek doğrulama yapıyoruz. Bu filme 4 puan verdiği gözüküyor ve doğrulamak için kullanıcının izlediği filmler arasında bu film var mı kontrol edebiliriz.

len(movies_watched) # 181 adet film izlemiş.

# Aynı filmleri izleyen diğer kullanıcıların verisine ve id'lerine erişelim.

movies_watched_df = user_movie_df[movies_watched] # 181 film ve tüm idler.
movies_watched_df.head()

movies_watched_df.T.notnull().sum().reset_index()
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Bu noktada kendimize bir kriter belirlemeliyiz.
# Örneğin öneri yapacak olduğumuz kullanıcı 181 film izlediği için bu kriteri 150'den fazla film izleyen kişiler yapabiliriz.
# Ya da bu işlemi yüzdelik belirleyerek de greçekleştirebiliriz.

# user_movie_count[user_movie_count["movie_count"] == 181].count() ----- (Öneri yapacağımız kullanıcı ile aynı sayıda film izleyen kişi kaç kişi diye merak ettik diyelim.)(3 kişi)
# user_movie_count[user_movie_count["movie_count"] > 150].count()  ----- (150'yi limit seçerek o sayıdan fazla film izleyen ve ratingleyen kişileri ele alabiliriz.)(47 kişi)

# Yüzde ile yapmaya karar verdik diyelim. Percantage diye bir değişken tanımlayıp, yüzdeliği 80 seçelim.
percantage = (len(movies_watched) * 80) / 100  # 144.8'ine tekabül eder.
users_same_movies = user_movie_count[user_movie_count["movie_count"] > percantage]
users_same_movies = users_same_movies["userId"] # 87 kişi.

# users_same_movies: Kullanıcı ile yüzde 80'den fazla aynı filmi izleyen kişilerin idleri
# movies_watched_df: Kullanıcının izlediği filmlere diğer kişilerin tepkisi (izlemiş, beğenmiş, izlememiş, az beğenmiş vs)
# random_user: Seçtiğimiz kullanıcı
# random_id_movies[movies_watched]: Seçtiğimiz kullanıcının izlediği filmlere tepkisi.

# Öneri yapılacak kişi ile en benzer davranışta bulunan kullanıcıları bulalım.

user_based_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = user_based_df.T.corr().unstack().sort_values().drop_duplicates()  # user_based dataframe'indeki user id'lerin birbirleri ile olan correlasyonuna bakıp bunları dataframede kaydediyoruz.

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Kullanıcımız ile aynı filmi izleyen kişilerin korelasyonuna bakıp en yüksek olanlarına bakıyoruz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.4)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('DersSonrasıNotlar/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head(50) # userId, corr, movieId, rating dataframe'ni oluşturduk.

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# Weighted average recommendation score'un hesaplanması

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] # Weighted rating hesaplanması

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}) # MovieId kırılımında weighted_ratinglerin ortalamasını aldık.
recommendation_df = recommendation_df.reset_index()

rating = pd.read_csv('DersSonrasıNotlar/rating.csv')
movie = pd.read_csv('DersSonrasıNotlar/movie.csv')

user = 17800

# Öneri yapılacak olan kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sini bulmak.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
# 1095 id'li movie.

# User-Based

recommendation_df_5 = recommendation_df.sort_values("weighted_rating", ascending=False).head()

recommendation_df_5 = recommendation_df_5.merge(movie[["movieId", "title"]])
user_based_recommender = recommendation_df_5["title"]

#     17800 id'sine sahip bir kullanıcıya diğer kullanıcılar ile olan ilişkisi sonucu izlemesi önerilen 5 film.
#
#     Gunga Din (1939)
#     Godfather, The (1972)
#     Raging Bull (1980)
#     Ox-Bow Incident, The (1943)
#     Guardians of the Galaxy (2014)

# Item-Based
# 1095 id'li movie.

movie[movie["movieId"]==1095]["title"] # Glengarry Glen Ross (1992)
movie_name = "Glengarry Glen Ross (1992)"
movie_name = user_movie_df[movie_name]

item_based_recommendation = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10).iloc[1:6]
film_order = [1,2,3,4,5]
item_based_recommender = pd.Series(item_based_recommendation.index, index=film_order, name='Item-Based Izlenmesi Gereken Filmler')

#     1095 id'li filmi (Glengarry Glen Ross (1992)) izleyen kişilere film bazında önerilen 5 film.
#
#     This Is England (2006)
#     Insomnia (1997)
#     Barton Fink (1991)
#     Exit Through the Gift Shop (2010)
#     King of Comedy, The (1983)


# Hybrid Recommendation için iki öneri sistemini birleştirelim.

item_based_recommendation_df = pd.DataFrame(item_based_recommendation.reset_index()['title'])
user_based_recommendation_df = pd.DataFrame(recommendation_df_5["title"])
hybrid_recommendation = pd.merge(item_based_recommendation_df,user_based_recommendation_df,how='outer')

#                                title
# 0             This Is England (2006)
# 1                    Insomnia (1997)
# 2                 Barton Fink (1991)
# 3  Exit Through the Gift Shop (2010)
# 4         King of Comedy, The (1983)
# 5                   Gunga Din (1939)
# 6              Godfather, The (1972)
# 7                 Raging Bull (1980)
# 8        Ox-Bow Incident, The (1943)
# 9     Guardians of the Galaxy (2014)


# (Yukardaki gibi dataframe yapıp sonra merge etmek yerine concat da kullanabiliriz. Kullanım farklılıkları var, dökümandan okuyup amaca yönelik kullanmak lazım.)
# Nihai çıktımızı DataFrame olarak elde ettik.
