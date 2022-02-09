# Customer Segmentation using RFM

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)


import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Reading Dataset (Task-1)
path = "Ödevler/online_retail_II.xlsx"
df_ = pd.read_excel(path, sheet_name="Year 2010-2011")
df = df_.copy()

# Task-2
df.head()
df.tail()
df.describe().T
df.shape
df.info()
df.columns()
df.columns
df.nunique()
df["Price"].value_counts().head()
df.index

# Task-3-4
df.isnull().sum()
df.dropna(inplace=True)

# Task-5
df.nunique()

# Task-6
df.groupby("Description").agg({"Quantity": "count"})

# Task-7
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# Task-8
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

# Task-9
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()


# RFM Metrics Calculations

# Recency: The date the customer made the last purchase
# Frequency = Total number of sales made by the customer
# Monetary = Income earned by the customer after sales

# For recency, we have to know that the last purchase date and today date.
df["InvoiceDate"].max()
today_date = dt.datetime(2011,12,11)

RFM = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     "Invoice": lambda Invoice : Invoice.nunique(),
                                     "TotalPrice" : lambda TotalPrice : TotalPrice.sum()})

RFM.head()

RFM.columns = ["Recency", "Frequency", "Monetary"]

RFM["Monetary"] =RFM[(RFM["Monetary"].values > 0)]

# RFM Scores

RFM["Recency_score"] = pd.qcut(RFM["Recency"], 5, labels=[5, 4, 3, 2, 1])
RFM["Frequency_score"] = pd.qcut(RFM["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
RFM["Monetary_score"] = pd.qcut(RFM["Monetary"], 5, labels=[1, 2, 3, 4, 5])

RFM.head()

RFM["RFM_Scores"] = RFM["Recency_score"].astype(str) + RFM["Frequency_score"].astype(str)

RFM[RFM["RFM_Scores"] == "33"].head()  # Need_Attention

# Creating & Analysing RFM Segments

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

RFM['segment'] = RFM['RFM_Scores'].replace(seg_map, regex=True)
RFM.head()

# Lets choose 3 segments and give suggestions our manager for actions.

RFM[["segment","Recency","Frequency","Monetary"]].groupby("segment").agg(["mean", "count", "std"])
# (Optional) RFM.groupby("segment")["Monetary"].describe()

# Lets look at new_customers.
# New_customers are 42 people. Their average recency is 7.42 and their frequency of course 1. What is that mean?
# They purchased once and average 1 week.
# Therefore, for example we can apply a personal discount for them to get good relationship.
# Hence, they can purchase more frequently.

new_customers = pd.DataFrame()
new_customers["new_customers_id"] = RFM[RFM["segment"] == "new_customers"].index
new_customers.head()
new_customers.to_csv("new_customers.csv")

# Lets look at potential_loyalists class.
# When we look at this class, they are 484 people and their average recency value is 17.39.
# They purchased average twice.
# Then, what is that mean?
# They are good potential customer. So, we can apply some membership or loyalty programs.
# In another suggestion, we can maybe offer to them to purchase our upsell product.

potential_loyalists = pd.DataFrame()
potential_loyalists["potential_loyalists_id"] = RFM[RFM["segment"] == "potential_loyalists"].index
potential_loyalists.head()
potential_loyalists.to_csv("potential_loyalists.csv")

# Lets look at At Risk class.
# They are 593 people and their average recency is 153.7.
# It means that they purchased from your company but they started to forget you.
# So, we have to remind us as company for reconnectivity. We can offer renewals, or new products them.

at_risks = pd.DataFrame()
at_risks["at_Risk_id"] = RFM[RFM["segment"] == "at_Risk"].index
at_risks.head()
at_risks.to_csv("at_Risk.csv")


# Loyal Customers!

loyal_customers = pd.DataFrame()
loyal_customers["loyal_customer_id"] = RFM[RFM["segment"] == "loyal_customers"].index
loyal_customers.head()
loyal_customers.to_csv("loyal_customers.csv")




