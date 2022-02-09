# Task.1

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Ders Öncesi Notlar/titanic.csv")

# Before
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
cat_summary(df, "Sex", True)


# We are gonna add some argument to function cat_summary()
# After
def cat_summary(dataframe, col_name, null=False, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if null:
        print(dataframe[col_name].isnull().values.any())
        if dataframe[col_name].isnull().values.any() > 0:
            print(dataframe[col_name].isnull().sum())
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
    categoric_list = []
    [categoric_list.append(col) for col in df.columns if df[col].dtype == "O"]

    return categoric_list

cat_summary(df, "Cabin", null=True, plot=False)


#######################################################################################################################


# Task.2
# Now we define the function by writing its docstring.
def cat_summary(dataframe, col_name, null=False, plot=False):
    """
    Prints number of each categorical variables of the dataframe and their percentage.

    Parameters
    ----------
    dataframe: (types)
        Dataset that are using.
    col_name:
        Categorical variables that are found in dataset.
    null: (Default = False)
        Checks for NaN variables. If there is, it gives how many of them.
    plot: (Default = False)
        It countplots the categorical variable according to dataframe.

    Returns
    -------
    categoric_list is a list that contains the categorical variables in the dataset.

    Examples
    -------
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv("Ders Öncesi Notlar/titanic.csv")
    cat_summary(df, col_name, null=False, plot=False):
    """



# Task.3
# Task.3.1
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
persona = pd.read_csv("Ders Öncesi Notlar/persona.csv")

# 1 - General View (Question - 1)

persona.head()
persona.columns
persona.index
persona.info
persona.describe().T
persona.isnull().values.any()
persona.dtypes
persona.quantile([0, 0.25, 0.5, 0.75, 1]).T


# 2 - Examine Categorical Variables (Question - 2)

categorical_var = [i for i in persona.columns if persona[i].dtype == "O"]
persona["SOURCE"].nunique()  # 2
persona["SEX"].nunique()    # 2
persona["COUNTRY"].nunique()   # 6

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(persona,"SOURCE",True)
cat_summary(persona,"SEX",True)
cat_summary(persona,"COUNTRY",True)

# 3 - Examine Numerical Variables (Question - 3, 4, 5)

numerical_var = [i for i in persona.columns if persona[i].dtype != "O"]
persona[numerical_var].describe().T

persona[numerical_var].hist(bins=30)
plt.show()

sns.boxplot(x=persona["PRICE"])
plt.show()

sns.boxplot(x=persona["AGE"])
plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

num_summary(persona, "PRICE", True)
num_summary(persona, "AGE", True)





# Question 6
def target_summation_with_categoric(dataframe, target, categorical_col):
    return pd.DataFrame({"PRICE": dataframe.groupby(categorical_col)[target].sum()})

target_summation_with_categoric(persona, "PRICE", "COUNTRY")

# Question 7 count
def target_count_with_categoric(dataframe, target, categorical_col):
    return pd.DataFrame({"PRICE": dataframe.groupby(categorical_col)[target].count()})
target_summation_with_categoric(persona, "PRICE", "SOURCE")

# Question 8, 9, 10
def target_summary_with_cat(dataframe, target, categorical_col):
    return pd.DataFrame({"PRICE": dataframe.groupby(categorical_col)[target].mean()})

target_summary_with_cat(persona, "PRICE", "COUNTRY")
target_summary_with_cat(persona, "PRICE", "SOURCE")
target_summary_with_cat(persona, "PRICE", ["COUNTRY", "SOURCE"])




# Task.3.2

agg_df = target_summary_with_cat(persona, "PRICE", ["COUNTRY", "SOURCE", "SEX", "AGE"])

# Task.3.3

agg_df = agg_df.sort_values("PRICE", ascending=False)

# Task.3.4

agg_df = agg_df.reset_index()


# Task.3.5

age = agg_df["AGE"].sort_values()
age.unique()
age.describe()  # We should consider the min and max values, 15 and 66, respectively. # (15-25, 26-35, 36-45, 46-55, 56-66)
labels = ["15_25","26_35","36_45","46_55","56_66"]

AGE_CAT = pd.cut(age, bins=[14,25,35,45,55,66], labels=labels)
agg_df["AGE_CAT"] = AGE_CAT

# Task.3.6

# Alternative
# agg_df["CUSTOMER_LEVEL_BASED"] = agg_df["COUNTRY"].map(str) + '_' + agg_df["SOURCE"].map(str) + '_' + agg_df["SEX"].map(str) + '_' + agg_df["AGE_CAT"].map(str) (?)
# agg_df.drop(["COUNTRY","SOURCE","SEX","AGE","AGE_CAT"], axis=1, inplace=True)

agg_df["customer_level_based"] = [val[0].upper() + '_' + val[1].upper() + '_' + val[2].upper() + '_' + val[5] for val in agg_df.values]  #+ val[5] (?)
agg_df = agg_df.drop(["COUNTRY","SOURCE","SEX","AGE","AGE_CAT"], axis=1)

agg_df = agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()

# Task.3.7

SEGMENT = pd.qcut(agg_df["PRICE"], 4, labels=["D","C","B","A"])
agg_df["SEGMENT"] = SEGMENT.values


check_df = agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]})
check_df = check_df.reset_index()

SEGMENT_c = check_df.loc[check_df["SEGMENT"] == "C"]

# Task.3.8

new_user_1 = "TUR_ANDROID_MALE_15_25"
segment_1 = agg_df[agg_df["customer_level_based"]==new_user_1]

new_user_2 = "FRA_ISO_FEMALE_20_25"
segment_2 = agg_df[agg_df["customer_level_based"]==new_user_2]
