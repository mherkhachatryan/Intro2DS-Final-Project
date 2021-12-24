import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

target = "winner"

def categorize_other(data, col_name="B_HomeTown", threshold=0.3):
    """
    Add other category to data
    """
    limiting_qunatile = np.quantile(data[[col_name]].value_counts().values, threshold)
    g = data.groupby(col_name)[col_name].transform('size')
    data.loc[g <= limiting_qunatile, col_name] = 'Other'
    return data 

def fill_age_nan(data, prefix="B"):
    """Change prefix to R for different fighters"""
    data[f"{prefix}_Age"] = data[f"{prefix}_Age"].fillna(data.groupby(
        [f"{prefix}_Height", f"{prefix}_Weight"])[f"{prefix}_Age"].transform("mean"))
    return data

def fill_height_nan(data, prefix="B"):
    data[f"{prefix}_Height"].fillna(data[f"{prefix}_Height"].mean(), inplace=True) 
    return data


def binarize_column(data, prefix="B", q=4):
    data[f"{prefix}_Weight_bin"] = pd.qcut(data[f"{prefix}_Weight"], q=q)
    data.drop(f"{prefix}_Weight", axis=1, inplace=True)
    return data 



data = pd.read_csv("data.csv", usecols=list(range(100))+[894])
id_columns = ["B_Name", "B_ID"]

data.drop(id_columns, axis=1,  inplace=True)   # drop ID columns those are useless

cat_columns = data.select_dtypes(include="object").columns
numeric_columns = data.select_dtypes(exclude="object").columns

# add other category to categorical variables which are not target
for col_name in cat_columns:
    if col_name != target:
        data = categorize_other(data, col_name)
        
data = fill_age_nan(data, prefix="B")
data = fill_height_nan(data, prefix="B")

strikes_columns = [i for i in numeric_columns if i.find("Strikes") !=-1]
data[strikes_columns] = data[strikes_columns].fillna(data[strikes_columns].median())
grampling_columns = [i for i in numeric_columns if i.find("Grappling") !=-1]
data[grampling_columns] = data[grampling_columns].fillna(data[grampling_columns].median())


time_columns = [i for i in numeric_columns if i.find("Time") !=-1]
data[time_columns] = data[time_columns].fillna(data[time_columns].mean())

data = binarize_column(data, prefix="B", q=4)


dummies = pd.get_dummies(data[[cat for cat in cat_columns if cat!=target]])
data.drop([cat for cat in cat_columns if cat!=target],axis=1, inplace=True)

data = pd.concat([data, dummies], axis=1)

train, test = train_test_split(data, test_size=0.3, random_state=42)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
