import numpy as np
import pandas as pd
#because the data has no column names we need to add them
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
# this is to test if we can see the first 5 rows of the data with the column names we will remove later
df = pd.read_csv("Data/heart.csv", names=columns)

#replaces all "?"/missing data to actual missing values
df.replace("?", pd.NA, inplace=True)
df = df.apply(pd.to_numeric)
#We are using Binary classification so this converts anything above 0 into a 1 which mean Heart disease
def convert(x):
    if x > 0:
        return 1
    else:
        return 0
# this shows the count of patients with and without heart disease
df["num"] = df["num"].apply(convert)
print(df["num"].value_counts())

# this shows where the missing values are we (will remove later)
print(df.isnull().sum())
# Replace the missing values with the median values of the data because the mean is sensitive to outliers, and so we don't drop rows
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())
#I did this again to show that there are no missing values anymore (will probably remove)
print(df.isnull().sum())
