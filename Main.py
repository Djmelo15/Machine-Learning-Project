import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

class_count = df["num"].value_counts().sort_index()
# Class Distribution graph for numer of patients with and without Heart diseases
plt.figure(figsize=(8, 5))
plt.bar(class_count.index, class_count.values)
plt.xlabel("Heart Disease Presence")
plt.ylabel("Number of Patients")
plt.title("Class Distribution of Heart Diseases")
plt.xticks([0, 1], ["No Disease", "Diseases"])
plt.show()
#Histogram of age range of patients
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=10)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram of patients Age")
plt.show()
# Histogram of Chol of patients
plt.figure(figsize=(8, 5))
plt.hist(df["chol"], bins=10)

plt.xlabel("Cholesterol (mg/dl)")
plt.ylabel("Frequency")
plt.title("Histogram of patients Cholesterol level")
plt.show()
#Histogram of Resting blood pressure
plt.figure(figsize=(8,5))
plt.hist(df["trestbps"], bins=10)

plt.xlabel("Resting Blood Pressure (mm Hg)")
plt.ylabel("Frequency")
plt.title("Distribution of Resting Blood Pressure")
plt.show()

#Histogram of max BPM
plt.figure(figsize=(8,5))
plt.hist(df["thalach"], bins=10)

plt.xlabel("Maximum Heart Rate (bpm)")
plt.ylabel("Frequency")
plt.title("Distribution of Maximum Heart Rate")

plt.show()
#Heatmap to show which variables are associated with heart disease
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Heart Disease Dataset")
plt.show()

#Define features and target
X = df.drop("num", axis=1)
y = df["num"]
#Splting data into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scaling features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
