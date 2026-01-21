import pandas as pd
from sklearn.impute import SimpleImputer

# fetch the data
df = pd.read_json("data.json")
# display the data
print(f"Original Dataset {df}")
# creating a object of SimpleImputer class
imputer = SimpleImputer(strategy="mean")


# process 1 to fill missing values
# df['Age'] = imputer.fit_transform(df[["Age"]])
# print(df["Age"])


# process 2 to fill missing values

# step1 :- create a new dataframe
df_imputed = pd.DataFrame(imputer.fit_transform(df[["Age","Salary"]]),columns=["Age","Salary"])
# step2 :- assign the columns value in main dataframe's columns
df["Age"] = df_imputed["Age"]
df["Salary"] = df_imputed["Salary"]
print(df)