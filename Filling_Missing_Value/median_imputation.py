import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_json('data.json')
print(f"Original Data {df}")
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df[["Age","Salary"]]),columns=["Age","Salary"])
df["Age"] = df_imputed["Age"]
df["Salary"] = df_imputed["Salary"]
print(f"After median imputation\n{df}")
