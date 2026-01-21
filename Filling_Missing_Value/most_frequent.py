import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_json('data.json')
# SimpleImputer expects:
# all strings
# OR all numbers
# Not a mix of None + str
df.replace({None:np.nan}, inplace=True)
print(f"Original data \n{df}")

imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df[['Age','Grade','Salary']]),columns=['Age','Grade','Salary'])
df['Age'] = df_imputed['Age']
df['Grade'] = df_imputed['Grade']
df['Salary'] = df_imputed['Salary']

print(f"After mode imputation \n{df}")