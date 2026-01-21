import pandas as pd
df = pd.read_json('data.json')

print(f"Original Data \n{df}")
df_f = df.ffill()
df['Age'] = df['Age'].bfill()
print(f"\nAfter forward fill \n{df_f}")
print(f"\nAfter backward fill in a particular column \n {df}")
