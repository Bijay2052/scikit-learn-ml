import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Create sample dataset
df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [np.nan, 5, 6, 7]})
print("Before Imputation:\n", df)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("After Imputation:\n", df_imputed)
