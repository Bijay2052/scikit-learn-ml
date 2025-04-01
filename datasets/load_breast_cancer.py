import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Breast Cancer dataset
breast_cancer = datasets.load_breast_cancer()

for attr in dir(breast_cancer):
    print('{attr}:'.format(attr=attr))
    print(getattr(breast_cancer, attr))
    print('\n\n')

# Convert to Pandas DataFrame
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Add target variable (species) to the DataFrame
df['target'] = breast_cancer.target

# Class distribution (0 = benign, 1 = malignant)
print(df['target'].value_counts())

# Show dataset info
print(df.head())
print(df.info())

# Summary statistics
print(df.describe())

# Visualize
sns.histplot(df, bins=2, kde=True)
# sns.pairplot(df, hue="target", diag_kind="kde")
plt.show()
