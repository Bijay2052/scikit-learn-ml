import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
print(dir(iris))
# print(iris.target_names)
# print(iris.feature_names)
# print(iris.DESCR)
# print(iris.data)
# print(iris.target)

# Convert to Pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target variable (species) to the DataFrame
df['target'] = iris.target
# df['species'] = iris.DESCR

# Show dataset info
print(df.head())
print(df.info())

# Visualize
sns.pairplot(df, hue="target", diag_kind="kde")
plt.show()
