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
