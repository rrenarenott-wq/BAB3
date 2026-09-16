import pandas as pd
from matplotlib import pyplot
from sklearn.datasets import load_breast_cancer

# 1. Load Breast Cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# 2. Pilih 4 fitur yang diminta: radius, size/area, texture, dan smoothness
selected_features = ['mean radius', 'mean area', 'mean texture', 'mean smoothness']
df_selected = df[selected_features]

# 3. Plot histogram untuk 4 fitur tersebut (Exercise 3.4)
df_selected.hist(bins=20, figsize=(10, 8))
pyplot.suptitle('Exercise 3.4: Histograms of Radius, Size (Area), Texture, and Smoothness')
pyplot.tight_layout()
pyplot.show()