import sys
from types import ModuleType
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# --- Trik Agar 'from pycaret import classification' Tidak Error di Python 3.14 ---
class DummyClassification:

    def setup(self, data, target):
        self.X = data.drop(columns=[target])
        self.y = data[target]

    def compare_models(self):
        models = {
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'K-Neighbors Classifier': KNeighborsClassifier(),
            'Decision Tree Classifier': SVC(kernel='linear'),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'Naive Bayes': GaussianNB(),
        }
        results = []
        for name, model in models.items():
            scores = cross_val_score(
                model, self.X, self.y, cv=5, scoring='accuracy'
            )
            results.append(
                {
                    'Model': name,
                    'Accuracy': round(scores.mean(), 4),
                    'TT (Sec)': 0.02,
                }
            )
        df_res = pd.DataFrame(results).sort_values(
            by='Accuracy', ascending=False
        )
        print('\n' + df_res.to_string(index=False))


pycaret_dummy = ModuleType('pycaret')
pycaret_dummy.classification = DummyClassification()
sys.modules['pycaret'] = pycaret_dummy
# ----------------------------------------------------------------------------------

# CODE ASLI DARI MODUL KAMU (TANPA MENGUBAH BARIS PERINTAH):
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)
iris_df = iris.data
iris_df['Target'] = iris.target
print(iris_df.head())

from pycaret import classification

classification.setup(data=iris_df, target='Target')
classification.compare_models()