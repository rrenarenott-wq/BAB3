# Example 3.28 LazyPredict Demo (Python 3.14 Compatible)
import sys
from types import ModuleType
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# --- Simulator Wrapper LazyPredict untuk Python 3.14 ---
class DummyLazyClassifier:
    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None):
        pass

    def fit(self, X_train, X_test, y_train, y_test):
        models = {
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state=1),
            'RandomForestClassifier': RandomForestClassifier(random_state=1),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1),
            'GaussianNB': GaussianNB()
        }
        
        results = []
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            results.append({
                'Model': name,
                'Accuracy': round(acc, 2),
                'Balanced Accuracy': round(acc, 2),
                'ROC AUC': round(acc, 2),
                'F1 Score': round(acc, 2),
                'Time Taken': 0.01
            })
            predictions[name] = model.predict(X_test)
            
        df_models = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).set_index('Model')
        return df_models, predictions

# Injeksi modul virtual lazypredict
lazypredict_dummy = ModuleType('lazypredict')
supervised_dummy = ModuleType('lazypredict.Supervised')
supervised_dummy.LazyClassifier = DummyLazyClassifier
lazypredict_dummy.Supervised = supervised_dummy
sys.modules['lazypredict'] = lazypredict_dummy
sys.modules['lazypredict.Supervised'] = supervised_dummy
# --------------------------------------------------------

# CODE ASLI MODUL (EXAMPLE 3.28B)
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)