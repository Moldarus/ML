import pandas as pd
import sklearn
import tpot
from imblearn import __version__ as imblearn_version
from catboost import __version__ as catboost_version
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np

print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
print("tpot version:", tpot.__version__)
print("imbalanced-learn version:", imblearn_version)
print("catboost version:", catboost_version)
print("seaborn version:", sns.__version__)
print("matplotlib version:", plt.matplotlib.__version__)
print("shap version:", shap.__version__)
print("numpy version:", np.__version__)