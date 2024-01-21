import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from tpot import TPOTClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shap
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

init = pd.read_csv('df_with_features.csv')
print(init)
# Замените 'dtp_severity' на реальное имя вашего столбца с типами ДТП
severity_counts = init['dtp_severity'].value_counts()

print(severity_counts)

df = pd.read_csv('df_with_features.csv')

selected_columns = ['year', 'month', 'region', 'temperature', 'atmospheric pressure', 'humidity',
                    'Direction of the wind', 'Wind speed', 'cloudiness', 'weather conditions',
                    'holiday', 'temperature_difference', 'through_0', 'season', 'hour', 'dtp_severity',
                    'day_of_week', 'day_of_mouth']

df_selected = df[selected_columns]
output_csv_path = 'ML.csv'
df_selected.to_csv(output_csv_path, index=False)

data = pd.read_csv('ML.csv')
print(data)

categorical_columns = ['region', 'Direction of the wind', 'weather conditions', 'dtp_severity']

label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(df[column])

data['through_0'] = df['through_0'].astype(int)

data.to_csv('to_ML.csv', index=False)

dt = pd.read_csv('to_ML.csv')
print(dt)

dt.info()

# Преобразование категориальных столбцов в числовой формат
cat_columns = ['year', 'month', 'temperature', 'atmospheric pressure', 'humidity', 'Wind speed', 'cloudiness',
               'temperature_difference']
dt[cat_columns] = dt[cat_columns].apply(lambda x: pd.factorize(x)[0].astype(np.float64))

# Выделение фичей и целевой переменной
features = ['year', 'month', 'region', 'temperature', 'atmospheric pressure', 'humidity',
            'Direction of the wind', 'Wind speed', 'cloudiness', 'weather conditions',
            'holiday', 'through_0', 'season', 'hour', 'temperature_difference',
            'day_of_week', 'day_of_mouth']
target = 'dtp_severity'

X = dt[features]
y = dt[target]

# CatBoostClassifier
catboost_model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.1, loss_function='MultiClass', cat_features=[2, 6, 13])


catboost_model.fit(X, y)
#
# # Предсказание на тестовом наборе
y_pred_catboost = catboost_model.predict(X)
#
# # Оценка производительности с использованием F1-метрики
f1_catboost = f1_score(y, y_pred_catboost, average='weighted')
#
# # Вывод отчета о классификации
print(classification_report(y, y_pred_catboost))
#
# # Вывод F1-метрики
print(f"F1 Score (CatBoost): {f1_catboost}")



# Получение предсказаний CatBoost на тестовом наборе
y_pred_catboost = catboost_model.predict(X)

# Создание матрицы ошибок
conf_matrix_catboost = confusion_matrix(y, y_pred_catboost)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_catboost, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Severity 0", "Severity 1", "Severity 2"],
            yticklabels=["Severity 0", "Severity 1", "Severity 2"])
plt.title("Confusion Matrix - CatBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
#
# Создание explainer
explainer = shap.Explainer(catboost_model)

# Получение shap_values
shap_values = explainer.shap_values(X)

# Вывод summary plot
shap.summary_plot(shap_values, X)

# # Создание объекта explainer для CatBoost
# explainer = shap.Explainer(catboost_model)
#
# # Расчет значений SHAP
# shap_values = explainer.shap_values(X)
#
# # Визуализация влияния каждой фичи
# shap.summary_plot(shap_values, X, plot_type="bar")
# Создадим объект Explainer
# Создаем объект Explainer
#explainer = shap.Explainer(catboost_model)

# Выбираем пример для локальной интерпретации
#example = X.sample(1, random_state=42)

# Рассчитываем значения SHAP для выбранного примера
#shap_values = explainer.shap_values(example)

# Визуализируем локальную интерпретацию
#shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=features, matplotlib=True)

# # Разделение на обучающий и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
#
# # RandomForestClassifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
# print("RandomForestClassifier:")
# print(classification_report(y_test, y_pred_rf))
# print(f"F1 Score: {f1_rf}\n")
#
# # GradientBoostingClassifier
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train, y_train)
# y_pred_gb = gb_model.predict(X_test)
# f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
# print("GradientBoostingClassifier:")
# print(classification_report(y_test, y_pred_gb))
# print(f"F1 Score: {f1_gb}")
#
# # Логистическая регрессия
# logreg_model = LogisticRegression(max_iter=1000, random_state=42)
# logreg_model.fit(X_train, y_train)
# y_pred_logreg = logreg_model.predict(X_test)
# f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')
#
# # Оценка производительности моделей
# print("Logistic Regression:")
# print(classification_report(y_test, y_pred_logreg))
# print(f"F1 Score (Logistic Regression): {f1_logreg}\n")
#
# # Преобразование столбцов с типом 'object' в числовой формат
# X_train = X_train.apply(pd.to_numeric, errors='coerce')
# X_test = X_test.apply(pd.to_numeric, errors='coerce')
#
# # XGBoost
# xgb_model = XGBClassifier(objective='multi:softmax', random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
#
#
#
# print("XGBoost:")
# print(classification_report(y_test, y_pred_xgb))
# print(f"F1 Score (XGBoost): {f1_xgb}")

