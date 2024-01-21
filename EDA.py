import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import missingno as msno
import numpy as np
import pandas_profiling
from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder

# EDA


# # Загрузка данных
# data_dtp = pd.read_excel('DataDTPкор.xlsx')
# print("DataDTPкор overview:")
# print(data_dtp)
#
# # Загрузка данных
# pogoda_data = pd.read_excel('Pogoda.xlsx')
# print("Pogoda overview:")
# print(pogoda_data)
#
#Загрузка данных
#data_dtp = pd.read_excel('DataDTPкор.xlsx')
#data_pogoda = pd.read_excel('Pogoda.xlsx')

# Дубликаты
#data_pogoda = data_pogoda.drop_duplicates('join_key')
#data_dtp = data_dtp.drop_duplicates('join_key')

# Объединение по столбцу join_key
#merged_data = pd.concat([data_pogoda.set_index('join_key'), data_dtp.set_index('join_key').add_prefix('dtp_')], axis=1, join='outer').reset_index()

# Сохранение данных в csv
#merged_data.to_csv('merged_data.csv', index=False)

data = pd.read_csv('merged_data.csv')
print(data)

# Подсчет уникальных значений в каждом столбце
unique_values_count = data.nunique()

# Вывод уникальных значений
print(unique_values_count)

# Увеличьте размеры графика, чтобы уместить все значения
plt.figure(figsize=(20, 10))

# Используйте heatmap для визуализации матрицы пропущенных значений
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')

# Отображение графика
plt.show()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(data['weather conditions'].head(20))

print(data.columns)

print(data.describe())

# Ограничим количество уникальных значений для "weather conditions" для улучшения читаемости
top_conditions = data['weather conditions'].value_counts().head(10).index
df_filtered = data[data['weather conditions'].isin(top_conditions)]

# Увеличим размеры графика
plt.figure(figsize=(14, 8))

# Создадим график
sns.countplot(x='month', hue='weather conditions', data=df_filtered, palette='viridis')

# Добавим подписи и легенду
plt.title('График погодных условий по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Количество ДТП')
plt.legend(title='Weather Conditions', bbox_to_anchor=(1.05, 1), loc='upper left')

# Покажем график
plt.show()

# Замените 'merged_data.csv' на ваш путь к файлу, если необходимо
merged_data = pd.read_csv('merged_data.csv')

# Группировка данных по тяжести ДТП и подсчет их количества
severity_counts = merged_data['dtp_severity'].value_counts()

# Создание графика
plt.figure(figsize=(10, 6))
severity_counts.plot(kind='bar', color='skyblue')
plt.title('Сравнение количества ДТП по тяжести')
plt.xlabel('Тяжесть ДТП')
plt.ylabel('Количество ДТП')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(merged_data['temperature'], bins=20, kde=True, color='skyblue')
plt.title('Распределение температуры')
plt.xlabel('Температура')
plt.ylabel('Частота')
plt.show()



merged_data['month'] = pd.Categorical(merged_data['month'], categories=range(1, 13), ordered=True)

plt.figure(figsize=(12, 8))
sns.boxplot(x='month', y='temperature', data=merged_data)
plt.title('Динамика изменения температуры по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Температура')
plt.show()

# Загрузим ваши данные из файла merged_data_filled.csv (или замените на другой файл, если используете другой)
df = pd.read_csv('merged_data_filled.csv')

# Группируем данные по температуре и считаем количество ДТП
temperature_dtp_count = df.groupby('temperature')['dtp_id'].count().reset_index()

# Построим график
plt.figure(figsize=(10, 6))
plt.plot(temperature_dtp_count['temperature'], temperature_dtp_count['dtp_id'], marker='o', linestyle='-', color='b')
plt.title('Зависимость температуры от количества ДТП')
plt.xlabel('Температура')
plt.ylabel('Количество ДТП')
plt.grid(True)
plt.show()

# Группируем данные по температуре и тяжести ДТП, считаем количество ДТП
grouped_data = df.groupby(['temperature', 'dtp_severity'])['dtp_id'].count().reset_index()

# Построим график
plt.figure(figsize=(12, 8))
sns.lineplot(x='temperature', y='dtp_id', hue='dtp_severity', data=grouped_data, marker='o')
plt.title('Зависимость количества ДТП по тяжести от температуры')
plt.xlabel('Температура')
plt.ylabel('Количество ДТП')
plt.grid(True)
plt.legend(title='Тяжесть ДТП')
plt.show()

# Подсчет пропущенных значений в объединенной таблице
merged_data_missing = merged_data.isnull().sum()

# Вывод результатов
print("Пропущенные значения в объединенной таблице:")
print(merged_data_missing)

# # Загрузим данные (замените 'your_data.csv' на имя вашего файла данных)
# df = pd.read_csv('merged_data.csv')
#
# # Определение категориальных столбцов
# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#
# # Label Encoding для категориальных столбцов
# label_encoder = LabelEncoder()
# for column in categorical_columns:
#     df[column] = label_encoder.fit_transform(df[column])
#
# # Определение столбцов, которые нужно заполнить
# columns_to_impute = df.columns
#
# # Применение KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=columns_to_impute)
#
# # Преобразование числовых значений в целые числа (если необходимо)
# df_imputed = df_imputed.round().astype(int)
#
# # Сохранение в CSV файл
# df_imputed.to_csv('KNN_imputed_data.csv', index=False)


# Загрузка данных
df = pd.read_csv('merged_data.csv')

# Определение столбцов для обработки
numeric_columns = ['year', 'month', 'temperature', 'atmospheric pressure', 'humidity', 'Wind speed', 'cloudiness']
categorical_columns = ['Direction of the wind', 'weather conditions']

# Заполнение числовых столбцов медианой
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Заполнение категориальных столбцов модой
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# Проверка результатов
print(df.isnull().sum())

# Сохранение обработанных данных
df.to_csv('filled_data.csv', index=False)

dd = pd.read_csv('processed_data.csv')
print(dd.head(20))

# Определение столбца 'weather conditions'
weather_conditions_column = 'weather conditions'

# Подсчет популярности уникальных значений
weather_conditions_popularity = df[weather_conditions_column].value_counts()

# Вывод результатов
print("Популярность уникальных значений в столбце 'weather conditions':")
print(weather_conditions_popularity)

# Подсчет пропущенных значений в объединенной таблице
rged_data_missing = dd.isnull().sum()

# Вывод результатов
print("Пропущенные значения в объединенной таблице:")
print(rged_data_missing)


# Определение столбца 'weather conditions'
weather_conditions_column = 'weather conditions'

# Подсчет популярности уникальных значений
weather_conditions_popularity = dd[weather_conditions_column].value_counts()

# Вывод результатов
print("Популярность уникальных значений в столбце 'weather conditions':")
print(weather_conditions_popularity)