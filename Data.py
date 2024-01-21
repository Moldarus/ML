import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

ta = pd.read_csv('merged_data.csv')
print(ta)

# Подсчет уникальных значений в каждом столбце
unique_values_count = ta.nunique()

# Вывод уникальных значений
print(unique_values_count)

# Увеличьте размеры графика, чтобы уместить все значения
plt.figure(figsize=(20, 10))

# Используйте heatmap для визуализации матрицы пропущенных значений
sns.heatmap(ta.isnull(), cbar=False, cmap='viridis')

# Отображение графика
plt.show()

#df = pd.read_csv('merged_data.csv')

# Удаление строк с нулевыми значениями
#df_cleaned = df.dropna()

# Вывод информации о новом DataFrame
#print("Размер исходного DataFrame:", df.shape)
#print("Размер DataFrame после удаления строк с нулевыми значениями:", df_cleaned.shape)

# Сохранение очищенных данных в новый файл
#df_cleaned.to_csv('cleaned_data.csv', index=False)

data = pd.read_csv('cleaned_data.csv')
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

data_daytime = data[data['dtp_light'] == 'Светлое время суток']

print(data_daytime)

# Преобразование столбца datetime в формат datetime
data_daytime['datetime'] = pd.to_datetime(data_daytime['datetime']).copy()

# Создание объекта-словаря с праздничными днями для России
rus_holidays = holidays.Russia()

# Создание нового столбца 'Праздник' и заполнение его значениями
data_daytime.loc[:, 'holiday'] = data_daytime['datetime'].apply(lambda x: 1 if x in rus_holidays else 0)

# Вывод DataFrame с новым столбцом 'Праздник'
print(data_daytime)

data_daytime.to_csv('daytime_with_holiday.csv', index=False)

dt = pd.read_csv('daytime_with_holiday.csv')
print(dt)

df = pd.read_csv('daytime_with_holiday.csv')
# Преобразование столбца datetime в формат datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Сортировка DataFrame по столбцу datetime
df = df.sort_values(by='datetime')

# Рассчет переменной: Температурный перепад
df['temperature_difference'] = df.groupby(['year', df['datetime'].dt.day])['temperature'].diff()

# Рассчет переменной: Переход через 0
df['through_0'] = (df['temperature'] * df['temperature'].shift(1)) < 0

# Рассчет переменных: Сезон, час, день недели, день месяца
df['season'] = df['datetime'].dt.month % 12 // 3 + 1
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_mouth'] = df['datetime'].dt.day

# Сохранение DataFrame с новыми переменными в новый CSV файл
df.to_csv('df_with_features.csv', index=False)

all = pd.read_csv('df_with_features.csv')
print(all)

all['temperature_difference'].fillna(0, inplace=True)

all.to_csv('df_with_features.csv', index=False)

all = pd.read_csv('df_with_features.csv')
print(all)

# Проверка NaN значений по всем столбцам
nan_check = all.isnull().sum()

# Вывод результатов
print(nan_check)