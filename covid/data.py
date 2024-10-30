import requests
import pandas as pd
import numpy as np
import arviz as az
from copy import deepcopy

idx = pd.IndexSlice


def summarize_inference_data(trace, post_pred, observed):
    """ Обобщает объект inference_data в форму, которую мы публикуем на
        rt.live """
    trace = trace.posterior
    hdi_mass = 80
    lower_percentile = (100 - hdi_mass) / 2
    upper_percentile = 100 - lower_percentile

    # Используем numpy для расчета границ
    lower_hdi = np.percentile(trace["r_t"].values, lower_percentile, axis=(0, 1))
    upper_hdi = np.percentile(trace["r_t"].values, upper_percentile, axis=(0, 1))

    summary = pd.DataFrame(
        data={
            "mean": trace["r_t"].mean(["draw", "chain"]),
            "median": trace["r_t"].median(["chain", "draw"]),
            f"lower_{hdi_mass}": lower_hdi,
            f"upper_{hdi_mass}": upper_hdi,
            "infections": trace['infections'].mean(["draw", "chain"]),
            "test_adjusted_positive": trace['test_adjusted_positive'].mean(["draw", "chain"]),
            "obs": post_pred['obs'].mean(),
        },
        index=pd.Index(observed.index.values, name="date")
    )
    return summary


def load_covid_data(file_path, drop_states=False, filter_n_days_100=None):
    # Загрузка данных из CSV
    df = pd.read_csv(file_path)
    # Преобразование типа столбца Date_reported в формат даты
    df['Date_reported'] = pd.to_datetime(df['Date_reported'], format='%Y-%m-%d', errors='coerce')
    # Приведение данных к числовому типу и перевод в np.int64
    df['New_cases'] = pd.to_numeric(df['New_cases'], errors='coerce').fillna(0)
    df['Cumulative_cases'] = pd.to_numeric(df['Cumulative_cases'], errors='coerce').fillna(0)
    df['New_deaths'] = pd.to_numeric(df['Cumulative_cases'], errors='coerce').fillna(0)
    df['New_cases'] = df['New_cases'].apply(lambda x: int(x))
    df['Cumulative_cases'] = df['Cumulative_cases'].apply(lambda x: int(x))
    df['New_deaths'] = df['New_deaths'].apply(lambda x: int(x))
    # Удаление столбцов Country_code и WHO_region
    df.drop(columns=['Country_code', 'WHO_region'], inplace=True)
    # Удаление всех строк с любыми NaN
    df.dropna(inplace=True)
    # Удаление данных по штатам, если указано drop_states=True
    if drop_states:
        df = df[df['Country'].notna()]  # Убедимся, что у всех строк есть названия стран
    # Сортировка по странам и датам
    df = df.sort_values(by=['Country', 'Date_reported'])
    # Фильтрация по количеству дней с 100+ случаев (если filter_n_days_100 задан)
    if filter_n_days_100 is not None:
        # Создаем столбец 'days_since_100', который отслеживает дни с момента, когда случаи >= 100
        df['days_since_100'] = df.groupby('Country')['Cumulative_cases'].apply(
            lambda x: (x >= 100).cumsum() - 1  # Начинаем счет дней с момента, когда случаев стало >= 100
        )
        # Фильтрация по странам, где есть хотя бы filter_n_days_100 дней с 100+ случаев
        valid_countries = df.groupby('Country').filter(
            lambda group: group['days_since_100'].max() >= filter_n_days_100
        )
        df = df[df['Country'].isin(valid_countries['Country'])]
        # Удаление строк, где 'days_since_100' == -1
        df = df[df['days_since_100'] >= 0]
    return df


def get_country_data(df, country, num_rows=30):
    """
    Полученные данны для выбранной страны

    Параметры:
    - df (pd.DataFrame): Исходный датафрейм.
    - country (str): Название страны для фильтрации.
    - num_rows (int): Количество строк для отображения (по умолчанию 30).
    
    Возвращает:
    - dataframe: Отображает график с подтвержденными случаями по выбранной стране.
    """
    # Создаем глубокую копию исходного датафрейма
    df_copy = deepcopy(df)
    # Фильтруем данные по стране и количеству строк
    df_country = df_copy.loc[df_copy['Country'] == country].iloc[:num_rows]
    # Устанавливаем индекс для удобства отображения
    if 'days_since_100' in df_country.columns:
        df_country.set_index('days_since_100', inplace=True)
    # Переименование индексов
    df_country.rename(index=lambda x: x + 1, inplace=True)
    return df_country


def load_owid_covid_data(file_path, filter_n_days_100=None):
    # Загрузка данных из CSV
    df = pd.read_csv(file_path)
    df = pd.DataFrame([df['location'], df['date'], df['total_cases'], df['new_cases']]).T
    # # Преобразование типа столбца date в формат даты
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    # Приведение данных к числовому типу и перевод в np.int64
    df['new_cases'] = pd.to_numeric(df['new_cases'], errors='coerce').fillna(0)
    df['new_cases'] = df['new_cases'].apply(lambda x: np.int64(x))
    # Удаление всех строк с любыми NaN
    df.dropna(inplace=True)
    # Сортировка по странам и датам
    df = df.sort_values(by=['location', 'date'])
    # Фильтрация по количеству дней с 100+ случаев (если filter_n_days_100 задан)
    if filter_n_days_100 is not None:
        # Создаем столбец 'days_since_100', который отслеживает дни с момента, когда случаи >= 100
        df['days_since_100'] = df.groupby('location')['total_cases'].apply(
            lambda x: (x >= 100).cumsum() - 1  # Начинаем счет дней с момента, когда случаев стало >= 100
        )
        # Фильтрация по странам, где есть хотя бы filter_n_days_100 дней с 100+ случаев
        valid_countries = df.groupby('location').filter(
            lambda group: group['days_since_100'].max() >= filter_n_days_100
        )
        df = df[df['location'].isin(valid_countries['location'])]
        # Удаление строк, где 'days_since_100' == -1
        df = df[df['days_since_100'] >= 0]
    df.rename(columns={'location': 'country'}, inplace=True)
    df.rename(columns={'total_cases': 'confirmed'}, inplace=True)
    df.set_index('date', inplace=True)
    return df
