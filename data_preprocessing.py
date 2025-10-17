import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Загрузка данных из CSV файла"""
    return pd.read_csv('ObesityDataSet.csv')


def preprocess_data(df):
    """
    Предобработка данных для регрессии
    Целевая переменная - Weight (вес)
    """
    # Кодирование категориальных переменных
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                         'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Разделение на признаки и целевую переменную
    X = df.drop('Weight', axis=1)
    y = df['Weight']
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            X_train.columns, label_encoders, scaler)


def evaluate_model(y_true, y_pred, model_name):
    """
    Оценка модели и вывод метрик
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"___ Результаты {model_name} ___")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
    print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.4f}")
    print(f"Коэффициент детерминации (R²): {r2:.4f}")
    print()
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def plot_results(y_true, y_pred, model_name):
    """
    Построение графиков для визуализации результатов
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # График 1: Фактические vs Предсказанные значения
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Фактические значения')
    axes[0].set_ylabel('Предсказанные значения')
    axes[0].set_title(f'{model_name}: Фактические vs Предсказанные')
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Ошибки предсказания
    errors = y_pred - y_true
    axes[1].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Ошибка предсказания')
    axes[1].set_ylabel('Частота')
    axes[1].set_title(f'{model_name}: Распределение ошибок')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig